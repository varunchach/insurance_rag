"""OCP qualification stages. Gold fixtures are consumed only by evaluate()."""
import argparse, base64, collections, hashlib, importlib.metadata as md
import json, logging, os, re, shutil, ssl, subprocess, sys, time, urllib.request
from pathlib import Path

def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

def digest(value):
    return hashlib.sha256(value).hexdigest()

def norm(value):
    return re.sub(r"\s+", " ", value or "").strip().casefold()

def tokens(value):
    return re.findall(r"\w+|[^\w\s]", value.casefold())

def api_resource(kind, name):
    sa = Path("/var/run/secrets/kubernetes.io/serviceaccount")
    api = "https://" + os.environ["KUBERNETES_SERVICE_HOST"] + ":" + os.environ.get("KUBERNETES_SERVICE_PORT_HTTPS", "443")
    namespace = sa.joinpath("namespace").read_text().strip()
    req = urllib.request.Request(api + "/api/v1/namespaces/" + namespace + "/" + kind + "/" + name,
        headers={"Authorization": "Bearer " + sa.joinpath("token").read_text().strip()})
    with urllib.request.urlopen(req, context=ssl.create_default_context(cafile=str(sa/"ca.crt")), timeout=30) as response:
        return json.load(response)

def storage(config):
    import boto3
    from botocore.config import Config
    cred = api_resource("secrets", config["storage_secret"])["data"]
    return boto3.client("s3", endpoint_url=config["storage_endpoint"], region_name="us-east-1",
        aws_access_key_id=base64.b64decode(cred["accesskey"]).decode(),
        aws_secret_access_key=base64.b64decode(cred["secretkey"]).decode(),
        config=Config(signature_version="s3v4", s3={"addressing_style":"path"}))

def fetch(client, config, key, target, expected=None):
    client.download_file(config["bucket"], key, str(target))
    actual = digest(Path(target).read_bytes())
    if expected and actual != expected:
        raise ValueError("Artifact hash mismatch: " + key)
    return actual

def get_source(config, output):
    source = output/"source.pdf"
    fetch(storage(config), config, config["source_key"], source, config["source_sha256"])
    return source

def get_kit_file(client, config, name, target):
    return fetch(client, config, config["kit_prefix"] + name, target, config["kit_hashes"][name])

def runtime_manifest(config, stage):
    return {"stage":stage, "source_sha256":config["source_sha256"],
        "runtime_image": config["runtime_image"],
        "versions":{n:md.version(n) for n in ["docling","docling-core","docling-ibm-models","pypdf","pdfplumber","PyMuPDF","torch","torchvision","transformers","kfp"]},
        "model_manifest":json.loads(Path("/opt/rag/model-manifest.json").read_text()),
        "configuration": config, "release_state":"review_required", "execution":"RHOAI-managed pipeline on OCP"}

def native(config, out):
    source = get_source(config, out)
    client = storage(config)
    script = out/"profile_pdf.py"
    get_kit_file(client, config, "profile_pdf.py", script)
    with (out/"native-run.log").open("w") as log:
        subprocess.run([sys.executable,str(script),str(source),"--output",str(out)],stdout=log,stderr=subprocess.STDOUT,check=True)
    manifest = runtime_manifest(config, "native")
    manifest["page_count"] = json.loads((out/"profile.json").read_text())["page_count"]
    dump(out/"manifest.json",manifest)
    print(json.dumps({"stage":"native","pages":manifest["page_count"]}),flush=True)

def parse(config, out):
    import fitz
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
    from docling.datamodel.accelerator_options import AcceleratorOptions, AcceleratorDevice
    from docling.document_converter import DocumentConverter, PdfFormatOption
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Requested GPU is unavailable; refusing a silent CPU fallback")
    print(json.dumps({"gpu":torch.cuda.get_device_name(0),"cuda":torch.version.cuda}),flush=True)
    source = get_source(config,out)
    logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(out/"conversion.log"),logging.StreamHandler()])
    options = PdfPipelineOptions(artifacts_path=Path("/opt/rag/models"))
    options.do_ocr = False
    options.do_table_structure = True
    options.table_structure_options.mode = TableFormerMode.ACCURATE
    options.table_structure_options.do_cell_matching = config["cell_matching"]
    options.accelerator_options = AcceleratorOptions(num_threads=config["cpu_threads"],device=AcceleratorDevice.CUDA)
    options.layout_batch_size = config.get("layout_batch_size",4)
    options.table_batch_size = config.get("table_batch_size",4)
    options.enable_remote_services = False
    options.do_picture_classification = False
    options.do_picture_description = False
    options.do_code_enrichment = False
    options.do_formula_enrichment = False
    start = time.monotonic()
    gpu_log=(out/"gpu-utilization.csv").open("w")
    gpu_monitor=subprocess.Popen(["nvidia-smi","--query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw",
        "--format=csv,noheader,nounits","-lms","200"],stdout=gpu_log,stderr=subprocess.DEVNULL)
    try:
        result = DocumentConverter(format_options={InputFormat.PDF:PdfFormatOption(pipeline_options=options)}).convert(source,raises_on_error=False)
    finally:
        gpu_monitor.terminate()
        gpu_monitor.wait(timeout=10)
        gpu_log.close()
    doc = result.document
    doc.save_as_json(out/"document.json")
    doc.save_as_markdown(out/"document.md")
    parser_config=options.model_dump(mode="json")
    config_id=digest(json.dumps(parser_config,sort_keys=True).encode())
    elements,tables,pages=materialize(config,doc,out,config_id)
    manifest=runtime_manifest(config,"parse")
    manifest.update({"conversion_status":result.status.value,"elapsed_seconds":round(time.monotonic()-start,2),
        "pages_in_source":len(pages),"pages_in_docling":sorted(doc.pages),"elements":len(elements),"tables":len(tables),
        "errors":[e.model_dump(mode="json") for e in result.errors],
        "parser_options":parser_config,"parser_configuration_sha256":config_id,
        "canonical_sha256":digest((out/"document.json").read_bytes()),
        "gold_annotations_consumed":False})
    import csv
    readings=[]
    for row in csv.reader((out/"gpu-utilization.csv").read_text().splitlines()):
        try: readings.append(float(row[1]))
        except (ValueError,IndexError): pass
    manifest["gpu_utilization"]={"sample_interval_ms":200,"samples":len(readings),
        "mean_pct":round(sum(readings)/len(readings),1) if readings else None,"peak_pct":max(readings) if readings else None}
    dump(out/"manifest.json",manifest)
    print(json.dumps({k:manifest[k] for k in ["conversion_status","elapsed_seconds","pages_in_source","elements","tables"]}),flush=True)
    if result.status.value not in ("success","partial_success"):
        raise RuntimeError("Docling conversion did not produce a usable candidate")

def materialize(config, doc, out, config_id):
    artifact_id=digest((out/"document.json").read_bytes())
    def stable(ref):
        return digest((config["source_sha256"]+"|"+config_id+"|"+artifact_id+"|"+ref).encode())
    def provenance(item):
        return [p.model_dump(mode="json") for p in item.prov]
    from docling_core.types.doc import ContentLayer
    import fitz
    source=out/"source.pdf"
    elements=[]
    tables=[]
    from preservation import assign_headings
    hierarchy=assign_headings(doc)
    headings=[]
    pages={i:[] for i in range(1,len(fitz.open(source))+1)}
    for item, depth in doc.iterate_items(with_groups=False, traverse_pictures=True, included_content_layers=set(ContentLayer)):
        label=str(item.label.value)
        headings=hierarchy.get(item.self_ref,[])
        text=getattr(item,"text","")
        marker=getattr(item,"marker","")
        if marker and not text.lstrip().startswith(marker):text=marker+" "+text
        prov=provenance(item)
        element={"id":stable(item.self_ref),"ref":item.self_ref,"label":label,
            "parent_ref":item.parent.cref if item.parent else None,
            "depth":depth,"text":text,"provenance":prov,"content_layer":item.content_layer.value,
            "heading_candidates":list(headings),"scope_status":"unverified"}
        if label=="table":
            cells=[c.model_dump(mode="json") for c in item.data.table_cells]
            rows=[[None]*item.data.num_cols for _ in range(item.data.num_rows)]
            for c in cells:
                r,cx=c["start_row_offset_idx"],c["start_col_offset_idx"]
                if r<len(rows) and cx<len(rows[r]):
                    rows[r][cx]=c["text"]
            bboxes=[]
            for p in item.prov:
                box=p.bbox.to_top_left_origin(doc.pages[p.page_no].size.height)
                bboxes.append({"page":p.page_no,"bbox":[box.l,box.t,box.r,box.b]})
            headers=[c for c in cells if c.get("column_header")]
            for c in cells:
                c["header_candidates"]=[{
                    "text":h["text"],"row_start":h["start_row_offset_idx"],"column_start":h["start_col_offset_idx"],
                    "column_end":h["end_col_offset_idx"]
                } for h in headers if h["start_row_offset_idx"]<=c["start_row_offset_idx"]
                   and max(h["start_col_offset_idx"],c["start_col_offset_idx"])<min(h["end_col_offset_idx"],c["end_col_offset_idx"])]
                c["literal_state"]="blank" if c["text"]=="" else "NA" if c["text"].strip().upper()=="NA" else "Nil" if c["text"].strip().lower()=="nil" else "text"
                c["header_status"]="parser_candidate_unverified"
            table={"id":element["id"],"ref":item.self_ref,"shape":[item.data.num_rows,item.data.num_cols],
                "regions":bboxes,"cells":rows,"raw_cells":cells,"heading_candidates":list(headings),
                "scope_status":"unverified","header_status":"parser_candidate_unverified"}
            tables.append(table)
            text="\n".join(" ".join(x or "" for x in r) for r in rows)
            element["text"]=text
        elements.append(element)
        for p in prov:
            pages.setdefault(p["page_no"],[]).append(text)
    known={e["ref"] for e in elements}
    for item in doc.texts:
        if item.self_ref in known:
            continue
        prov=provenance(item)
        elements.append({"id":stable(item.self_ref),"ref":item.self_ref,"label":item.label.value,
            "parent_ref":item.parent.cref if item.parent else None,"depth":None,"text":item.text,
            "provenance":prov,"content_layer":item.content_layer.value,"heading_candidates":[],
            "scope_status":"unverified","tree_status":"not_in_traversal"})
        for p in prov:
            pages.setdefault(p["page_no"],[]).append(item.text)
    dump(out/"elements.json",elements)
    dump(out/"tables.json",tables)
    dump(out/"page-text.json",{str(k):"\n".join(v) for k,v in pages.items()})
    return elements,tables,pages

def reconcile(config, parsed, out):
    from docling_core.types.doc import DoclingDocument
    from preservation import repair_document, build_context_graph, VERSION
    shutil.copytree(parsed,out,dirs_exist_ok=True)
    canonical=out/"document.json"
    original=out/"original-document.json"
    if (parsed/"original-document.json").exists():
        shutil.copyfile(original,canonical)
    else:
        shutil.copyfile(canonical,original)
    original_hash=digest(canonical.read_bytes())
    doc=DoclingDocument.load_from_json(canonical)
    audit=repair_document(doc,out/"source.pdf",config.get("preservation",{}))
    doc.save_as_json(canonical)
    doc.save_as_markdown(out/"document.md")
    config_id=digest((original_hash+digest(canonical.read_bytes())+json.dumps(audit["options"],sort_keys=True)).encode())
    elements,tables,pages=materialize(config,doc,out,config_id)
    by_ref={e["ref"]:e for e in elements}
    for record in audit["native_supplements"]:
        by_ref[record["ref"]]["native_recovery"]=record
    dump(out/"elements.json",elements)
    graph=build_context_graph(elements,tables,{p:page.size.height for p,page in doc.pages.items()},config.get("preservation",{}))
    for record in audit["native_supplements"]:
        for table_ref in record["table_candidates"]:
            graph["edges"].append({"source":table_ref,"target":record["ref"],
                "kind":"native_recovery","status":"cell_assignment_unverified"})
    dump(out/"context-graph.json",graph)
    dump(out/"preservation-audit.json",audit)
    manifest=json.loads((out/"manifest.json").read_text())
    manifest.update({"elements":len(elements),"tables":len(tables),"preservation_version":VERSION,
        "original_canonical_sha256":original_hash,"canonical_sha256":digest(canonical.read_bytes()),
        "derivation_sha256":config_id,"reused_gpu_parse":True,"derivation_configuration":config})
    dump(out/"manifest.json",manifest)
    print(json.dumps({"stage":"reconcile","canonical_parse_reused":True,"elements":len(elements),
        "table_repairs":len(audit["table_repairs"]),"native_supplements":len(audit["native_supplements"]),
        "list_markers":len(audit["list_markers"]),"context_edges":len(graph["edges"])}),flush=True)

def chunk(config, parsed, out):
    from docling_core.types.doc import DoclingDocument
    from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
    from docling_core.transforms.chunker.hierarchical_chunker import HierarchicalChunker, ChunkingDocSerializer, ChunkingSerializerProvider
    from docling_core.transforms.serializer.markdown import MarkdownParams, MarkdownTableSerializer
    from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
    from transformers import AutoTokenizer
    doc=DoclingDocument.load_from_json(parsed/"document.json")
    element_map={x["ref"]:x for x in json.loads((parsed/"elements.json").read_text())}
    class MDTableSerializerProvider(ChunkingSerializerProvider):
        def get_serializer(self,doc):
            return ChunkingDocSerializer(doc=doc,table_serializer=MarkdownTableSerializer(),params=MarkdownParams(compact_tables=True))
    raw_tokenizer=AutoTokenizer.from_pretrained("/opt/rag/models/tokenizer",local_files_only=True,trust_remote_code=False)
    raw_tokenizer.model_max_length=10**9
    strategies={"hierarchical":HierarchicalChunker(serializer_provider=MDTableSerializerProvider())}
    for budget in config["chunk_budgets"]:
        # Reserve the tokenizer's special tokens from the content budget.
        tokenizer=HuggingFaceTokenizer(tokenizer=raw_tokenizer,max_tokens=budget-raw_tokenizer.num_special_tokens_to_add(False))
        strategies["hybrid_"+str(budget)]=HybridChunker(tokenizer=tokenizer,repeat_table_header=True,
            serializer_provider=MDTableSerializerProvider(),merge_peers=True)
    from preservation import expand_refs, split_text
    graph=json.loads((parsed/"context-graph.json").read_text()) if (parsed/"context-graph.json").exists() else {"edges":[],"options":{}}
    expansion_budget=config.get("preservation",{}).get("expansion_tokens",4096)
    derivation=json.loads((parsed/"manifest.json").read_text()).get("canonical_sha256","")
    summary={}
    for name,chunker in strategies.items():
        output=[]
        for i,c in enumerate(chunker.chunk(dl_doc=doc)):
            context=chunker.contextualize(c)
            refs=[x.self_ref for x in c.meta.doc_items]
            resolved=[element_map[r] for r in refs if r in element_map]
            body_pages=sorted({p["page_no"] for e in resolved for p in e["provenance"]})
            header_refs=[]
            for e in resolved:
                header_refs.extend(h["ref"] for h in e["heading_candidates"] if h["text"] in (c.meta.headings or []))
            context_refs=sorted(set(refs+header_refs))
            context_pages=sorted({p["page_no"] for r in context_refs if r in element_map for p in element_map[r]["provenance"]})
            count=len(raw_tokenizer.encode(context,add_special_tokens=True))
            budget=int(name.split("_")[1]) if name.startswith("hybrid_") else None
            payload={"strategy":name,"text":c.text,"contextualized_text":context,"element_refs":refs,
                "element_ids":[e["id"] for e in resolved],"context_refs":context_refs,
                "body_pages":body_pages,"context_pages":context_pages,"headings":c.meta.headings or [],
                "tokens_with_special_tokens":count,"budget":budget,"fits_budget":budget is None or count<=budget,
                "provenance_resolved":len(resolved)==len(refs) and bool(refs),"metadata":c.meta.model_dump(mode="json"),
                "parent_expansion_refs":context_refs,"release_state":"review_required"}
            payload["id"]=digest((config["source_sha256"]+"|"+name+"|"+str(i)+"|"+context+"|"+"|".join(refs)).encode())
            output.append(payload)
        # Emit standalone chunks for body text that a serializer excludes (often headings).
        covered={r for record in output for r in record["context_refs"]}
        def represented(element):
            seen=set()
            while element and element["ref"] not in seen:
                if element["ref"] in covered:return True
                seen.add(element["ref"])
                element=element_map.get(element.get("parent_ref"))
            return False
        budget=int(name.split("_")[1]) if name.startswith("hybrid_") else None
        for e in element_map.values():
            if e.get("content_layer")!="body" or not e["text"].strip() or represented(e):continue
            # Split text while preserving source references and the actual tokenizer contract.
            pieces=split_text(e["text"],raw_tokenizer,budget)
            for piece in pieces:
                count=len(raw_tokenizer.encode(piece,add_special_tokens=True))
                pages=sorted({p["page_no"] for p in e["provenance"]})
                output.append({"strategy":name,"text":piece,"contextualized_text":piece,
                    "element_refs":[e["ref"]],"element_ids":[e["id"]],"context_refs":[e["ref"]],
                    "body_pages":pages,"context_pages":pages,"headings":[],
                    "tokens_with_special_tokens":count,"budget":budget,"fits_budget":budget is None or count<=budget,
                    "provenance_resolved":True,"metadata":{"preservation_fallback":True},
                    "release_state":"review_required"})
        for record in output:
            refs,expanded,deferred=expand_refs(record["context_refs"],graph,element_map,raw_tokenizer,
                expansion_budget,config.get("preservation",{}).get("expansion_hops",2))
            record.update({"parent_expansion_refs":refs,"expanded_text":expanded,
                "expanded_tokens":len(raw_tokenizer.encode(expanded,add_special_tokens=True)),
                "expansion_budget":expansion_budget,"deferred_expansion_refs":deferred,
                "expansion_status":"structural_candidate_unverified"})
            record["id"]=digest((derivation+"|"+name+"|"+record["contextualized_text"]+"|"+"|".join(record["element_refs"])).encode())
        dump(out/(name+".json"),output)
        summary[name]={"chunks":len(output),"max_tokens":max((c["tokens_with_special_tokens"] for c in output),default=0),
            "oversized":sum(not c["fits_budget"] for c in output),"unresolved_provenance":sum(not c["provenance_resolved"] for c in output),
            "chunks_sha256":digest(json.dumps(output,sort_keys=True,ensure_ascii=False).encode())}
    dump(out/"summary.json",summary)
    dump(out/"manifest.json",runtime_manifest(config,"chunk"))
    print(json.dumps(summary),flush=True)

def evaluate(config, native_dir, parsed, chunks, out):
    client=storage(config)
    for name in ["gold_annotations.json","benchmark_config.json","benchmark.py"]:
        get_kit_file(client,config,name,out/name)
    gold=json.loads((out/"gold_annotations.json").read_text())
    assert gold["source_sha256"]==config["source_sha256"]
    sys.path.insert(0,str(out))
    import benchmark
    # Repeat original baseline in the pinned OCP runtime for a valid comparison.
    with (out/"baseline-benchmark.log").open("w") as log:
        subprocess.run([sys.executable,str(out/"benchmark.py"),str(native_dir/"source.pdf"),
            "--baseline",str(native_dir/"extractions.json"),"--gold",str(out/"gold_annotations.json"),
            "--config",str(out/"benchmark_config.json"),"--output",str(out/"baseline")],
            stdout=log,stderr=subprocess.STDOUT,check=True)
    baseline=json.loads((out/"baseline/benchmark_results.json").read_text())
    baseline_text=json.loads((native_dir/"extractions.json").read_text())
    doc_text=json.loads((parsed/"page-text.json").read_text())
    tables=json.loads((parsed/"tables.json").read_text())
    elements=json.loads((parsed/"elements.json").read_text())
    by_ref={e["ref"]:e for e in elements}
    cfg=json.loads((out/"benchmark_config.json").read_text())
    candidate=[{"page":p["page"],"tables":{"docling_accurate":[]}} for p in baseline_text]
    for t in tables:
        for region in t["regions"]:
            candidate[region["page"]-1]["tables"]["docling_accurate"].append({"bbox":region["bbox"],"cells":t["cells"]})
    table_scores=benchmark.table_scores(candidate,gold,cfg)
    dump(out/"table-scores.json",table_scores)
    queue=[]
    for t in tables:
        queue.append({"kind":"scope_header_validation","severity":"review","table_id":t["id"],
            "pages":[r["page"] for r in t["regions"]],"reason":"Parser headings and header flags are candidates; product, date, exception and header relationships have not been independently validated."})
    for score in table_scores:
        if not score["shape_pass"] or any(not r["pass"] for r in score["rows"]):
            queue.append({"kind":"annotated_table_mismatch","severity":"high","fixture":score["id"],"page":score["page"],
                "actual_shape":score["actual_shape"],"expected_shape":score["expected_shape"],
                "failed_rows":[r["expected"] for r in score["rows"] if not r["pass"]]})
    from preservation import lexical
    page_scores=[]
    for p in baseline_text:
        expected=collections.Counter(tokens(p["pypdf"]))
        actual=collections.Counter(tokens(doc_text.get(str(p["page"]),"")))
        missing=expected-actual
        numeric=[{"token":k,"missing_occurrences":v} for k,v in missing.items() if any(ch.isdigit() for ch in k)]
        overlap=sum((expected&actual).values())
        page_scores.append({"page":p["page"],"native_chars":len(p["pypdf"]),"docling_chars":len(doc_text.get(str(p["page"]),"")),
            "token_multiset_recall":overlap/max(1,sum(expected.values())),
            "token_multiset_f1":2*overlap/max(1,sum(expected.values())+sum(actual.values())),
            "missing_native_numeric_tokens":numeric})
        normalized_missing=collections.Counter(lexical(p["pypdf"]))-collections.Counter(lexical(doc_text.get(str(p["page"]),"")))
        page_scores[-1]["missing_normalized_numeric_tokens"]=[{"token":k,"missing_occurrences":v} for k,v in normalized_missing.items() if k.isdigit()]
        if page_scores[-1]["missing_normalized_numeric_tokens"]:
            queue.append({"kind":"native_numeric_omission_candidate","severity":"high","page":p["page"],"missing":numeric,
                "reason":"Independent parser disagreement; inspect source before treating as a confirmed extraction error."})
    dump(out/"page-comparison.json",page_scores)
    # Geometric cross-page candidates are kept separate and never silently joined.
    continuation=[]
    page_heights={p["page"]:p["height"] for p in json.loads((native_dir/"profile.json").read_text())["pages"]}
    for left in tables:
        for l in left["regions"]:
            for right in tables:
                for r in right["regions"]:
                    if r["page"]!=l["page"]+1 or left["shape"][1]!=right["shape"][1]:
                        continue
                    lb,rb=l["bbox"],r["bbox"]
                    overlap=max(0,min(lb[2],rb[2])-max(lb[0],rb[0]))/max(1,min(lb[2]-lb[0],rb[2]-rb[0]))
                    if lb[3]>=page_heights[l["page"]]*0.72 and rb[1]<=page_heights[r["page"]]*0.23 and overlap>=0.8:
                        continuation.append({"left":left["id"],"right":right["id"],"pages":[l["page"],r["page"]],
                            "status":"review_required","reason":"Adjacent edge tables with compatible columns and horizontal extent; semantic scope and row continuation unverified."})
    dump(out/"continuation-candidates.json",continuation)
    queue.extend({"kind":"cross_page_continuation","severity":"high",**c} for c in continuation)
    chunk_results=[]
    # Bind every anchor to a source page and contributing element before checking co-location.
    def eligible(anchor,refs):
        return any(anchor["page"] in [p["page_no"] for p in by_ref[r]["provenance"]]
            and norm(anchor["text"]) in norm(by_ref[r]["text"]) for r in refs if r in by_ref)
    summary=json.loads((chunks/"summary.json").read_text())
    for strategy in summary:
        records=json.loads((chunks/(strategy+".json")).read_text())
        referenced={r for c in records for r in c["context_refs"]}
        def ancestor_referenced(element):
            visited=set()
            ref=element["ref"]
            while ref and ref not in visited:
                if ref in referenced:
                    return True
                visited.add(ref)
                ref=by_ref.get(ref,{}).get("parent_ref")
            return False
        unrepresented=[e for e in elements if e.get("content_layer")=="body" and e.get("text","").strip()
            and not ancestor_referenced(e)]
        summary[strategy]["body_elements_without_chunk_lineage"]=len(unrepresented)
        summary[strategy]["pages_with_unrepresented_body_elements"]=sorted({
            p["page_no"] for e in unrepresented for p in e["provenance"]})
        if strategy=="hybrid_"+str(config["primary_chunk_budget"]):
            queue.extend({"kind":"body_element_without_chunk_lineage","severity":"high","element_id":e["id"],
                "ref":e["ref"],"pages":[p["page_no"] for p in e["provenance"]],
                "parent_ref":e["parent_ref"],
                "reason":"Canonical body text has no direct or ancestor reference in emitted chunks."} for e in unrepresented)
        for g in gold["evidence_dependencies"]:
            direct=[]
            expanded=[]
            for c in records:
                refs=c["context_refs"]
                if all(norm(a["text"]) in norm(c["contextualized_text"]) and eligible(a,refs) for a in g["anchors"]):
                    direct.append(c["id"])
                # Parent expansion explicitly restores full originating elements, not arbitrary neighboring pages.
                if all(eligible(a,c["parent_expansion_refs"]) and norm(a["text"]) in norm(c.get("expanded_text","")) for a in g["anchors"]):
                    expanded.append(c["id"])
            chunk_results.append({"id":g["id"],"strategy":strategy,"evidence_colocated":bool(direct),
                "parent_expansion_colocated":bool(expanded),"matching_chunks":direct[:3],
                "matching_expanded_chunks":expanded[:3]})
            if not direct and strategy=="hybrid_"+str(config["primary_chunk_budget"]):
                queue.append({"kind":"evidence_dependency_not_colocated","severity":"high","fixture":g["id"],"anchors":g["anchors"],
                    "parent_expansion_colocated":bool(expanded)})
        for c in records:
            if not c["fits_budget"] or not c["provenance_resolved"]:
                queue.append({"kind":"chunk_contract_violation","severity":"high","chunk_id":c["id"],
                    "strategy":strategy,"tokens":c["tokens_with_special_tokens"],"budget":c["budget"]})
    dump(out/"chunk-scores.json",chunk_results)
    dump(out/"review-queue.json",queue)
    def table_summary(scores):
        return {"exact_dimensions":sum(x["shape_pass"] for x in scores),"tables":len(scores),
            "complete_rows":sum(r["pass"] for x in scores for r in x["rows"]),
            "row_checks":sum(len(x["rows"]) for x in scores)}
    scorecard={"release_state":"review_required","indexing_performed":False,"source_sha256":config["source_sha256"],
        "runtime_image":config["runtime_image"],"docling":table_summary(table_scores),
        "baseline":{s:table_summary([x for x in baseline["table_scores"] if x["strategy"]==s])
            for s in ["lines","thin_rules_1.0"]},
        "docling_by_split":{s:table_summary([x for x in table_scores if x["split"]==s]) for s in ["development","validation"]},
        "source_pages":len(baseline_text),"docling_pages":len(json.loads((parsed/"manifest.json").read_text())["pages_in_docling"]),
        "pages_without_docling_text":[p["page"] for p in page_scores if p["docling_chars"]==0],
        "pages_with_numeric_disagreements":sum(bool(p["missing_native_numeric_tokens"]) for p in page_scores),
        "pages_with_normalized_numeric_disagreements":sum(bool(p["missing_normalized_numeric_tokens"]) for p in page_scores),
        "accounting_includes_furniture_and_picture_children":True,
        "chunks":{s:{**v,"evidence_colocated":sum(x["evidence_colocated"] for x in chunk_results if x["strategy"]==s),
            "parent_expansion_colocated":sum(x["parent_expansion_colocated"] for x in chunk_results if x["strategy"]==s),
            "evidence_cases":len(gold["evidence_dependencies"])} for s,v in summary.items()},
        "review_items":len(queue),"continuation_candidates":len(continuation),
        "candidate_name":"Docling accurate with native preservation",
        "normalized_numeric_method":"Alphanumeric splitting; punctuation retained. Multiset coverage is not value-to-cell validation.",
        "expansion_method":"Bounded structural graph; candidate associations remain unverified",
        "limitations":["One PDF with provisional annotations; no unseen-document validation.",
            "Shape and normalized complete-row checks do not validate merged cells or business relationships.",
            "Numeric disagreements are review candidates, not verified errors.",
            "Page-bound anchor co-location is not retrieval accuracy or answer correctness.",
            "Product scope, effective-date precedence, exceptions and cross-page joins remain unvalidated.",
            "No embedding model, vector database, LLM, or retrieval pipeline deployed."]}
    dump(out/"scorecard.json",scorecard)
    lines=["# Docling ingestion qualification","", "Release state: **review required**. Nothing indexed.","",
        "| Candidate | Exact table dimensions | Complete row strings |","|---|---:|---:|"]
    for name,score in [*scorecard["baseline"].items(),(scorecard["candidate_name"],scorecard["docling"])]:
        lines.append(f"| {name} | {score['exact_dimensions']}/{score['tables']} | {score['complete_rows']}/{score['row_checks']} |")
    lines+=["","| Chunker | Chunks | Evidence co-location | With bounded context expansion | Oversized |","|---|---:|---:|---:|---:|"]
    for name,score in scorecard["chunks"].items():
        lines.append(f"| {name} | {score['chunks']} | {score['evidence_colocated']}/{score['evidence_cases']} | {score['parent_expansion_colocated']}/{score['evidence_cases']} | {score['oversized']} |")
    lines+=["",f"Pages: {scorecard['docling_pages']}/{scorecard['source_pages']}. Review items: {len(queue)}.",""]+["- "+x for x in scorecard["limitations"]]
    (out/"qualification-report.md").write_text("\n".join(lines),encoding="utf-8")
    dump(out/"manifest.json",runtime_manifest(config,"evaluate"))
    print(json.dumps(scorecard),flush=True)

def ablate(config,out):
    import gc, torch
    for matching in [True,False]:
        case=out/("matching_on" if matching else "matching_off")
        case.mkdir(parents=True,exist_ok=True)
        case_config=dict(config,cell_matching=matching)
        parse(case_config,case)
        gc.collect()
        torch.cuda.empty_cache()

def compare_ablation(config,native_dir,parsed,out):
    summaries={}
    for name in ["matching_on","matching_off"]:
        case_out=out/name
        chunks=case_out/"chunks"
        report=case_out/"report"
        chunks.mkdir(parents=True,exist_ok=True)
        report.mkdir(parents=True,exist_ok=True)
        case_config=dict(config,cell_matching=(name=="matching_on"))
        repaired=case_out/"repaired"
        repaired.mkdir(parents=True,exist_ok=True)
        reconcile(case_config,parsed/name,repaired)
        chunk(case_config,repaired,chunks)
        evaluate(case_config,native_dir,repaired,chunks,report)
        score=json.loads((report/"scorecard.json").read_text())
        manifest=json.loads((parsed/name/"manifest.json").read_text())
        summaries[name]={"scorecard":score,"parse_seconds":manifest["elapsed_seconds"],
            "gpu_utilization":manifest["gpu_utilization"],"layout_batch_size":manifest["parser_options"]["layout_batch_size"],
            "table_batch_size":manifest["parser_options"]["table_batch_size"]}
    dump(out/"ablation-summary.json",summaries)
    print(json.dumps(summaries),flush=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("stage",choices=["native","parse","reconcile","chunk","evaluate","ablate","compare-ablation"])
    ap.add_argument("--config",required=True,type=Path)
    ap.add_argument("--out",required=True,type=Path)
    ap.add_argument("--native",type=Path)
    ap.add_argument("--parsed",type=Path)
    ap.add_argument("--chunks",type=Path)
    args=ap.parse_args()
    cfg=json.loads(args.config.read_text())
    args.out.mkdir(parents=True,exist_ok=True)
    if args.stage=="ablate":ablate(cfg,args.out)
    elif args.stage=="compare-ablation":compare_ablation(cfg,args.native,args.parsed,args.out)
    elif args.stage=="native":native(cfg,args.out)
    elif args.stage=="parse":parse(cfg,args.out)
    elif args.stage=="reconcile":reconcile(cfg,args.parsed,args.out)
    elif args.stage=="chunk":chunk(cfg,args.parsed,args.out)
    else:evaluate(cfg,args.native,args.parsed,args.chunks,args.out)

if __name__=="__main__":
    main()
