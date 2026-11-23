"""Geometry and native-text preservation. Never reads benchmark annotations."""
import collections, hashlib, json, re
from pathlib import Path

VERSION = 1
DEFAULTS = {"rule_width": 1.0, "table_match_iou": .95,
            "native_margin": 2.0, "continuation_edge": .72,
            "continuation_top": .23, "expansion_tokens": 4096,
            "expansion_hops": 2, "note_gap": 12.0,
            "nested_table_overlap": .5, "missing_table_iou": .1, "minimum_text_retention": .95,
            "native_table_overlap": .8, "note_overlap": .8, "continuation_width_tolerance": .04}

def validated_options(options=None):
    options=options or {}
    unknown=set(options)-set(DEFAULTS)
    if unknown:raise ValueError("Unknown preservation settings: "+", ".join(sorted(unknown)))
    result={**DEFAULTS,**options}
    for name,value in result.items():
        if not isinstance(value,(int,float)) or isinstance(value,bool) or value<=0:
            raise ValueError(name+" must be a positive number")
        if name not in ("rule_width","native_margin","note_gap","expansion_tokens","expansion_hops") and value>1:
            raise ValueError(name+" must be at most 1")
    for name in ("expansion_tokens","expansion_hops"):
        if not isinstance(result[name],int):raise ValueError(name+" must be an integer")
    return result

def lexical(text):
    """Split alphanumerics consistently; retain signs, decimal points and punctuation."""
    return re.findall(r"[^\W\d_]+|\d+|[^\w\s]", text.casefold(), re.UNICODE)

def counts(text):
    return collections.Counter(lexical(text))

def intersection(a, b):
    return max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))

def area(a):
    return max(0, a[2]-a[0])*max(0, a[3]-a[1])

def iou(a, b):
    v = intersection(a, b)
    return v / max(1, area(a)+area(b)-v)

def region(item, doc):
    if not getattr(item,"prov",None):
        for child in getattr(item,"children",[]):
            found=region(child.resolve(doc),doc)
            if found:return found
        return None
    p = item.prov[0]
    b = p.bbox.to_top_left_origin(doc.pages[p.page_no].size.height)
    return p.page_no, [b.l, b.t, b.r, b.b]

def numbered_level(text):
    if re.match(r"^(?:appendix|annex|chapter|part)\s+(?:\d+|[IVXLCDM]+)\b",text,re.I):
        return 1
    if re.match(r"^[A-Z][.)]\s+", text):
        return 1
    match = re.match(r"^(\d+(?:\.\d+)*)(?:[.)]|\s)\s*\D", text)
    return len(match[1].split("."))+1 if match else None

def assign_headings(doc):
    """Numbered hierarchy plus consecutive heading nesting; ownership remains unverified."""
    from docling_core.types.doc import ContentLayer
    stack, result, numbered, previous_heading = [], {}, 1, False
    for item, _ in doc.iterate_items(traverse_pictures=True, included_content_layers=set(ContentLayer)):
        if item.label.value in ("title", "section_header"):
            explicit = numbered_level(item.text)
            if explicit is not None:
                level = explicit
                numbered = level
                reason = "explicit_numbering"
            else:
                level = stack[-1]["level"]+1 if previous_heading and stack else numbered+1
                reason = "consecutive_heading" if previous_heading else "unnumbered_section_candidate"
            while stack and stack[-1]["level"] >= level:
                stack.pop()
            stack.append({"ref": item.self_ref, "text": item.text, "level": level, "basis": reason})
            previous_heading = True
        elif item.content_layer.value == "body":
            previous_heading = False
        result[item.self_ref] = [dict(h) for h in stack]
    return result

def ruled_data(table):
    """Use actual ruling boundaries, including full-width merged cells."""
    from docling_core.types.doc import TableData, TableCell, BoundingBox, CoordOrigin
    cells = sorted(set(c for row in table.rows for c in row.cells if c is not None))
    xs = sorted({round(c[k], 3) for c in cells for k in (0, 2)})
    ys = sorted({round(c[k], 3) for c in cells for k in (1, 3)})
    rows = table.extract()
    text_by_box = {tuple(box): (rows[ri][ci] or "") for ri, row in enumerate(table.rows)
                   for ci, box in enumerate(row.cells) if box is not None}
    output = []
    for box in cells:
        c0, c1 = xs.index(round(box[0], 3)), xs.index(round(box[2], 3))
        r0, r1 = ys.index(round(box[1], 3)), ys.index(round(box[3], 3))
        output.append(TableCell(text=text_by_box[box], row_span=r1-r0, col_span=c1-c0,
            start_row_offset_idx=r0, end_row_offset_idx=r1,
            start_col_offset_idx=c0, end_col_offset_idx=c1,
            bbox=BoundingBox(l=box[0], t=box[1], r=box[2], b=box[3], coord_origin=CoordOrigin.TOPLEFT),
            column_header=r0 == 0))
    return TableData(num_rows=len(ys)-1, num_cols=len(xs)-1, table_cells=output)

def repair_document(doc, source, options=None):
    import pdfplumber
    from docling_core.types.doc import ProvenanceItem, BoundingBox, CoordOrigin, DocItemLabel, ContentLayer
    opts = validated_options(options)
    audit = {"version": VERSION, "options": opts, "gold_annotations_consumed": False,
             "table_repairs": [], "native_supplements": [], "list_markers": []}
    for item in doc.texts:
        marker = getattr(item, "marker", "")
        if marker and not item.text.lstrip().startswith(marker):
            before = item.text
            audit["list_markers"].append({"ref": item.self_ref, "marker": marker, "original_text": before})
    with pdfplumber.open(source) as pdf:
        for page in pdf.pages:
            filtered = page.filter(lambda obj: obj.get("object_type") != "rect"
                                   or min(obj.get("width", 0), obj.get("height", 0)) <= opts["rule_width"])
            for candidate in filtered.find_tables():
                data = ruled_data(candidate)
                # Reject empty separator columns and nested sub-tables.
                occupied = {c.start_col_offset_idx for c in data.table_cells if c.text.strip()}
                if data.num_cols < 2 or len(occupied) != data.num_cols:
                    continue
                matches = [(iou(candidate.bbox, box), item, box) for item in doc.tables
                           if (reg := region(item, doc)) and reg[0] == page.page_number
                           for box in [reg[1]]]
                overlap, item, box = max(matches, key=lambda v: v[0], default=(0, None, None))
                nested = any(intersection(candidate.bbox, b)/max(1,area(candidate.bbox)) > opts["nested_table_overlap"] for _, _, b in matches)
                reason = None
                if not nested and overlap < opts["missing_table_iou"]:
                    reason = "ruled_table_missing_from_layout"
                elif overlap >= opts["table_match_iou"] and data.num_rows == item.data.num_rows and data.num_cols < item.data.num_cols:
                    old = counts(" ".join(c.text for c in item.data.table_cells))
                    new = counts(" ".join(c.text for c in data.table_cells))
                    retained = sum((old & new).values()) / max(1, sum(old.values()))
                    missing_numbers = {k:v for k,v in (old-new).items() if k.isdigit()}
                    if retained >= opts["minimum_text_retention"] and not missing_numbers:
                        reason = "ruling_boundaries_resolve_spurious_columns"
                if not reason:
                    continue
                before = item.data.model_dump(mode="json") if reason.startswith("ruling_") else None
                if before is None:
                    b = candidate.bbox
                    item = doc.add_table(data=data, prov=ProvenanceItem(page_no=page.page_number,
                        bbox=BoundingBox(l=b[0], t=b[1], r=b[2], b=b[3], coord_origin=CoordOrigin.TOPLEFT), charspan=(0,0)))
                    # Place new root item next to its source location; leave original elements intact.
                    added = doc.body.children.pop()
                    position = len(doc.body.children)
                    for idx, ref in enumerate(doc.body.children):
                        target = ref.resolve(doc)
                        loc = region(target, doc)
                        if loc and (loc[0], loc[1][1]) > (page.page_number, b[1]):
                            position = idx
                            break
                    doc.body.children.insert(position, added)
                else:
                    item.data = data
                audit["table_repairs"].append({"ref": item.self_ref, "page": page.page_number,
                    "bbox": list(candidate.bbox), "reason": reason, "original_data": before,
                    "shape": [data.num_rows, data.num_cols], "source": "pdfplumber_native_ruled_cells",
                    "header_status": "first_row_candidate_unverified"})
            # Recover individual native lines whose words are missing in their local parse region.
            # Do not replace model cells with unrelated text or infer a cell assignment.
            originals = [it for it in [*doc.texts, *doc.tables] if any(p.page_no == page.page_number for p in it.prov)]
            spans = []
            for it in originals:
                if it.label.value == "table":
                    for cell in it.data.table_cells:
                        if cell.bbox:
                            b = cell.bbox.to_top_left_origin(page.height)
                            spans.append(([b.l,b.t,b.r,b.b], cell.text))
                else:
                    loc = region(it, doc)
                    if loc: spans.append((loc[1], (getattr(it,"marker","")+" "+it.text).strip()))
            for line in page.extract_text_lines(layout=False):
                b = [line["x0"],line["top"],line["x1"],line["bottom"]]
                pad = opts["native_margin"]
                expanded = [b[0]-pad,b[1]-pad,b[2]+pad,b[3]+pad]
                nearby = " ".join(text for box,text in spans if intersection(expanded,box)>0)
                missing = counts(line["text"]) - counts(nearby)
                words_missing = {k:v for k,v in missing.items() if any(c.isalnum() for c in k)}
                if not words_missing:
                    continue
                item = doc.add_text(label=DocItemLabel.TEXT, text=line["text"],
                    prov=ProvenanceItem(page_no=page.page_number,
                        bbox=BoundingBox(l=b[0],t=b[1],r=b[2],b=b[3],coord_origin=CoordOrigin.TOPLEFT),
                        charspan=(0,len(line["text"]))))
                added = doc.body.children.pop()
                position = len(doc.body.children)
                for idx, ref in enumerate(doc.body.children):
                    loc = region(ref.resolve(doc),doc)
                    if loc and (loc[0],loc[1][1]) > (page.page_number,b[1]):
                        position=idx;break
                doc.body.children.insert(position,added)
                candidates = [t.self_ref for t in doc.tables if (loc:=region(t,doc))
                              and loc[0]==page.page_number and intersection(b,loc[1])/max(1,area(b))>opts["native_table_overlap"]]
                audit["native_supplements"].append({"ref":item.self_ref,"page":page.page_number,
                    "bbox":b,"text":line["text"],"missing_local_tokens":words_missing,
                    "table_candidates":candidates,"cell_assignment":"unverified","source":"native_pdf_line"})
    return audit

def build_context_graph(elements, tables, heights, options=None):
    """Bounded evidence relationships; edges are candidates, never physical table merges."""
    opts=validated_options(options)
    edges=[]
    by_ref={e["ref"]:e for e in elements}
    table_map={t["ref"]:t for t in tables}
    body=[e for e in elements if e.get("content_layer")=="body"]
    def add(a,b,kind):
        if a!=b: edges.append({"source":a,"target":b,"kind":kind,"status":"structural_candidate_unverified"})
    for e in body:
        for h in e["heading_candidates"]:add(e["ref"],h["ref"],"heading")
    def box(element):
        p=element["provenance"][0]
        b=p["bbox"]
        return p["page_no"], [b["l"],b["t"],b["r"],b["b"]] if b["coord_origin"]=="TOPLEFT" else [b["l"],heights[p["page_no"]]-b["t"],b["r"],heights[p["page_no"]]-b["b"]]
    def near_previous_note(previous,current):
        if current["label"]!="text" or not current["provenance"] or not previous["provenance"]:return False
        pp,pb=box(previous);cp,cb=box(current)
        gap=cb[1]-pb[3]
        overlap=max(0,min(pb[2],cb[2])-max(pb[0],cb[0]))/max(1,min(pb[2]-pb[0],cb[2]-cb[0]))
        return pp==cp and 0<=gap<=opts["note_gap"] and overlap>=opts["note_overlap"]
    previous=None
    pending=[]
    for e in body:
        if e["label"] in ("title","section_header"):
            pending=[]
            previous=None
        elif e["label"]=="table":
            for p in pending:add(e["ref"],p["ref"],"preceding_section_text")
            previous=e
            pending=[]
        else:
            if previous and (re.match(r"^\s*(?:[*†‡]|note\b|provided\b)",e["text"],re.I) or near_previous_note(previous,e)):
                add(previous["ref"],e["ref"],"following_note")
                add(e["ref"],previous["ref"],"note_table")
            else:
                pending.append(e)
    # Use source positions, not parser traversal order, for cross-page adjacency.
    positioned=[]
    for t in tables:
        for reg in t["regions"]:positioned.append((reg["page"],reg["bbox"][1],t,reg["bbox"]))
    positioned.sort(key=lambda z:(z[0],z[1]))
    for left,right in zip(positioned,positioned[1:]):
        lp,_,lt,lb=left;rp,_,rt,rb=right
        if rp!=lp+1 or lt["shape"][1]!=rt["shape"][1]:continue
        if lb[3]<heights[lp]*opts["continuation_edge"] or rb[1]>heights[rp]*opts["continuation_top"]:continue
        width=max(1,lb[2]-lb[0])
        if max(abs(lb[0]-rb[0]),abs(lb[2]-rb[2]))/width>opts["continuation_width_tolerance"]:continue
        intervening=False
        for e in body:
            if e["label"] not in ("title","section_header"):continue
            for p in e["provenance"]:
                b=p["bbox"]
                top=b["t"] if b["coord_origin"]=="TOPLEFT" else heights[p["page_no"]]-b["t"]
                if (lp,lb[3])<(p["page_no"],top)<(rp,rb[1]):intervening=True
        # A fresh spanning caption indicates a separate table.
        caption=any(c["start_row_offset_idx"]==0 and c["end_col_offset_idx"]-c["start_col_offset_idx"]==rt["shape"][1]
                    and any(ch.isalpha() for ch in c["text"]) for c in rt["raw_cells"])
        if not intervening and not caption:
            add(lt["ref"],rt["ref"],"cross_page_table")
            add(rt["ref"],lt["ref"],"cross_page_table")
    return {"version":VERSION,"edges":list({(e["source"],e["target"],e["kind"]):e for e in edges}.values()),
            "options":opts,"scope_status":"unverified","physical_table_merges":0}

def expand_refs(seeds, graph, by_ref, tokenizer, max_tokens, hops=2):
    neighbors=collections.defaultdict(list)
    for edge in graph["edges"]:neighbors[edge["source"]].append(edge["target"])
    selected=list(dict.fromkeys(r for r in seeds if r in by_ref))
    frontier=list(selected)
    for _ in range(hops):
        found=[]
        for r in frontier:
            for other in neighbors[r]:
                if other not in selected:
                    selected.append(other);found.append(other)
        frontier=found
    retained=[];deferred=[]
    for ref in selected:
        candidate="\n\n".join(by_ref[r]["text"] for r in retained+[ref])
        if len(tokenizer.encode(candidate,add_special_tokens=True))<=max_tokens:retained.append(ref)
        else:deferred.append(ref)
    text="\n\n".join(by_ref[r]["text"] for r in retained)
    return retained,text,deferred

def split_text(text, tokenizer, budget):
    """Split source substrings to a hard tokenizer budget, without dropping characters."""
    if budget is None:
        return [text]
    if budget <= len(tokenizer.encode("",add_special_tokens=True)):
        raise ValueError("Chunk budget must allow content plus special tokens")
    output=[]
    while text:
        if len(tokenizer.encode(text,add_special_tokens=True))<=budget:
            output.append(text);break
        lo,hi=1,len(text)
        while lo<hi:
            mid=(lo+hi+1)//2
            if len(tokenizer.encode(text[:mid],add_special_tokens=True))<=budget:lo=mid
            else:hi=mid-1
        cut=lo
        boundary=text.rfind(" ",0,cut)
        if boundary>cut//2:cut=boundary+1
        piece=text[:cut]
        # Token counts need not be strictly monotone for BPE; verify the actual piece.
        while piece and len(tokenizer.encode(piece,add_special_tokens=True))>budget:
            cut-=1;piece=text[:cut]
        if not piece:raise ValueError("One source character exceeds the token budget")
        output.append(piece);text=text[cut:]
    return output
