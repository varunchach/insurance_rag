"""Per-upload source contracts and format checks; independent of benchmark fixtures."""
import hashlib, json, re
from pathlib import Path, PurePosixPath

INPUT_DEFAULTS = {"max_source_bytes": 100 * 1024 * 1024, "max_pages": 1000,
                  "min_native_chars": 20, "review_image_area_ratio": .05}
RUNTIME_FIELDS = {"storage_secret", "storage_endpoint", "bucket", "runtime_image",
    "code_configmap", "cell_matching", "cpu_threads", "chunk_budgets", "primary_chunk_budget",
    "layout_batch_size", "table_batch_size", "accelerator", "preservation", "input_policy",
    "model_path", "tokenizer_path"}

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def policy(config):
    options = config.get("input_policy", {})
    unknown = set(options) - set(INPUT_DEFAULTS)
    if unknown:
        raise ValueError("Unknown input policy settings: " + ", ".join(sorted(unknown)))
    result = {**INPUT_DEFAULTS, **options}
    for name in ("max_source_bytes", "max_pages", "min_native_chars"):
        if not isinstance(result[name], int) or isinstance(result[name], bool) or result[name] <= 0:
            raise ValueError(name + " must be a positive integer")
    if not 0 < result["review_image_area_ratio"] <= 1:
        raise ValueError("review_image_area_ratio must be in (0, 1]")
    return result

def validate_source(config):
    key = config.get("source_key", "")
    fingerprint = config.get("source_sha256", "")
    if not isinstance(key, str) or not key or key.startswith("/") or "\\" in key or "://" in key:
        raise ValueError("source_key must be a relative object key in the configured bucket")
    if any(p in (".", "..") for p in key.split("/")) or any(ord(c)<32 for c in key):
        raise ValueError("Invalid source object key")
    if not isinstance(fingerprint, str) or not re.fullmatch(r"[0-9a-f]{64}", fingerprint):
        raise ValueError("source_sha256 must be an explicit lowercase SHA-256")
    policy(config)

def bind_runtime_source(base, source_key, source_sha256, document_id):
    if not document_id or not document_id.strip() or len(document_id)>256:
        raise ValueError("document_id must be supplied for each upload")
    config = {k:v for k,v in base.items() if k in RUNTIME_FIELDS}
    config.update(source_key=source_key, source_sha256=source_sha256,
                  document_id=document_id, execution_mode="per_document_ingestion")
    validate_source(config)
    return config

def verify_artifact(config, directory, require_source=True, canonical=False):
    validate_source(config)
    directory=Path(directory)
    manifest=json.loads((directory/"manifest.json").read_text())
    if manifest.get("source_sha256") != config["source_sha256"]:
        raise ValueError("Artifact source identity does not match the requested document")
    if config.get("document_id") and manifest.get("document_id") != config["document_id"]:
        raise ValueError("Artifact document_id does not match this upload")
    if require_source and sha256(directory/"source.pdf") != config["source_sha256"]:
        raise ValueError("Artifact PDF checksum does not match this upload")
    if canonical:
        if not manifest.get("canonical_sha256") or sha256(directory/"document.json") != manifest["canonical_sha256"]:
            raise ValueError("Canonical parse checksum mismatch")
        if manifest.get("original_canonical_sha256"):
            if sha256(directory/"original-document.json") != manifest["original_canonical_sha256"]:
                raise ValueError("Original parse checksum mismatch")
    for name,expected in manifest.get("files_sha256",{}).items():
        relative=PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or "\\" in name:
            raise ValueError("Invalid artifact manifest path")
        if sha256(directory/name)!=expected:
            raise ValueError("Artifact payload checksum mismatch: "+name)
    return manifest

def seal_manifest(directory, manifest):
    directory=Path(directory)
    manifest["files_sha256"]={p.name:sha256(p) for p in directory.iterdir()
        if p.is_file() and p.name!="manifest.json" and p.suffix in (".json",".pdf")}
    (directory/"manifest.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False,default=str))
    return manifest

def verify_chunk_binding(config, parsed, chunks):
    source=verify_artifact(config,parsed,canonical=True)
    child=verify_artifact(config,chunks,require_source=False)
    if child.get("parsed_canonical_sha256") != source["canonical_sha256"]:
        raise ValueError("Chunks do not belong to this parse version")
    return source,child

def inspect_pdf(path, config):
    import fitz
    opts=policy(config)
    path=Path(path)
    if path.stat().st_size>opts["max_source_bytes"]:
        raise ValueError("Source exceeds configured byte limit")
    if path.read_bytes()[:5]!=b"%PDF-":
        raise ValueError("This ingestion path accepts PDF documents only")
    records=[];reasons=[]
    with fitz.open(path) as doc:
        if doc.is_encrypted:
            raise ValueError("Encrypted PDFs require an approved decryption path")
        if len(doc)>opts["max_pages"]:
            raise ValueError("Source exceeds configured page limit")
        for number,page in enumerate(doc,1):
            text=page.get_text("text")
            rectangles=[]
            for image in page.get_images(full=True):
                for box in page.get_image_rects(image[0]):
                    clipped=box & page.rect
                    if not clipped.is_empty:rectangles.append(tuple(clipped))
            image_area=min(1,sum(fitz.Rect(b).get_area() for b in set(rectangles))/max(1,page.rect.get_area()))
            drawings=len(page.get_drawings())
            chars=len(re.sub(r"\s+","",text))
            blank=chars==0 and not rectangles and drawings==0
            issues=[]
            if image_area>=opts["review_image_area_ratio"]:
                issues.append("ocr_required" if chars<opts["min_native_chars"] else "visual_content_review_required")
            if chars==0 and not blank and not issues:
                issues.append("non_native_content_review_required")
            records.append({"page":number,"native_chars":chars,"image_area_ratio":image_area,
                "drawing_count":drawings,"blank":blank,"issues":issues,"width":page.rect.width,"height":page.rect.height})
            reasons.extend({"page":number,"reason":reason} for reason in issues)
    if not records or all(p["blank"] for p in records):
        reasons.append({"page":None,"reason":"empty_document"})
    return {"pages":records,"page_count":len(records),"parse_allowed":not reasons,
            "review_reasons":reasons,"policy":opts,"ocr_available":False,
            "status":"native_parse_allowed" if not reasons else "review_required"}
