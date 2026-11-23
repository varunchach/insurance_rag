import hashlib, json
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download
from transformers import AutoTokenizer
from docling.datamodel.pipeline_options import LayoutObjectDetectionOptions
root = Path("/opt/rag/models")
root.mkdir(parents=True, exist_ok=True)
api = HfApi()
layout = LayoutObjectDetectionOptions().model_spec
specs = [
    (layout.repo_id, layout.revision, None),
    ("docling-project/docling-models", "v2.3.0", ["model_artifacts/tableformer/accurate/*", "README.md", "LICENSE*"]),
]
manifest = {"models": [], "tokenizer_purpose": "qualification only; embedding model selection remains open"}
for repo, revision, patterns in specs:
    sha = api.model_info(repo, revision=revision).sha
    dest = root / repo.replace("/", "--")
    snapshot_download(repo_id=repo, revision=sha, local_dir=dest, allow_patterns=patterns)
    manifest["models"].append({"repo": repo, "requested_revision": revision, "commit": sha, "directory": str(dest), "allow_patterns": patterns})
repo = "sentence-transformers/all-MiniLM-L6-v2"
sha = api.model_info(repo).sha
AutoTokenizer.from_pretrained(repo, revision=sha, trust_remote_code=False).save_pretrained(root / "tokenizer")
manifest["tokenizer"] = {"repo": repo, "commit": sha, "directory": str(root / "tokenizer"), "embedding_model_deployed": False}
manifest["files"] = {}
for p in sorted(root.rglob("*")):
    if p.is_file() and ".cache" not in p.parts:
        manifest["files"][str(p.relative_to(root))] = hashlib.sha256(p.read_bytes()).hexdigest()
Path("/opt/rag/model-manifest.json").write_text(json.dumps(manifest, indent=2))
print(json.dumps({k:v for k,v in manifest.items() if k != "files"}, indent=2))
