import json, os, pathlib, ssl, sys, urllib.request, hashlib
sa=pathlib.Path("/var/run/secrets/kubernetes.io/serviceaccount")
api="https://"+os.environ["KUBERNETES_SERVICE_HOST"]+":"+os.environ.get("KUBERNETES_SERVICE_PORT_HTTPS","443")
req=urllib.request.Request(api+"/api/v1/namespaces/"+sa.joinpath("namespace").read_text().strip()+"/configmaps/"+sys.argv[1],
    headers={"Authorization":"Bearer "+sa.joinpath("token").read_text().strip()})
with urllib.request.urlopen(req,context=ssl.create_default_context(cafile=str(sa/"ca.crt")),timeout=30) as r:
    data=json.load(r)["data"]
root=pathlib.Path("/tmp/rag-code")
root.mkdir(exist_ok=True)
module_hashes=json.loads(data["module.sha256"])
assert set(module_hashes)=={"qualification.py","preservation.py","document_contract.py"}, "Incomplete executable manifest"
for name, expected_hash in module_hashes.items():
    assert name in ("qualification.py","preservation.py","document_contract.py"), "Unexpected executable module"
    assert hashlib.sha256(data[name].encode()).hexdigest()==expected_hash, "Module hash mismatch"
    compile(data[name],name,"exec")
    root.joinpath(name).write_text(data[name],encoding="utf-8")
for name in ["config.json"]:
    root.joinpath(name).write_text(data[name],encoding="utf-8")
expected=data["code.sha256"]
assert hashlib.sha256(data["qualification.py"].encode()).hexdigest()==expected, "Code hash mismatch"
compile(data["qualification.py"],"qualification.py","exec")
os.execv(sys.executable,[sys.executable,str(root/"qualification.py"),sys.argv[2],"--config",str(root/"config.json"),*sys.argv[3:]])
