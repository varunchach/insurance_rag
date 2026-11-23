"""Compile the qualification DAG on OCP with the installed KFP SDK."""
import json, os, pathlib, sys
from kfp import dsl, compiler
from kfp.dsl import Artifact, Input, Output
import yaml
CONFIG=json.loads(pathlib.Path(sys.argv[1]).read_text())
IMAGE=CONFIG["runtime_image"]
CODE_CONFIGMAP=CONFIG["code_configmap"]
BOOTSTRAP=pathlib.Path(sys.argv[2]).read_text()
OUT=pathlib.Path(sys.argv[3])
@dsl.container_component
def native_check(native:Output[Artifact]):
    return dsl.ContainerSpec(image=IMAGE,command=["python","-c",BOOTSTRAP],
        args=[CODE_CONFIGMAP,"native","--out",native.path])
@dsl.container_component
def docling_parse(parsed:Output[Artifact]):
    return dsl.ContainerSpec(image=IMAGE,command=["python","-c",BOOTSTRAP],
        args=[CODE_CONFIGMAP,"parse","--out",parsed.path])
@dsl.container_component
def reconcile_parse(parsed:Input[Artifact],reconciled:Output[Artifact]):
    return dsl.ContainerSpec(image=IMAGE,command=["python","-c",BOOTSTRAP],
        args=[CODE_CONFIGMAP,"reconcile","--parsed",parsed.path,"--out",reconciled.path])
@dsl.container_component
def context_chunks(parsed:Input[Artifact],chunks:Output[Artifact]):
    return dsl.ContainerSpec(image=IMAGE,command=["python","-c",BOOTSTRAP],
        args=[CODE_CONFIGMAP,"chunk","--parsed",parsed.path,"--out",chunks.path])
@dsl.container_component
def qualification_score(native:Input[Artifact],parsed:Input[Artifact],chunks:Input[Artifact],report:Output[Artifact]):
    return dsl.ContainerSpec(image=IMAGE,command=["python","-c",BOOTSTRAP],
        args=[CODE_CONFIGMAP,"evaluate","--native",native.path,"--parsed",parsed.path,"--chunks",chunks.path,"--out",report.path])
@dsl.pipeline(name="rag-docling-qualification",description="Native comparison, Docling accurate native-text parse, tokenizer-bound chunks and independent evaluation. Review required; no indexing.")
def pipeline():
    if CONFIG.get("native_artifact_uri"):
        n=dsl.importer(artifact_uri=CONFIG["native_artifact_uri"],artifact_class=Artifact,reimport=False)
        native_artifact=n.output
    else:
        n=native_check().set_cpu_request("1").set_cpu_limit("2").set_memory_request("2Gi").set_memory_limit("4Gi").set_caching_options(False)
        native_artifact=n.outputs["native"]
    if CONFIG.get("parsed_artifact_uri"):
        saved=dsl.importer(artifact_uri=CONFIG["parsed_artifact_uri"],artifact_class=Artifact,reimport=False)
        parsed_artifact=saved.output
    else:
        p=docling_parse().set_cpu_request("4").set_cpu_limit("4").set_memory_request("8Gi").set_memory_limit("16Gi").set_accelerator_type("nvidia.com/gpu").set_accelerator_limit(1).set_caching_options(False)
        parsed_artifact=p.outputs["parsed"]
    repaired=reconcile_parse(parsed=parsed_artifact).set_cpu_request("1").set_cpu_limit("2").set_memory_request("2Gi").set_memory_limit("4Gi").set_caching_options(False)
    parsed_artifact=repaired.outputs["reconciled"]
    c=context_chunks(parsed=parsed_artifact).set_cpu_request("1").set_cpu_limit("2").set_memory_request("2Gi").set_memory_limit("8Gi").set_caching_options(False)
    qualification_score(native=native_artifact,parsed=parsed_artifact,chunks=c.outputs["chunks"]).set_cpu_request("1").set_cpu_limit("2").set_memory_request("2Gi").set_memory_limit("4Gi").set_caching_options(False)
OUT.mkdir(parents=True,exist_ok=True)
compiler.Compiler().compile(pipeline_func=pipeline,package_path=str(OUT/"pipeline.yaml"))
spec=yaml.safe_load((OUT/"pipeline.yaml").read_text())
(OUT/"pipeline.json").write_text(json.dumps(spec,indent=2))
print(json.dumps({"compiled":True,"tasks":list(spec["root"]["dag"]["tasks"]),"image":IMAGE}))
