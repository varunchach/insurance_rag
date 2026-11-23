"""Compile one reusable, per-upload ingestion DAG. No fixture or artifact URI inputs."""
import json, pathlib, sys
from kfp import dsl, compiler
from kfp.dsl import Artifact, Input, Output, OutputPath
import yaml

CONFIG=json.loads(pathlib.Path(sys.argv[1]).read_text())
IMAGE=CONFIG["runtime_image"]
CODE_CONFIGMAP=CONFIG["code_configmap"]
BOOTSTRAP=pathlib.Path(sys.argv[2]).read_text()
OUT=pathlib.Path(sys.argv[3])

def request_args(stage, source_key, source_sha256, document_id):
    return [CODE_CONFIGMAP,stage,"--dynamic-source","--source-key",source_key,
            "--source-sha256",source_sha256,"--document-id",document_id]

@dsl.container_component
def inspect_source(source_key:str,source_sha256:str,document_id:str,
                   source:Output[Artifact],parse_allowed:OutputPath(str)):
    return dsl.ContainerSpec(image=IMAGE,command=["python","-c",BOOTSTRAP],
        args=request_args("preflight",source_key,source_sha256,document_id)+
             ["--out",source.path,"--gate",parse_allowed])

@dsl.container_component
def parse_document(source_key:str,source_sha256:str,document_id:str,
                   source:Input[Artifact],parsed:Output[Artifact]):
    return dsl.ContainerSpec(image=IMAGE,command=["python","-c",BOOTSTRAP],
        args=request_args("parse",source_key,source_sha256,document_id)+
             ["--source-profile",source.path,"--out",parsed.path])

@dsl.container_component
def preserve_document(source_key:str,source_sha256:str,document_id:str,
                      parsed:Input[Artifact],preserved:Output[Artifact]):
    return dsl.ContainerSpec(image=IMAGE,command=["python","-c",BOOTSTRAP],
        args=request_args("reconcile",source_key,source_sha256,document_id)+
             ["--parsed",parsed.path,"--out",preserved.path])

@dsl.container_component
def chunk_document(source_key:str,source_sha256:str,document_id:str,
                   parsed:Input[Artifact],chunks:Output[Artifact]):
    return dsl.ContainerSpec(image=IMAGE,command=["python","-c",BOOTSTRAP],
        args=request_args("chunk",source_key,source_sha256,document_id)+
             ["--parsed",parsed.path,"--out",chunks.path])

@dsl.container_component
def inspect_ingestion(source_key:str,source_sha256:str,document_id:str,source:Input[Artifact],
                      parsed:Input[Artifact],chunks:Input[Artifact],report:Output[Artifact]):
    return dsl.ContainerSpec(image=IMAGE,command=["python","-c",BOOTSTRAP],
        args=request_args("ingestion-report",source_key,source_sha256,document_id)+
             ["--source-profile",source.path,"--parsed",parsed.path,"--chunks",chunks.path,"--out",report.path])

@dsl.pipeline(name="rag-document-ingestion",
              description="Per-upload PDF ingestion with required source identity, explicit visual-review routing and no benchmark fixtures.")
def pipeline(source_key:str,source_sha256:str,document_id:str):
    source=inspect_source(source_key=source_key,source_sha256=source_sha256,document_id=document_id
        ).set_cpu_request("1").set_cpu_limit("2").set_memory_request("1Gi").set_memory_limit("4Gi").set_caching_options(False)
    with dsl.If(source.outputs["parse_allowed"]=="true",name="native-text-supported"):
        parsed=parse_document(source_key=source_key,source_sha256=source_sha256,document_id=document_id,
            source=source.outputs["source"]).set_cpu_request("4").set_cpu_limit("4").set_memory_request(
            "8Gi").set_memory_limit("16Gi").set_accelerator_type("nvidia.com/gpu").set_accelerator_limit(1).set_caching_options(False)
        preserved=preserve_document(source_key=source_key,source_sha256=source_sha256,document_id=document_id,
            parsed=parsed.outputs["parsed"]).set_cpu_request("1").set_cpu_limit("2").set_memory_request(
            "2Gi").set_memory_limit("4Gi").set_caching_options(False)
        chunks=chunk_document(source_key=source_key,source_sha256=source_sha256,document_id=document_id,
            parsed=preserved.outputs["preserved"]).set_cpu_request("1").set_cpu_limit("2").set_memory_request(
            "2Gi").set_memory_limit("8Gi").set_caching_options(False)
        inspect_ingestion(source_key=source_key,source_sha256=source_sha256,document_id=document_id,
            source=source.outputs["source"],parsed=preserved.outputs["preserved"],chunks=chunks.outputs["chunks"]
            ).set_cpu_request("1").set_cpu_limit("2").set_memory_request("1Gi").set_memory_limit("4Gi").set_caching_options(False)

OUT.mkdir(parents=True,exist_ok=True)
compiler.Compiler().compile(pipeline_func=pipeline,package_path=str(OUT/"pipeline.yaml"))
spec=yaml.safe_load((OUT/"pipeline.yaml").read_text())
(OUT/"pipeline.json").write_text(json.dumps(spec,indent=2))
print(json.dumps({"compiled":True,"required_upload_parameters":list(spec["root"]["inputDefinitions"]["parameters"]),
                  "image":IMAGE,"gold_annotations_required":False}))
