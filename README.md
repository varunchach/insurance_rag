# Insurance RAG ingestion qualification

This project runs document ingestion qualification through RHOAI-managed Kubeflow Pipelines on OpenShift. It does not deploy a chat application or index unqualified content.

The pipeline preserves native PDF evidence, parses with Docling's accurate table model and OCR disabled, creates hierarchical and tokenizer-bound hybrid chunks, and scores independent evaluation fixtures. Parser output, cell spans, source references, chunk lineage and a review queue remain in private object storage.

## Execution

Build containers on OpenShift. Keep credentials in OpenShift Secrets. Do not commit PDFs, extracted client text, annotations, model weights, kubeconfigs or tokens.

1. Build the pinned CPU runtime using `runtime/Dockerfile.cpu`, then the CUDA runtime using `runtime/Dockerfile.gpu`. Update the GPU Dockerfile's source image reference for your own registry.
2. Upload a source PDF and independently maintained evaluation kit to approved S3 storage. Set the source SHA-256, fixture checksums, endpoint and object keys in a private configuration derived from `config.example.json`.
3. Put `src/qualification.py`, `pipelines/bootstrap.py`, `pipelines/compile_pipeline.py` and the private configuration into a versioned OpenShift ConfigMap. Set `code.sha256` to the SHA-256 of `qualification.py`; make the ConfigMap immutable after validation.
4. Compile on OCP using the installed KFP SDK:
   `python pipelines/compile_pipeline.py /path/config.json pipelines/bootstrap.py /path/output`
5. Register the resulting pipeline specification with the existing RHOAI pipeline server and execute it. The Docling parsing task requests one `nvidia.com/gpu`; text extraction, chunking and scoring use CPUs. The pipeline service account requires read access to its named ConfigMap and storage Secret.
6. Inspect `scorecard.json`, `qualification-report.md` and `review-queue.json` in the output artifact before considering retrieval.

A completed native baseline artifact can be supplied as `native_artifact_uri` to avoid repeating extraction. Omit it to run the native stage in the same DAG. The baseline kit must provide `profile_pdf.py`, `benchmark.py`, `benchmark_config.json` and `gold_annotations.json`; hashes are mandatory and fixtures are never loaded by the parser.

## Interpretation

Successful execution is not production qualification. Table dimensions and normalized row strings do not validate cell-to-header meaning. Page-bound anchor co-location is not retrieval or answer accuracy. Native numeric disagreements and cross-page table candidates require source review. Product scope, dates, exceptions and continuation links remain unverified until independently validated.

The tokenizer is used for qualification only; no embedding model has been selected or deployed. Model revisions and file hashes are recorded in the runtime model manifest. The built image digest and dependency lock identify the executed environment; a future rebuild must reuse these locks and model revisions to reproduce it.

The sample operator-managed MinIO and MariaDB used during qualification are development infrastructure. Upstream Docling and the custom runtime are not represented as Red Hat-supported parser components.

## References

- [RHOAI 3.5 pipelines](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/3.5/html/working_with_ai_pipelines/)
- [Docling offline model artifacts](https://docling-project.github.io/docling/usage/advanced_options/)
- [Docling hybrid chunking](https://docling-project.github.io/docling/_generated/examples/hybrid_chunking/)
- [PyTorch CUDA wheels](https://pytorch.org/get-started/previous-versions/)

The optional GPU comparison is compiled with pipelines/compile_ablation.py using the same arguments. It runs matching on and off consecutively, records GPU utilization every 200 ms, and evaluates both outputs. Larger batches are configurable and must be assessed by elapsed time and quality, not peak utilization alone.
