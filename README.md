# Insurance RAG ingestion qualification

This project runs document ingestion qualification through RHOAI-managed Kubeflow Pipelines on OpenShift. It does not deploy a chat application or index unqualified content.

The pipeline preserves native PDF evidence, parses with Docling's accurate table model and OCR disabled, creates hierarchical and tokenizer-bound hybrid chunks, and scores independent evaluation fixtures. Parser output, cell spans, source references, chunk lineage and a review queue remain in private object storage.

## Execution

Build containers on OpenShift. Keep credentials in OpenShift Secrets. Do not commit PDFs, extracted client text, annotations, model weights, kubeconfigs or tokens.

1. Build the pinned CPU runtime using `runtime/Dockerfile.cpu`, then the CUDA runtime using `runtime/Dockerfile.gpu`. Update the GPU Dockerfile's source image reference for your own registry.
2. Upload a source PDF and independently maintained evaluation kit to approved S3 storage. Set the source SHA-256, fixture checksums, endpoint and object keys in a private configuration derived from `config.example.json`.
3. Put `src/qualification.py`, `src/preservation.py`, `pipelines/bootstrap.py`, `pipelines/compile_pipeline.py` and the private configuration into a versioned OpenShift ConfigMap. Set `code.sha256` to the SHA-256 of `qualification.py`; set `module.sha256` to a JSON mapping of both Python module names to their SHA-256 hashes. Make the ConfigMap immutable after validation.
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

## Preservation stage

Every new or imported parse passes through reconcile before chunking. original-document.json retains the original model output. document.json is the derived representation; the manifest binds both hashes and the repair configuration. A rerun begins from the saved original, so recoveries do not accumulate.

The stage reconstructs missing ruled tables and accepts a narrower ruled structure only when row counts agree, geometry overlaps, and native text retains the original tokens and numbers. Native lines missing from local parser regions are retained separately with page coordinates. Their table references are candidates; no cell assignment is invented. preservation-audit.json records original table data, recovery reasons, list markers and native snippets.

Numbered headings and consecutive heading groups retain ancestry. Standalone fallback chunks preserve body elements omitted by the serializer. context-graph.json records heading, nearby-note, preceding-section and cross-page table candidates. Table segments remain separate. Expanded evidence has a configurable token budget and hop limit, and deferred references are explicit. Candidate relationships do not establish product ownership, effective-date precedence or exception applicability.

config.example.json lists the geometric thresholds and expansion limits under preservation. They are general document rules; production code does not read page-specific annotations or expected answers. Supply parsed_artifact_uri to reuse a GPU parse for CPU-only preservation changes.

The scorecard retains the original numeric disagreement metric and also reports coverage after alphanumeric splitting. This treats split ordinal suffixes consistently while retaining punctuation in the comparison. Numeric multiset coverage cannot prove that a value belongs to the correct cell.

## Verification

Run python -m unittest discover -s tests -v inside the approved OCP runtime. Synthetic tests cover hierarchy, merged cells, native recovery, literal NA/Nil/blank values, continuation rejection, note direction, numeric signs and decimals, and tokenizer limits. The document benchmark runs separately with independently maintained private fixtures. Keep source documents, annotations, outputs and scorecards out of this repository.
