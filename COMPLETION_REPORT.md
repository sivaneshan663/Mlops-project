# Completion and verification report

Verified on 29 September 2026 for the local, synthetic-data review prototype.

| Requirement | Implementation | Evidence and result |
|---|---|---|
| Repeatable training pipeline | `pipeline.py`, `lifecycle.py`; `train_cost.py` delegates to the pipeline | Ran data validation → training → holdout evaluation → quality gates → versioned artifacts → promotion. `artifacts/pipeline_report.json` records the successful MLflow run. |
| Controlled promotion and rollback | Atomic serving manifest, previous version, baseline/current-model gates, artifact checksums | Actual isolated drift run promoted a candidate and rolled back. Automated API integration test verified predictions and version change without restart; rejection/checksum tests passed. |
| Monitoring dashboard | `/monitoring`, linked from chat | Browser-verified cards, serving version, request/error/latency metrics, missing-evidence/citation fallback counts, live input-drift status and evaluation reports. |
| Drift connected to retraining | PSI on synthetic reference and changed inputs; separate demo registry | Area PSI 2.9608 exceeded 0.2; labeled synthetic retraining ran, passed gates and rolled back. Serving registry remained unchanged. See `artifacts/drift_report.json`. |
| Broader RAG evaluation | Actual upload → hybrid retrieval → Qwen3-1.7B, two temporary project workspaces | 10/12 full passes. Both remaining cases returned citation-checked evidence excerpts, rather than uncited generated answers. All project-isolation and retrieval checks passed. See `artifacts/rag_evaluation.json`. |
| Repeatable checks and local deployment | `checks.py`, pinned clean runtime, GitHub workflow using the same command | Fresh Python 3.12 environment installed successfully. All 11 tests passed. A real temporary uvicorn process served health, system, chat and monitoring endpoints. Fresh environment also read the MLflow experiment store. |
| Review materials | README, review guide and architecture diagram | `REVIEW_GUIDE.md` provides a five-minute demo, architecture, experiment explanation and questions/answers. |
| Release evidence | Source, data/model hashes, Ollama identities and MLflow release record | `artifacts/release.json` generated after the implementation and checks. |

## Model results

- Served Random Forest version: `5a25bb4c58f9`.
- Cost pipeline MLflow run: `1003c4dc818243bbacce7d5ef5173ade`.
- Synthetic held-out cost MAE: INR 1,750,341; RMSE: INR 2,442,122; R²: 0.9817.
- Drift candidate version: `29fab6038d3c`; run: `c87608aa32c9403893c4130e6a572b0b`.
- Drift candidate MAE on shifted holdout: INR 1,992,631, compared with INR 2,327,539 for the current model on that same holdout.
- Drift demonstration rolled back to `5a25bb4c58f9` in its isolated registry.
- Default chat remains Qwen3-1.7B with all-minilm retrieval.
- Existing Qwen3-0.6B LoRA artifact and MLflow record were verified present. Its evaluation remains experimental/not promoted because of citation problems. The adapter worker was restarted and its health endpoint and application availability check passed; it loads weights lazily on first request.

## Verification limits

The completed scope is the local MLOps prototype. No expert-reviewed real construction dataset was available. Synthetic cost metrics and small heuristic RAG checks do not establish real-world engineering reliability. Citation-ID validation checks the reference format and membership, not full claim entailment.

The shared CI command ran locally; hosted GitHub Actions did not run because no remote repository is configured. Docker is not installed, so the Dockerfile is provided but no container build is claimed. Cloud deployment, authentication, OCR/drawings and production monitoring remain outside the prototype scope.

User project files and chat histories were preserved. Tests/evaluations used temporary project storage, and the drift demonstration used a separate model registry. Monitoring correctly reports insufficient live input-drift data until at least 30 estimates are available.
