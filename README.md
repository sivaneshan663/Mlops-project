# ConstructAI — MLOps review prototype

ConstructAI is a local construction project assistant with project-specific document chat, synthetic cost estimation, and a demonstrable ML lifecycle. It is a single-user educational prototype, not a validated engineering tool.

## Start on this laptop

Keep Ollama running, then run `start-all.ps1` from this folder in PowerShell. Open http://127.0.0.1:8000. Select **MLOps monitor** in the sidebar for the monitoring dashboard.

The MLflow experiment dashboard is at http://127.0.0.1:5000. `start-all.ps1` starts it automatically when the dedicated UI environment is present. You can also run `start-mlflow.ps1` by itself.

The promoted Random Forest is also available in MLflow under **Models → ConstructAI-Cost-RandomForest**. The `champion` alias identifies the version used by the application. Run `..\..\work\mlflow-ui-env\Scripts\python.exe register_cost_model.py` after promoting a new cost model to publish its packaged model, signature and input example to the registry.

## DVC pipeline

DVC versions the training dataset and reproduces the cost-model stage. This local prototype uses a no-SCM DVC repository because the application folder is distributed independently of Git. The pipeline definition is in `dvc.yaml`, the reproducibility lockfile is `dvc.lock`, and the dataset pointer is `artifacts/synthetic_cost.csv.dvc`.

From this folder, install the optional DVC tooling once and reproduce the pipeline:

```powershell
python -m pip install -r requirements-dvc.txt
dvc repro
dvc status
```

`dvc repro` runs the same validated training script used by MLflow. DVC tracks the dataset and pipeline outputs; the run's parameters, metrics, and model registration remain in MLflow. A local DVC cache is used for this review prototype; a shared remote storage target can be configured later with `dvc remote add`.

The default chat model is local `qwen3:1.7b`; embeddings use `all-minilm`. The separately trained Qwen3-0.6B LoRA adapter is experimental and requires its service on port 8001. Standard chat does not depend on that service.

`start.ps1` starts only the main application. Scripts prefer a `.venv` in this folder, falling back to the existing laptop environment at `../../work/prototype-env`. Stop the old server before restarting after backend edits.

## Clean setup

Use Python 3.12. From this folder:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-runtime-lock.txt
ollama pull qwen3:1.7b
ollama pull all-minilm
.\start.ps1
```

Direct application dependencies are pinned in `requirements.txt`; `requirements-runtime-lock.txt` captures the complete clean environment that passed the review checks. `requirements-lock.txt` is the original laptop snapshot, including optional training packages; it is not required for standard chat. User uploads and conversations persist in `data/` across restarts. Back up that folder yourself; there is no cloud backup.

## Repeatable model lifecycle

In the following commands, use the installed environment's Python. On this laptop it is `../../work/prototype-env/Scripts/python.exe`.

```text
python pipeline.py
python pipeline.py --drift-demo
python checks.py
python evaluate_rag.py
python release.py
```

1. **Validate:** enforce five feature ranges, integer floors/quality, finite values, positive labels, and minimum dataset size.
2. **Train:** generate 2,500 explicitly synthetic records; seed 42; 2,000 training and 500 held-out rows; 120 bootstrapped CART trees.
3. **Evaluate:** compute MAE, RMSE and R². Compare candidate and current model on the same candidate holdout.
4. **Gate:** require MAE at least 20% better than the mean baseline, R² at least 0.8, and no more than 2% MAE regression against the current model.
5. **Register:** save dataset/model hashes, versioned artifacts, candidate manifest, and real MLflow run parameters/metrics/artifacts.
6. **Promote:** atomically update the serving manifest only on a passing gate. Keep the previous version. A failed gate leaves the current serving model unchanged.
7. **Rollback:** `python pipeline.py --rollback` restores the saved previous manifest; the next estimate loads that version, without server restart.

`train_cost.py` delegates to this same gated pipeline. This is a small local manifest registry, not the hosted MLflow Model Registry. Repeated experiments on this fixed synthetic holdout are a lifecycle demonstration, not an unbiased final accuracy benchmark. Run only one training/rollback command at a time.

## Drift and monitoring

The dashboard shows request count, HTTP/server errors, 95th percentile latency, missing-evidence replies, citation fallbacks, model availability fallbacks, current models, and the latest pipeline/evaluation reports.

Live cost-input drift compares up to 500 saved estimates with the reference dataset using population stability index (PSI), with at least 30 observations and a demonstration threshold of 0.2. This is input drift, not measured model degradation. There is no live ground-truth cost feedback stream.

`--drift-demo` uses its own registry under `artifacts/drift_demo/`. It verifies a stable reference, shifts synthetic areas toward larger projects, detects drift, generates labeled synthetic training data, retrains, applies the quality gates, and demonstrates rollback. It does not alter the serving registry. Real drift does not automatically fabricate labels or retrain the serving model.

## Chat and RAG

PDFs with selectable text, UTF-8 TXT and CSV files up to 10 MB are saved locally. Document candidates and conversation history are restricted to the selected project. Hybrid search uses all-minilm embeddings and lexical similarity. Answers show expandable passages. Duplicate file contents are detected within a project.

The chatbot distinguishes synthetic estimates from actual spending. A response with missing/unknown citation IDs is replaced by labeled document excerpts. Citation-ID checks do not prove that every claim is supported. Long-document summaries cover retrieved passages, not every page.

`evaluate_rag.py` exercises real upload → hybrid retrieval → Qwen generation in a temporary database, without touching user uploads. The latest 12-question run passed 10 full checks; two questions fell back to source excerpts because generation omitted citations. Both project isolation and retrieval checks passed throughout. This small synthetic suite uses heuristic checks and is not a real-world accuracy claim. The pre-fix report is retained for comparison.

## Fine-tuning

A Qwen3-0.6B rank-4 LoRA adapter was trained on 24 synthetic instruction/answer examples for one epoch. Eight examples were reserved separately. `evaluate.py` compares base/adapted 0.6B with and without supplied gold context on four synthetic questions; it is distinct from end-to-end RAG evaluation. Unsupported/repetitive citations prevented promotion to the default chatbot. The standard Qwen3-1.7B was not fine-tuned.

The optional service requires `requirements-training.txt`, the saved adapter, and original Qwen3-0.6B base files at `../../work/base-qwen06`. The adapter alone is not a full model. `train_adapter.py`, `adapter_service.py`, and `verify_adapter.py` contain this existing laptop setup.

## Evidence files

| Evidence | Location |
|---|---|
| Serving cost model and measured synthetic metrics | `artifacts/cost_manifest.json` |
| Training validation, quality gate and promotion | `artifacts/pipeline_report.json` |
| Drift, retraining and rollback | `artifacts/drift_report.json` |
| Real upload/retrieval/generation checks | `artifacts/rag_evaluation.json` |
| Adapter training and controlled comparison | `artifacts/sft_manifest.json`, `artifacts/evaluation.json` |
| Automated tests and process startup | `artifacts/checks_report.json` |
| Code/model release identities | `artifacts/release.json` |
| MLflow experiment records | `mlruns/` |
| Local request status/latency logs (no bodies) | `data/requests.jsonl` |

The current cost run has synthetic MAE INR 1,750,341, RMSE INR 2,442,122, R² 0.9817; the mean baseline MAE is INR 15,123,939. The forest uses NumPy because Windows blocked a compiled scikit-learn dependency. No Windows protection was disabled. MLflow uses its supported file store because this machine blocked its SQL backend dependency.

## CI and deployment boundary

`checks.py` runs the automated test suite, then launches a real temporary uvicorn process and verifies health, system, chat page and monitoring endpoints. The same command is used by `.github/workflows/checks.yml` when this folder is a repository root. It has passed locally. Hosted CI has not run because no remote repository is configured.

A Dockerfile is supplied; Docker is not installed on this laptop, so no container build has been verified. For Docker Desktop, use host gateway URLs for Ollama/optional adapter, set OLLAMA_MODEL and EMBEDDING_MODEL, and mount `/app/data` persistently. The verified deployment is the local Python service. Authentication, public/cloud deployment, OCR/drawing understanding and expert-reviewed construction data remain outside scope.

See REVIEW_GUIDE.md for the presentation script and COMPLETION_REPORT.md for verification results.
