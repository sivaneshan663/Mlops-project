"""Register the currently promoted cost model in the MLflow Model Registry."""
import json
import os
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
REGISTERED_NAME = "ConstructAI-Cost-RandomForest"


class ConstructAICostModel(mlflow.pyfunc.PythonModel):
    """MLflow wrapper around the project's NumPy Random Forest."""

    def load_context(self, context):
        self.model = joblib.load(context.artifacts["model_file"])
        self.features = list(context.model_config["features"])

    def predict(self, context, model_input, params=None):
        if isinstance(model_input, pd.DataFrame):
            values = model_input[self.features].to_numpy(dtype=float)
        else:
            values = np.asarray(model_input, dtype=float)
        return self.model.predict(values)


def register():
    manifest = json.loads((ARTIFACTS / "cost_manifest.json").read_text(encoding="utf-8"))
    model_path = ARTIFACTS / manifest["model_file"]
    features = manifest["features"]
    input_example = pd.DataFrame([{
        "area_sqft": 3000.0,
        "floors": 3.0,
        "quality": 2.0,
        "location_factor": 1.0,
        "duration_months": 12.0,
    }])[features]
    native_model = joblib.load(model_path)
    output_example = native_model.predict(input_example.to_numpy(dtype=float))

    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    mlflow.set_tracking_uri((ROOT / "mlruns").as_uri())
    with mlflow.start_run(run_id=manifest["run_id"]):
        result = mlflow.pyfunc.log_model(
            artifact_path="registered_cost_model",
            python_model=ConstructAICostModel(),
            artifacts={"model_file": str(model_path)},
            code_paths=[str(ROOT / "forest.py")],
            model_config={"features": features},
            input_example=input_example,
            signature=infer_signature(input_example, output_example),
            pip_requirements=["numpy==2.5.3", "pandas==2.3.3", "joblib==1.6.0", "mlflow==2.22.2"],
            registered_model_name=REGISTERED_NAME,
            metadata={
                "model_version_id": manifest["version"],
                "data_kind": manifest["data_kind"],
                "currency": manifest["currency"],
                "release_gate": manifest["release_gate"],
            },
        )

    client = mlflow.MlflowClient()
    versions = client.search_model_versions(f"name='{REGISTERED_NAME}'")
    current = max(versions, key=lambda item: int(item.version))
    client.set_registered_model_alias(REGISTERED_NAME, "champion", current.version)
    client.update_registered_model(
        REGISTERED_NAME,
        description="Promoted ConstructAI cost estimator. NumPy Random Forest trained on synthetic prototype data.",
    )
    client.update_model_version(
        REGISTERED_NAME,
        current.version,
        description=(
            f"Internal version {manifest['version']}; release gate {manifest['release_gate']}; "
            f"MAE INR {manifest['metrics']['mae_inr']:.0f}; R2 {manifest['metrics']['r2']:.4f}."
        ),
    )
    return {
        "registered_model": REGISTERED_NAME,
        "registry_version": current.version,
        "alias": "champion",
        "run_id": manifest["run_id"],
        "model_uri": result.model_uri,
        "internal_version": manifest["version"],
    }


if __name__ == "__main__":
    print(json.dumps(register(), indent=2))
