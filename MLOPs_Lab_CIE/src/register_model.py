import mlflow
import json
import os
from mlflow.tracking import MlflowClient

client = MlflowClient()
experiment_name = "adpulse-click-through-rate"
reg_name = "adpulse-click-through-rate-predictor"

# Get the absolute latest run from Task 1
exp = client.get_experiment_by_name(experiment_name)
runs = client.search_runs(
    experiment_ids=[exp.experiment_id],
    filter_string="tags.priority = 'high'",
    order_by=["attribute.start_time DESC"], # Get the one you JUST ran
    max_results=1
)

best_run_id = runs[0].info.run_id
mae = runs[0].data.metrics.get('mae', 0.0)

# Register version
model_uri = f"runs:/{best_run_id}/model"
model_details = mlflow.register_model(model_uri, reg_name)

output = {
    "registered_model_name": reg_name,
    "version": int(model_details.version),
    "run_id": best_run_id,
    "source_metric": "mae",
    "source_metric_value": mae
}

with open("results/step3_s6.json", "w") as f:
    json.dump(output, f, indent=4)

print(f"Task 3 Complete. Version {model_details.version} registered.")
