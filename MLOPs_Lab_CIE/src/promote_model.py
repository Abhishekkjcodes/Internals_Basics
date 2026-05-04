import pandas as pd
import mlflow
import mlflow.sklearn
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from mlflow.tracking import MlflowClient

# 1. Setup
client = MlflowClient()
model_name = "adpulse-click-through-rate-predictor"
df = pd.read_csv("data/training_data.csv")
X = df.drop("click_through_rate", axis=1)
y = df["click_through_rate"]

# 2. Train Challenger (Version 2) with random_state=99
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=99)
challenger = GradientBoostingRegressor(random_state=99)

with mlflow.start_run(run_name="challenger_model"):
    challenger.fit(X_train, y_train)
    c_mae = mean_absolute_error(y_test, challenger.predict(X_test))
    mlflow.log_metric("mae", c_mae)
    mlflow.sklearn.log_model(challenger, "model")
    run_id_2 = mlflow.active_run().info.run_id

# 3. Register Version (This will become Version 3 or higher since you ran it before)
v_details = mlflow.register_model(f"runs:/{run_id_2}/model", model_name)
new_version = int(v_details.version)

# 4. Get Version 1 Run ID to find its MAE
v1_info = client.get_model_version(model_name, 1)
v1_run = client.get_run(v1_info.run_id)
v1_mae = v1_run.data.metrics.get("mae", 999.0)

# 5. Promotion Logic
action = "kept_champion"
winner_version = 1

if c_mae < v1_mae:
    action = "promoted"
    winner_version = new_version

# Assign "live" alias to the winner
client.set_registered_model_alias(model_name, "live", str(winner_version))

# 6. Save JSON Result
output = {
    "registered_model_name": model_name,
    "alias_name": "live",
    "champion_version": 1,
    "challenger_version": new_version,
    "action": action
}

os.makedirs("results", exist_ok=True)
with open("results/step4_s7.json", "w") as f:
    json.dump(output, f, indent=4)

print(f"Task 4 Complete. Winner Version: {winner_version}")
