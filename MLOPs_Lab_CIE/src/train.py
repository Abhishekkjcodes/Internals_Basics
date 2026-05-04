import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Setup
mlflow.set_experiment("adpulse-click-through-rate")

def train_model(model, name, X_train, X_test, y_train, y_test):
    with mlflow.start_run(run_name=name):
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)
        
        # LOGGING THE ACTUAL MODEL (Crucial for Task 3)
        mlflow.sklearn.log_model(model, "model")
        
        mlflow.log_params(model.get_params())
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.set_tag("priority", "high")
        
        return {"name": name, "mae": mae, "rmse": rmse, "r2": r2}

df = pd.read_csv("data/training_data.csv")
X = df.drop("click_through_rate", axis=1)
y = df["click_through_rate"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and log
ridge_res = train_model(Ridge(), "Ridge", X_train, X_test, y_train, y_test)
gb_res = train_model(GradientBoostingRegressor(random_state=42), "GradientBoosting", X_train, X_test, y_train, y_test)

results_list = [ridge_res, gb_res]
best_model_info = min(results_list, key=lambda x: x['mae'])

output = {
    "experiment_name": "adpulse-click-through-rate",
    "models": results_list,
    "best_model": best_model_info["name"],
    "best_metric_name": "mae",
    "best_metric_value": best_model_info["mae"]
}

os.makedirs("results", exist_ok=True)
with open("results/step1_s1.json", "w") as f:
    json.dump(output, f, indent=4)

print(f"Task 1 Fixed. Model saved for run.")
