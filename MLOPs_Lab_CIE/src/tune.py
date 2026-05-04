import pandas as pd
import mlflow
import json
import os
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

# 1. Setup
mlflow.set_experiment("adpulse-click-through-rate")
df = pd.read_csv("data/training_data.csv")
X = df.drop("click_through_rate", axis=1)
y = df["click_through_rate"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 2. Define Parameter Grid from the paper
param_dist = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth': [3, 5]
}

# 3. Hyperparameter Tuning with Parent Run and Nested Logging
with mlflow.start_run(run_name="tuning-adpulse") as parent_run:
    # Enable autologging to capture nested trials automatically
    mlflow.sklearn.autolog(log_datasets=False) 
    
    gb = GradientBoostingRegressor(random_state=42)
    random_search = RandomizedSearchCV(
        estimator=gb,
        param_distributions=param_dist,
        n_iter=10, 
        cv=5,      
        scoring='neg_mean_absolute_error',
        random_state=42,
        n_jobs=-1
    )
    
    random_search.fit(X_train, y_train)
    
    # Calculate best MAE on the test set
    best_model = random_search.best_estimator_
    predictions = best_model.predict(X_test)
    best_mae = mean_absolute_error(y_test, predictions)
    
    # 4. Save JSON Result as required by Task 2
    output = {
        "search_type": "random",
        "n_folds": 5,
        "total_trials": 10,
        "best_params": random_search.best_params_,
        "best_mae": best_mae,
        "best_cv_mae": abs(random_search.best_score_),
        "parent_run_name": "tuning-adpulse"
    }

    os.makedirs("results", exist_ok=True)
    with open("results/step2_s2.json", "w") as f:
        json.dump(output, f, indent=4)

print("Task 2 Complete. Best Params:", random_search.best_params_)
