# for data manipulation
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, recall_score
# for model serialization
import joblib
# for creating a folder
import os
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("mlops-training-experiment")

api = HfApi()


Xtrain_path = "hf://datasets/harasar/Tourist-Customer-Taken/Xtrain.csv"
Xtest_path = "hf://datasets/harasar/Tourist-Customer-Taken/Xtest.csv"
ytrain_path = "hf://datasets/harasar/Tourist-Customer-Taken/ytrain.csv"
ytest_path = "hf://datasets/harasar/Tourist-Customer-Taken/ytest.csv"

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path)
ytest = pd.read_csv(ytest_path)


# List of numerical features in the dataset
numeric_features = [
    'Age',               # Customer's age
    'CityTier',            # Number of years the customer has been with the bank
    'DurationOfPitch',           # Customer’s account balance
    'NumberOfPersonVisiting',     # Number of products the customer has with the bank
    'NumberOfFollowups',         # Whether the customer has a credit card (binary: 0 or 1)
    'PreferredPropertyStar',    # Whether the customer is an active member (binary: 0 or 1)
    'NumberOfTrips',            # Customer’s estimated salary
    'Passport',
    'PitchSatisfactionScore',
    'OwnCar',
    'NumberOfChildrenVisiting',
    'MonthlyIncome',


]

# List of categorical features in the dataset
categorical_features = [
    'TypeofContact', 
    'Occupation',
    'Gender', 
    'ProductPitched',  
    'MaritalStatus', 
    'Designation' ,  
]


# Set the clas weight to handle class imbalance
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]
class_weight

# Define the preprocessing steps
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)

# Define base XGBoost model
xgb_model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42)

# Define hyperparameter grid
param_grid = {
    'xgbclassifier__n_estimators': [50, 75, 100, 125, 150],    # number of tree to build
    'xgbclassifier__max_depth': [2, 3, 4],    # maximum depth of each tree
    'xgbclassifier__colsample_bytree': [0.4, 0.5, 0.6],    # percentage of attributes to be considered (randomly) for each tree
    'xgbclassifier__colsample_bylevel': [0.4, 0.5, 0.6],    # percentage of attributes to be considered (randomly) for each level of a tree
    'xgbclassifier__learning_rate': [0.01, 0.05, 0.1],    # learning rate
    'xgbclassifier__reg_lambda': [0.4, 0.5, 0.6],    # L2 regularization factor
}

# Model pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)

# Start MLflow run
with mlflow.start_run(nested=True):
    # Hyperparameter tuning
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(Xtrain, ytrain)

    # Log all parameter combinations and their mean test scores
    results = grid_search.cv_results_
    for i in range(len(results['params'])):
        param_set = results['params'][i]
        mean_score = results['mean_test_score'][i]
        std_score = results['std_test_score'][i]

        # Log each combination as a separate MLflow run
        with mlflow.start_run(nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric("mean_test_score", mean_score)
            mlflow.log_metric("std_test_score", std_score)

    # Log best parameters separately in main run
    mlflow.log_params(grid_search.best_params_)

    # Store and evaluate the best model
    best_model = grid_search.best_estimator_

    classification_threshold = 0.45

    y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]
    y_pred_train = (y_pred_train_proba >= classification_threshold).astype(int)

    y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]
    y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)

    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

    # Log the metrics for the best model
    mlflow.log_metrics({
        "train_accuracy": train_report['accuracy'],
        "train_precision": train_report['1']['precision'],
        "train_recall": train_report['1']['recall'],
        "train_f1-score": train_report['1']['f1-score'],
        "test_accuracy": test_report['accuracy'],
        "test_precision": test_report['1']['precision'],
        "test_recall": test_report['1']['recall'],
        "test_f1-score": test_report['1']['f1-score']
    })


     # -------------------------
    # Random Forest Hyperparameter Tuning
    # -------------------------
    from sklearn.ensemble import RandomForestClassifier

    rf_model = RandomForestClassifier(random_state=42)

    rf_param_grid = {
        "randomforestclassifier__n_estimators": [100, 200],
        "randomforestclassifier__max_depth": [None, 5, 10],
        "randomforestclassifier__min_samples_split": [2, 5],
        "randomforestclassifier__min_samples_leaf": [1, 2]
    }

    rf_pipeline = make_pipeline(preprocessor, rf_model)

    rf_grid = GridSearchCV(
        rf_pipeline,
        param_grid=rf_param_grid,
        cv=5,
        n_jobs=-1,
        scoring="accuracy"
    )
    rf_grid.fit(Xtrain, ytrain)

# Log all parameter combinations and their mean test scores
rf_results = rf_grid.cv_results_
for i in range(len(rf_results['params'])):
    rf_param_set = rf_results['params'][i]
    rf_mean_score = rf_results['mean_test_score'][i]
    rf_std_score = rf_results['std_test_score'][i]

    # Log each combination as a separate MLflow run (nested)
    with mlflow.start_run(nested=True):
        mlflow.log_params(rf_param_set)
        mlflow.log_metric("rf_mean_test_score", rf_mean_score)
        mlflow.log_metric("rf_std_test_score", rf_std_score)

# Log best parameters separately in main run
mlflow.log_params(rf_grid.best_params_)

rf_best = rf_grid.best_estimator_
rf_pred_train = rf_best.predict(Xtrain)
rf_pred_test = rf_best.predict(Xtest)

rf_train_report = classification_report(ytrain, rf_pred_train, output_dict=True)
rf_test_report = classification_report(ytest, rf_pred_test, output_dict=True)

print("===== Random Forest Results =====")
print("Best Params:", rf_grid.best_params_)
print("Train Accuracy:", rf_train_report['accuracy'])
print("Test Accuracy:", rf_test_report['accuracy'])

mlflow.log_metrics({
    "rf_train_accuracy": rf_train_report['accuracy'],
    "rf_train_precision": rf_train_report['1']['precision'],
    "rf_train_recall": rf_train_report['1']['recall'],
    "rf_train_f1-score": rf_train_report['1']['f1-score'],
    "rf_test_accuracy": rf_test_report['accuracy'],
    "rf_test_precision": rf_test_report['1']['precision'],
    "rf_test_recall": rf_test_report['1']['recall'],
    "rf_test_f1-score": rf_test_report['1']['f1-score']
})

    #####################################################################

# -------------------------
    # Compare & Select Best
    # -------------------------
    xgb_acc = test_report['accuracy']
    rf_acc = rf_test_report['accuracy']

    if rf_acc > xgb_acc:
        best_model = rf_best
        best_name = "RandomForest"
        best_acc = rf_acc
    else:
        # keep the XGBoost best_model from above
        best_name = "XGBoost"
        best_acc = xgb_acc

    print(f"===== Selected Best Model: {best_name} with Accuracy = {best_acc} =====")

    # Save the chosen best model locally
    model_path = f"best_{best_name}_model_v1.joblib"
    joblib.dump(best_model, model_path)

    # Log which model was chosen
    mlflow.log_param("best_model", best_name)
    mlflow.log_metric("best_test_accuracy", best_acc)

    # Log the model artifact
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved as artifact at: {model_path}")

    ############################################################################

    # Log the model artifact
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved as artifact at: {model_path}")

    # Upload to Hugging Face
    repo_id = "harasar/churn-model"
    repo_type = "model"

    # Step 1: Check if the space exists
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"Space '{repo_id}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"Space '{repo_id}' not found. Creating new space...")
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
        print(f"Space '{repo_id}' created.")

    # create_repo("churn-model", repo_type="model", private=False)
    api.upload_file(
        path_or_fileobj=model_path,             # ✅ use correct variable
        path_in_repo=os.path.basename(model_path),
        repo_id=repo_id,
        repo_type=repo_type,
    )
