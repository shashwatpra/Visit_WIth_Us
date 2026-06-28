"""
Model Training Script:
1.  Set MLflow tracking URI (SQLite backend for local; localhost:5000 for server).
2.  Set experiment name: 'tourism-Package-Prediction-Experiment'.
3.  Establish connection with HF using HF token.
4.  Read Xtrain.csv, Xtest.csv, ytrain.csv, ytest.csv from HF dataset repo.
5.  Load the datasets into DataFrames.
6.  Separate numerical and categorical variables.
7.  Preprocess data using make_column_transformer (imputation + scaling/encoding).
8.  Initialize base models: RF, AdaBoost, GradientBoosting, XGBoost, etc.
9.  Define hyperparameter grids for each model.
10. Build a model pipeline (preprocessor + classifier).
11. Start MLflow run, run GridSearchCV, log params and metrics.
12. Get best model, evaluate on train and test data.
13. Log Accuracy, Recall, Precision, F1 Score to MLflow.
14. Save model with joblib and log to MLflow.
15. Upload model.pkl to HF Model Hub repo.
"""
import os
import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    BaggingClassifier, RandomForestClassifier,
    AdaBoostClassifier, GradientBoostingClassifier
)
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from huggingface_hub import HfApi, login
from huggingface_hub.utils import RepositoryNotFoundError

# --- Hugging Face Auth ---
hf_token = os.environ.get('HF_TOKEN')
if hf_token:
    login(token=hf_token)
api = HfApi()
username = api.whoami(token=hf_token)['name']
dataset_repo_id = f'{username}/tourism-dataset'
model_repo_id = f'{username}/tourism-package-model'

# --- Configure MLflow ---
try:
    import requests
    requests.get('http://localhost:5000', timeout=1)
    mlflow.set_tracking_uri('http://localhost:5000')
except Exception:
    # Use SQLite backend (supported by all modern MLflow versions)
    mlflow.set_tracking_uri('sqlite:///mlruns.db')

mlflow.set_experiment('tourism-Package-Prediction-Experiment')

# --- Load Data from Hugging Face ---
Xtrain_path = f'hf://datasets/{dataset_repo_id}/Xtrain.csv'
Xtest_path  = f'hf://datasets/{dataset_repo_id}/Xtest.csv'
ytrain_path = f'hf://datasets/{dataset_repo_id}/ytrain.csv'
ytest_path  = f'hf://datasets/{dataset_repo_id}/ytest.csv'

X_train = pd.read_csv(Xtrain_path)
X_test  = pd.read_csv(Xtest_path)
y_train = pd.read_csv(ytrain_path).squeeze()
y_test  = pd.read_csv(ytest_path).squeeze()

# --- Identify Column Types ---
num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X_train.select_dtypes(exclude=[np.number]).columns.tolist()

# --- Preprocessing Pipeline ---
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])
preprocessor = ColumnTransformer(transformers=[
    ('num', num_transformer, num_cols),
    ('cat', cat_transformer, cat_cols)
])

# --- Define Models and Hyperparameters ---
models = {
    'Decision Tree': {
        'model': DecisionTreeClassifier(random_state=42),
        'params': {
            'classifier__max_depth': [3, 5, 10, None],
            'classifier__min_samples_split': [2, 5, 10]
        }
    },
    'Bagging': {
        'model': BaggingClassifier(random_state=42),
        'params': {'classifier__n_estimators': [10, 30, 50]}
    },
    'Random Forest': {
        'model': RandomForestClassifier(random_state=42),
        'params': {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__max_depth': [5, 10, None]
        }
    },
    'AdaBoost': {
        'model': AdaBoostClassifier(random_state=42),
        'params': {
            'classifier__n_estimators': [50, 100],
            'classifier__learning_rate': [0.1, 0.5, 1.0]
        }
    },
    'Gradient Boosting': {
        'model': GradientBoostingClassifier(random_state=42),
        'params': {
            'classifier__n_estimators': [50, 100],
            'classifier__learning_rate': [0.05, 0.1],
            'classifier__max_depth': [3, 5]
        }
    },
    'XGBoost': {
        'model': XGBClassifier(random_state=42, eval_metric='logloss'),
        'params': {
            'classifier__n_estimators': [50, 100],
            'classifier__learning_rate': [0.05, 0.1, 0.2],
            'classifier__max_depth': [3, 5, 7]
        }
    }
}

best_f1 = 0.0
best_pipeline = None
best_model_name = ''

for name, m_info in models.items():
    print(f'Training and tuning: {name}...')
    with mlflow.start_run(run_name=name):
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', m_info['model'])
        ])
        grid = GridSearchCV(pipeline, m_info['params'], cv=3, scoring='f1', n_jobs=-1)
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test)

        # Evaluate metrics
        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec  = recall_score(y_test, y_pred, zero_division=0)
        f1   = f1_score(y_test, y_pred, zero_division=0)

        print(f'  Best params : {grid.best_params_}')
        print(f'  F1={f1:.4f}  Acc={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}')

        # Log parameters and metrics to MLflow
        for param_name, param_val in grid.best_params_.items():
            mlflow.log_param(param_name, param_val)
        mlflow.log_metric('accuracy',  acc)
        mlflow.log_metric('precision', prec)
        mlflow.log_metric('recall',    rec)
        mlflow.log_metric('f1_score',  f1)

        # Log model artifact to MLflow
        mlflow.sklearn.log_model(best_model, 'model', skops_trusted_types=['numpy.dtype', 'xgboost.core.Booster', 'xgboost.sklearn.XGBClassifier'])

        if f1 > best_f1:
            best_f1 = f1
            best_pipeline = best_model
            best_model_name = name

print(f'\nBest Overall Model: {best_model_name} with F1-Score: {best_f1:.4f}')

# Save best model locally
os.makedirs('tourism_project/model_building', exist_ok=True)
joblib.dump(best_pipeline, 'tourism_project/model_building/model.pkl')
print('Model saved locally as model.pkl')

# Upload model to Hugging Face Model Hub
try:
    api.repo_info(repo_id=model_repo_id, repo_type='model')
    print(f"Model repo '{model_repo_id}' already exists.")
except RepositoryNotFoundError:
    api.create_repo(repo_id=model_repo_id, repo_type='model', exist_ok=True)
    print(f"Model repo '{model_repo_id}' created.")

api.upload_file(
    path_or_fileobj='tourism_project/model_building/model.pkl',
    path_in_repo='model.pkl',
    repo_id=model_repo_id,
    repo_type='model'
)
print('Model training finished and uploaded to Hugging Face successfully.')
