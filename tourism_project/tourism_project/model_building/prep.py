"""
Data Preparation Script:
1. Establish connection with Hugging Face using HF token.
2. Read the raw tourism.csv from the HF dataset repo.
3. Clean data: drop irrelevant columns, fix typos.
4. Separate target (ProdTaken) and independent variables.
5. Split into train/test sets (80/20 stratified).
6. Save: Xtrain.csv, Xtest.csv, ytrain.csv, ytest.csv.
7. Upload all split files back to the HF dataset repo.
"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi, login

hf_token = os.environ.get('HF_TOKEN')
if hf_token:
    login(token=hf_token)
api = HfApi()
username = api.whoami(token=hf_token)['name']
repo_id = f'{username}/tourism-dataset'

# Load raw data from Hugging Face
DATASET_PATH = f'hf://datasets/{repo_id}/tourism.csv'
df = pd.read_csv(DATASET_PATH)

# Clean data
cols_to_drop = ['CustomerID']
for col in df.columns:
    if col == '' or 'Unnamed' in col:
        cols_to_drop.append(col)
df = df.drop(columns=cols_to_drop, errors='ignore')

# Fix anomalies
if 'Gender' in df.columns:
    df['Gender'] = df['Gender'].replace('Fe Male', 'Female')
if 'MaritalStatus' in df.columns:
    df['MaritalStatus'] = df['MaritalStatus'].replace('Unmarried', 'Single')

# Separate features and target
X = df.drop(columns=['ProdTaken'])
y = df['ProdTaken']

# Train/test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Save splits locally
os.makedirs('tourism_project/data', exist_ok=True)
X_train.to_csv('tourism_project/data/Xtrain.csv', index=False)
X_test.to_csv('tourism_project/data/Xtest.csv', index=False)
y_train.to_csv('tourism_project/data/ytrain.csv', index=False)
y_test.to_csv('tourism_project/data/ytest.csv', index=False)
print(f'Splits saved: X_train={X_train.shape}, X_test={X_test.shape}')

# Upload splits to Hugging Face
for fname in ['Xtrain.csv', 'Xtest.csv', 'ytrain.csv', 'ytest.csv']:
    api.upload_file(
        path_or_fileobj=f'tourism_project/data/{fname}',
        path_in_repo=fname,
        repo_id=repo_id,
        repo_type='dataset'
    )
print('Splits successfully uploaded to Hugging Face.')
