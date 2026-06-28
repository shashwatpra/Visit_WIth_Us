from huggingface_hub.utils import RepositoryNotFoundError
from huggingface_hub import HfApi, create_repo
import os

hf_token = os.environ.get('HF_TOKEN')
api = HfApi(token=hf_token)

username = api.whoami(token=hf_token)['name']
repo_id = f'{username}/tourism-dataset'
repo_type = 'dataset'

# Step 1: Check if the repo exists, create if not
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Repo '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Repo '{repo_id}' not found. Creating...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Repo '{repo_id}' created.")

# Step 2: Upload the raw tourism.csv
api.upload_file(
    path_or_fileobj='tourism_project/data/tourism.csv',
    path_in_repo='tourism.csv',
    repo_id=repo_id,
    repo_type=repo_type,
)
print('Raw dataset registered successfully.')
