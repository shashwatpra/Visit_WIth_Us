"""
Hosting Script:
1. Initialize the HF token to establish connection with Hugging Face.
2. Define the Space repo_id (Spaces are the HF equivalent of a hosted app).
3. Create the Space repo if it does not exist (repo_type='space', sdk='streamlit').
4. Upload the deployment folder (app.py, requirements.txt, Dockerfile) to the Space.
"""
import os
from huggingface_hub import HfApi, login
from huggingface_hub.utils import RepositoryNotFoundError

hf_token = os.environ.get('HF_TOKEN')
if hf_token:
    login(token=hf_token)

api = HfApi()
username = api.whoami(token=hf_token)['name']
space_repo_id = f'{username}/tourism-package-app'

# Create HF Space if not exists
try:
    api.repo_info(repo_id=space_repo_id, repo_type='space')
    print(f"Space '{space_repo_id}' already exists.")
except RepositoryNotFoundError:
    api.create_repo(
        repo_id=space_repo_id,
        repo_type='space',
        space_sdk='streamlit',
        exist_ok=True
    )
    print(f"Space '{space_repo_id}' created.")

# Upload deployment folder to the Space
api.upload_folder(
    folder_path='tourism_project/deployment',  # local folder containing app.py, requirements.txt, Dockerfile
    repo_id=space_repo_id,
    repo_type='space'
)
print(f'Deployment files uploaded to Space: https://huggingface.co/spaces/{space_repo_id}')
