from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo
from huggingface_hub import HfApi
from pathlib import Path
import os

# EARLY DEBUG: check structure before any API call
cwd = Path.cwd()
print("=" * 40)
print("[DEBUG] Starting data_register.py")
print("[DEBUG] Current working directory:", cwd)
print("[DEBUG] Contents of current dir:", os.listdir(cwd))
print("[DEBUG] Contents of 'mlops':", os.listdir(cwd / "mlops") if (cwd / "mlops").exists() else "mlops not found")
print("[DEBUG] Contents of 'mlops/data':", os.listdir(cwd / "mlops" / "data") if (cwd / "mlops" / "data").exists() else "mlops/data not found")
print("=" * 40)

# Define path
data_path = cwd / "mlops" / "data"

# Check existence before uploading
if not data_path.is_dir():
    print(f"[ERROR] The directory '{data_path}' does not exist or is not a folder!")
    exit(1)

# Proceed with upload
print("[INFO] Uploading files:", os.listdir(data_path))

api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path=str(data_path),
    path_in_repo="data",
    repo_id="harasar/bank-customer-churn",
    repo_type="space"
)

