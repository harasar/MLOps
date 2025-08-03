from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo
import os
from pathlib import Path
from huggingface_hub import HfApi

# Get current working directory (GitHub Actions root)
cwd = Path.cwd()
print(f"[INFO] Current working directory: {cwd}")

# Set path explicitly
data_path = cwd / "mlops" / "data"
print(f"[INFO] Looking for dataset in: {data_path}")

# Check if the directory exists
if not data_path.is_dir():
    print(f"[ERROR] The directory '{data_path}' does not exist!")
    print("[DEBUG] ls mlops output:", os.listdir(cwd / "mlops") if (cwd / "mlops").exists() else "mlops missing")
    exit(1)

# List files inside data folder
print("[INFO] Files to upload:", os.listdir(data_path))

# Upload
api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path=str(data_path),
    path_in_repo="data",
    repo_id="harasar/bank-customer-churn",
    repo_type="space"
)
