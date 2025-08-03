from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo
import os
from pathlib import Path
from huggingface_hub import HfApi

# Use your preferred token method
api = HfApi(token=os.getenv("HF_TOKEN"))

# Resolve dataset path relative to this script, minimal change
data_path = Path(__file__).resolve().parents[1] / "data"

# ✅ Debug: show path info
print(f"[INFO] Current script path: {__file__}")
print(f"[INFO] Resolved data folder path: {data_path}")

# ✅ Safety check
if not data_path.is_dir():
    print(f"[ERROR] The directory '{data_path}' does not exist!")
    print("[DEBUG] CWD:", os.getcwd())
    print("[DEBUG] Files in mlops:", os.listdir(data_path.parent) if data_path.parent.exists() else "Missing parent folder")
    exit(1)

# ✅ Optional: list files
print("[INFO] Files to be uploaded:", os.listdir(data_path))

# Upload to Hugging Face
api.upload_folder(
    folder_path=str(data_path),
    path_in_repo="data",
    repo_id="harasar/bank-customer-churn",
    repo_type="space"
)
