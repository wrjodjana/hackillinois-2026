import os
import uuid
from typing import Dict

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data_store")
os.makedirs(DATA_DIR, exist_ok=True)

# Simple in-memory registry: dataset_id -> path
_DATASETS: Dict[str, str] = {}


def new_dataset_id() -> str:
    return f"ds_{uuid.uuid4().hex[:12]}"


def save_dataset_bytes(dataset_id: str, filename: str, content: bytes) -> str:
    safe_name = filename.replace("\\", "_").replace("/", "_")
    path = os.path.join(DATA_DIR, f"{dataset_id}__{safe_name}")
    with open(path, "wb") as f:
        f.write(content)
    _DATASETS[dataset_id] = path
    return path


def get_dataset_path(dataset_id: str) -> str:
    if dataset_id not in _DATASETS:
        raise KeyError("Unknown dataset_id (upload first, or server restarted).")
    return _DATASETS[dataset_id]