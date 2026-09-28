import json
import os
from typing import Dict, Optional


def get_timing_path() -> Optional[str]:
    path = os.environ.get("QMLLM_TIMING_PATH", "").strip()
    return path or None


def save_timing_payload(payload: Dict) -> None:
    path = get_timing_path()
    if not path:
        return
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[TIMING] saved timing payload: {path}")
