import importlib
import os
import sys

import hf_transfer
from loguru import logger

from qmllm.models.internvl2 import InternVL2
from qmllm.models.llava_onevision import LLaVA_onevision
from qmllm.models.qwen2_vl import Qwen2_VL
try:
    from qmllm.models.llava_v15 import LLaVA_v15  # noqa: F401
except Exception as e:
    print(f"LLaVA v1.5 not available (skipping register): {e}")

try:
    from qmllm.models.vila import vila  # noqa: F401
except Exception as e:
    print(f"VILA not available (skipping register): {e}")

from qmllm.utils.registry import MODEL_REGISTRY

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

logger.remove()
logger.add(sys.stdout, level="WARNING")

def get_process_model(model_name):
    return MODEL_REGISTRY[model_name]
