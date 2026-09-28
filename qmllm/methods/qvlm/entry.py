import os
import time
import torch

from qmllm.methods.qvlm.quantize.pre_quant_qvlm import run_qvlm
from qmllm.methods.mbq_fg.quantize.pre_quant import apply_mbq
from qmllm.methods.mbq_fg.quantize.quantizer import (
    pseudo_quantize_model_weight,
    pseudo_quantize_model_weight_act,
)
from qmllm.utils.timing import save_timing_payload


def qvlm_entry(
    model,
    prompt_inputs,
    prompt_kwargs,
    run_qvlm_process: bool,
    pseudo_quant: bool,
    scale_path: str = None,
    q_group_size: int = 128,
    w_bit: int = 4,
    a_bit: int = 8,
    wa_quant: bool = False,
    loss_mode: str = "mae",
):
    """
    Minimal Q-VLM-style PTQ hook.

    Reuses the MBQ calibration pipeline (calibration data & hooks) but leaves room to plug in
    entropy-based block search later. This keeps interface aligned with other methods while
    enabling experiments at W4/A8 (or user-specified w_bit/a_bit).
    """
    q_config = {
        "zero_point": True,
        "q_group_size": q_group_size,
    }

    assert scale_path is not None, "scale_path is required for QVLM"

    scale_exist = os.path.exists(scale_path)

    if run_qvlm_process and not scale_exist:
        # Some models (e.g., OneVision) may refuse .to('cpu') on meta tensors.
        try:
            model.to_cpu()
        except NotImplementedError:
            pass

        timing_payload = {
            "method": "qvlm",
            "analysis_label": "entropy_partition",
            "search_label": "cwe_scale_search",
            "analysis_seconds": 0.0,
            "search_seconds": 0.0,
        }
        total_start = time.perf_counter()
        search_start = time.perf_counter()
        mbq_results = run_qvlm(
            model,
            prompt_inputs,
            prompt_kwargs,
            w_bit=w_bit,
            a_bit=a_bit,
            q_config=q_config,
            loss_mode=loss_mode,
            wa_quant=wa_quant,
            desc="Running QVLM...",
            timing_log=timing_payload,
        )
        if float(timing_payload.get("search_seconds", 0.0)) <= 0.0:
            timing_payload["search_seconds"] += time.perf_counter() - search_start
        timing_payload["total_seconds"] = time.perf_counter() - total_start
        timing_payload["search_seconds"] = max(
            0.0,
            float(timing_payload["total_seconds"]) - float(timing_payload.get("analysis_seconds", 0.0)),
        )
        timing_payload["overhead_seconds"] = 0.0
        save_timing_payload(timing_payload)

        dirpath = os.path.dirname(scale_path)
        os.makedirs(dirpath, exist_ok=True)
        torch.save(mbq_results, scale_path)
        print("Q-VLM scales saved at", scale_path)

    if pseudo_quant:
        mbq_results = torch.load(scale_path, map_location="cpu")
        apply_mbq(model.model, mbq_results)

        if not wa_quant:
            pseudo_quantize_model_weight(model.model, w_bit=w_bit, q_config=q_config)
        else:
            pseudo_quantize_model_weight_act(model.model, w_bit=w_bit, a_bit=a_bit)

    model.to_cuda()
    return model
