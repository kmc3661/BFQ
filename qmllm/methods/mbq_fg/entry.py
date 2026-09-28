import os
import time
import torch

from qmllm.methods.mbq_fg.quantize.pre_quant import run_mbq, apply_mbq
from qmllm.methods.mbq_fg.quantize.quantizer import pseudo_quantize_model_weight, pseudo_quantize_model_weight_act
from qmllm.utils.timing import save_timing_payload


def mbq_entry(
    model,
    prompt_inputs,
    prompt_kwargs,
    run_mbq_process: bool,
    pseudo_quant: bool,
    scale_path: str = None,
    zero_point: str = True,
    q_group_size: int = 128,
    w_bit: int = 4,
    a_bit: int = 16,
    wa_quant: bool = False,
    reweight: bool = False,
    distort: bool = False,
    loss_mode: str = "mae",
    finegrained: bool = False,
    qig_steps: int = 8,
    qig_iqr_factor: float = 1.5,
    qig_eps: float = 1e-6,
    qig_disable_iqr: bool = False,
    qig_use_abs: bool = True,
    finegrained_mode: str = "qig",
    lagq_layer_groups: str = "0-10,11-25,26-35",
    lagq_noise_bits: int = 8,
    lagq_noise_strength: float = 1.0,
    lagq_micro_batch_size: int = 0,
    lagq_stage1_samples: int = 0,
    lagq_disable_logit_sensitivity: bool = False,
    lagq_disable_attention_token_weight: bool = False,
    lagq_stage1_metric: str = "full_logit",
    lagq_text_attn_correction: str = "exposure_baseline_ratio",
    lagq_vision_query_source: str = "both",
    lagq_query_mix_alpha: float = 0.5,
):
    '''
    model: here the model is the LLM, you have to extract the LLM first! 
    prompt_tokens: the prompt tokens
    prompt_mask: the prompt mask, mask the answer language tokens
    run_mbq_process: whether to run the MBQ process
    '''
    q_config = {
        "zero_point": zero_point,  # by default True
        "q_group_size": q_group_size,  # whether to use group quantization
    }

    assert scale_path is not None

    scale_exist = os.path.exists(scale_path)
    # reparameterization
    if run_mbq_process and not scale_exist:
        # 일부 모델(예: OneVision)에서는 meta tensor로 인해 cpu 이동이 실패할 수 있음
        try:
            model.to_cpu()
        except NotImplementedError:
            pass
        analysis_label = "token_importance" if finegrained_mode == "qig" else "attention_token_weight"
        timing_payload = {
            "method": "qig" if finegrained_mode == "qig" else "lagq",
            "analysis_label": analysis_label,
            "search_label": "cwe_scale_search",
            "analysis_seconds": 0.0,
            "search_seconds": 0.0,
        }
        total_start = time.perf_counter()
        mbq_results = run_mbq(
            model,
            prompt_inputs,
            prompt_kwargs,
            w_bit=w_bit,
            a_bit=a_bit,
            q_config=q_config,
            auto_scale=True,
            loss_mode=loss_mode,
            wa_quant=wa_quant,
            reweight=reweight,
            distort=distort,
            finegrained=finegrained,
            qig_steps=qig_steps,
            qig_iqr_factor=qig_iqr_factor,
            qig_eps=qig_eps,
            qig_disable_iqr=qig_disable_iqr,
            qig_use_abs=qig_use_abs,
            finegrained_mode=finegrained_mode,
            lagq_layer_groups=lagq_layer_groups,
            lagq_noise_bits=lagq_noise_bits,
            lagq_noise_strength=lagq_noise_strength,
            lagq_micro_batch_size=lagq_micro_batch_size,
            lagq_stage1_samples=lagq_stage1_samples,
            lagq_disable_logit_sensitivity=lagq_disable_logit_sensitivity,
            lagq_disable_attention_token_weight=lagq_disable_attention_token_weight,
            lagq_stage1_metric=lagq_stage1_metric,
            lagq_text_attn_correction=lagq_text_attn_correction,
            lagq_vision_query_source=lagq_vision_query_source,
            lagq_query_mix_alpha=lagq_query_mix_alpha,
            timing_log=timing_payload,
        )
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
        print("MBQ results saved at", scale_path)

    if pseudo_quant:
        mbq_results = torch.load(scale_path, map_location="cpu")
        apply_mbq(model.model, mbq_results)

        if not wa_quant:
            # weight quantization
            pseudo_quantize_model_weight(model.model, w_bit=w_bit, q_config=q_config)
        else:
            # weight activation quantization
            pseudo_quantize_model_weight_act(model.model, w_bit=w_bit, a_bit=a_bit)

    model.to_cuda()
    return model
