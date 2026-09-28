import csv
import json
import os
import time
import torch

from qmllm.methods.glmi.policy import build_glmi_policy
from qmllm.methods.mbq.quantize.pre_quant import run_mbq as run_mbq_base, apply_mbq as apply_mbq_base
from qmllm.methods.mbq_fg.quantize.pre_quant import run_mbq as run_mbq_fg, apply_mbq as apply_mbq_fg
from qmllm.methods.mbq.quantize.quantizer import pseudo_quantize_model_weight, pseudo_quantize_model_weight_act
from qmllm.utils.timing import save_timing_payload


def _save_glmi_policy_artifacts(scale_path: str, glmi_policy: dict) -> None:
    base, _ = os.path.splitext(scale_path)
    json_path = f"{base}_policy.json"
    csv_path = f"{base}_policy.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(glmi_policy, f, indent=2)

    layers = glmi_policy.get("layers", {})
    rows = []
    for layer_id, layer in sorted(layers.items(), key=lambda kv: int(kv[0])):
        comps = layer.get("components", {})
        weight_policy = layer.get("weight_policy", {})
        act_policy = layer.get("activation_policy", {})
        rows.append(
            {
                "layer_id": int(layer_id),
                "weight_grid": weight_policy.get("n_grid"),
                "weight_ratio_max": weight_policy.get("ratio_max"),
                "weight_signed_score": weight_policy.get("signed_score"),
                "weight_signed_score_raw": weight_policy.get("signed_score_raw"),
                "weight_signed_score_clipped": weight_policy.get("signed_score_clipped"),
                "weight_signed_norm_score": weight_policy.get("signed_norm_score"),
                "weight_budget_score": weight_policy.get("budget_score"),
                "weight_budget_norm_score": weight_policy.get("budget_norm_score"),
                "act_grid": act_policy.get("n_grid"),
                "act_ratio_max": act_policy.get("ratio_max"),
                "act_signed_score": act_policy.get("signed_score"),
                "act_signed_norm_score": act_policy.get("signed_norm_score"),
                "act_budget_score": act_policy.get("budget_score"),
                "act_budget_norm_score": act_policy.get("budget_norm_score"),
                "search_weight_share": layer.get("search_policy", {}).get("weight_share"),
                "search_activation_share": layer.get("search_policy", {}).get("activation_share"),
                "search_grid": layer.get("search_policy", {}).get("n_grid"),
                "search_ratio_max": layer.get("search_policy", {}).get("ratio_max"),
                "search_signed_score": layer.get("search_policy", {}).get("signed_score"),
                "search_signed_norm_score": layer.get("search_policy", {}).get("signed_norm_score"),
                "search_budget_score": layer.get("search_policy", {}).get("budget_score"),
                "search_budget_norm_score": layer.get("search_policy", {}).get("budget_norm_score"),
                "reweight_ratio": layer.get("reweight_ratio"),
                "conflict_score": layer.get("conflict_score"),
                "weight_conflict_score": layer.get("weight_conflict_score"),
                "activation_conflict_score": layer.get("activation_conflict_score"),
                "conflict_norm_score": layer.get("conflict_norm_score"),
                "weight_delta_loss": comps.get("weight", {}).get("delta_loss"),
                "weight_harmful_score": comps.get("weight", {}).get("harmful_score"),
                "weight_beneficial_score": comps.get("weight", {}).get("beneficial_score"),
                "vision_delta_loss": comps.get("vision", {}).get("delta_loss"),
                "vision_harmful_score": comps.get("vision", {}).get("harmful_score"),
                "vision_beneficial_score": comps.get("vision", {}).get("beneficial_score"),
                "text_delta_loss": comps.get("text", {}).get("delta_loss"),
                "text_harmful_score": comps.get("text", {}).get("harmful_score"),
                "text_beneficial_score": comps.get("text", {}).get("beneficial_score"),
            }
        )

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "layer_id",
                "weight_grid",
                "weight_ratio_max",
                "weight_signed_score",
                "weight_signed_score_raw",
                "weight_signed_score_clipped",
                "weight_signed_norm_score",
                "weight_budget_score",
                "weight_budget_norm_score",
                "act_grid",
                "act_ratio_max",
                "act_signed_score",
                "act_signed_norm_score",
                "act_budget_score",
                "act_budget_norm_score",
                "search_weight_share",
                "search_activation_share",
                "search_grid",
                "search_ratio_max",
                "search_signed_score",
                "search_signed_norm_score",
                "search_budget_score",
                "search_budget_norm_score",
                "reweight_ratio",
                "conflict_score",
                "weight_conflict_score",
                "activation_conflict_score",
                "conflict_norm_score",
                "weight_delta_loss",
                "weight_harmful_score",
                "weight_beneficial_score",
                "vision_delta_loss",
                "vision_harmful_score",
                "vision_beneficial_score",
                "text_delta_loss",
                "text_harmful_score",
                "text_beneficial_score",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    if rows:
        weight_grids = [row["weight_grid"] for row in rows if isinstance(row["weight_grid"], int)]
        act_grids = [row["act_grid"] for row in rows if isinstance(row["act_grid"], int)]
        if weight_grids and act_grids:
            print(
                "[BFQ] saved policy artifacts: "
                f"{json_path}, {csv_path} | "
                f"weight_grid_mean={sum(weight_grids)/len(weight_grids):.2f} "
                f"act_grid_mean={sum(act_grids)/len(act_grids):.2f}"
            )
        else:
            print(f"[BFQ] saved policy artifacts: {json_path}, {csv_path}")


def glmi_entry(
    model,
    prompt_inputs,
    prompt_kwargs,
    run_glmi_process: bool,
    pseudo_quant: bool,
    scale_path: str = None,
    zero_point: bool = True,
    q_group_size: int = 128,
    w_bit: int = 4,
    a_bit: int = 16,
    wa_quant: bool = False,
    distort: bool = False,
    loss_mode: str = "mae",
    glmi_policy_n_samples: int = 16,
    glmi_micro_batch_size: int = 4,
    glmi_probe_a_bit: int = 8,
    glmi_beta: float = 0.5,
    glmi_n_grid_max: int = 48,
    glmi_n_grid_target_mean: float = 20.0,
    glmi_budget_conflict_alpha: float = 0.5,
    glmi_budget_temperature: float = 8.0,
    glmi_aggressiveness_temperature: float = 3.0,
    glmi_ratio_upper_min: float = 0.5,
    glmi_ratio_upper_max: float = 1.0,
    glmi_modality_ratio_min: float = 0.25,
    glmi_modality_ratio_max: float = 4.0,
    glmi_weight_signed_clip_percentile: float = 100.0,
    glmi_fixed_budget: bool = False,
    glmi_fixed_aggressiveness: bool = False,
    glmi_upper_tail_budget: bool = False,
    glmi_budget_base_grid: int = 16,
    glmi_budget_bonus_low_percentile: float = 50.0,
    glmi_budget_bonus_high_percentile: float = 95.0,
    glmi_budget_bonus_gamma: float = 2.0,
    glmi_budget_bonus_threshold_mode: str = "percentile",
    glmi_budget_bonus_threshold_alpha: float = 1.0,
    glmi_lower_tail_penalty: bool = False,
    glmi_budget_penalty_low_percentile: float = 5.0,
    glmi_budget_penalty_high_percentile: float = 20.0,
    glmi_budget_penalty_gamma: float = 2.0,
    glmi_conflict_mode: str = "shared",
    glmi_budget_target_mode: str = "both",
    glmi_enable_attention_token_weight: bool = False,
    glmi_importance_cache_path: str = None,
    glmi_disable_importance_cache: bool = False,
    glmi_importance_only: bool = False,
    glmi_policy_override_path: str = None,
    lagq_disable_logit_sensitivity: bool = False,
    lagq_disable_attention_token_weight: bool = False,
    lagq_micro_batch_size: int = 0,
    lagq_stage1_metric: str = "full_logit",
    lagq_text_attn_correction: str = "exposure_baseline_ratio",
    lagq_vision_query_source: str = "both",
    lagq_query_mix_alpha: float = 0.5,
):
    q_config = {
        "zero_point": zero_point,
        "q_group_size": q_group_size,
    }

    assert scale_path is not None
    scale_exist = os.path.exists(scale_path)

    use_attention_token_weight = bool(glmi_enable_attention_token_weight)
    selected_run_mbq = run_mbq_fg if use_attention_token_weight else run_mbq_base
    selected_apply_mbq = apply_mbq_fg if use_attention_token_weight else apply_mbq_base

    if run_glmi_process and not scale_exist:
        model.to_cuda()
        timing_payload = {
            "method": "ours",
            "analysis_label": "quantization_effect_policy",
            "search_label": "cwe_scale_search",
            "analysis_seconds": 0.0,
            "search_seconds": 0.0,
        }
        total_start = time.perf_counter()
        if glmi_policy_override_path:
            with open(glmi_policy_override_path, "r", encoding="utf-8") as f:
                glmi_policy = json.load(f)
            print(f"[BFQ] using policy override: {glmi_policy_override_path}")
        else:
            analysis_start = time.perf_counter()
            glmi_policy = build_glmi_policy(
                process_model=model,
                prompt_inputs=prompt_inputs,
                prompt_kwargs=prompt_kwargs,
                w_bit=w_bit,
                a_bit=glmi_probe_a_bit,
                w_group=q_group_size,
                wa_quant=wa_quant,
                policy_n_samples=glmi_policy_n_samples,
                micro_batch_size=glmi_micro_batch_size,
                beta=glmi_beta,
                n_grid_max=glmi_n_grid_max,
                n_grid_target_mean=glmi_n_grid_target_mean,
                budget_conflict_alpha=glmi_budget_conflict_alpha,
                budget_temperature=glmi_budget_temperature,
                aggressiveness_temperature=glmi_aggressiveness_temperature,
                ratio_upper_min=glmi_ratio_upper_min,
                ratio_upper_max=glmi_ratio_upper_max,
                modality_ratio_min=glmi_modality_ratio_min,
                modality_ratio_max=glmi_modality_ratio_max,
                weight_signed_clip_percentile=glmi_weight_signed_clip_percentile,
                fixed_budget=glmi_fixed_budget,
                fixed_aggressiveness=glmi_fixed_aggressiveness,
                upper_tail_budget=glmi_upper_tail_budget,
                budget_base_grid=glmi_budget_base_grid,
                budget_bonus_low_percentile=glmi_budget_bonus_low_percentile,
                budget_bonus_high_percentile=glmi_budget_bonus_high_percentile,
                budget_bonus_gamma=glmi_budget_bonus_gamma,
                budget_bonus_threshold_mode=glmi_budget_bonus_threshold_mode,
                budget_bonus_threshold_alpha=glmi_budget_bonus_threshold_alpha,
                lower_tail_penalty=glmi_lower_tail_penalty,
                budget_penalty_low_percentile=glmi_budget_penalty_low_percentile,
                budget_penalty_high_percentile=glmi_budget_penalty_high_percentile,
                budget_penalty_gamma=glmi_budget_penalty_gamma,
                conflict_mode=glmi_conflict_mode,
                budget_target_mode=glmi_budget_target_mode,
                importance_cache_path=glmi_importance_cache_path,
                disable_importance_cache=glmi_disable_importance_cache,
            )
            timing_payload["analysis_seconds"] += time.perf_counter() - analysis_start
        try:
            model.to_cpu()
        except NotImplementedError:
            pass

        if glmi_importance_only:
            print("[BFQ] importance-only mode: stopping after policy analysis/cache save")
            timing_payload["total_seconds"] = time.perf_counter() - total_start
            timing_payload["overhead_seconds"] = max(
                0.0,
                float(timing_payload["total_seconds"])
                - float(timing_payload.get("analysis_seconds", 0.0))
                - float(timing_payload.get("search_seconds", 0.0)),
            )
            save_timing_payload(timing_payload)
            return model

        mbq_kwargs = dict(
            model=model,
            prompt_inputs=prompt_inputs,
            prompt_kwargs=prompt_kwargs,
            w_bit=w_bit,
            a_bit=a_bit,
            q_config=q_config,
            auto_scale=True,
            loss_mode=loss_mode,
            wa_quant=wa_quant,
            reweight=False,
            distort=distort,
            glmi_policy=glmi_policy,
        )
        if use_attention_token_weight:
            mbq_kwargs.update(
                finegrained=True,
                finegrained_mode="attention",
                lagq_disable_logit_sensitivity=lagq_disable_logit_sensitivity,
                lagq_disable_attention_token_weight=lagq_disable_attention_token_weight,
                lagq_micro_batch_size=lagq_micro_batch_size,
                lagq_stage1_metric=lagq_stage1_metric,
                lagq_text_attn_correction=lagq_text_attn_correction,
                lagq_vision_query_source=lagq_vision_query_source,
                lagq_query_mix_alpha=lagq_query_mix_alpha,
            )

        search_start = time.perf_counter()
        mbq_results = selected_run_mbq(**mbq_kwargs)
        timing_payload["search_seconds"] += time.perf_counter() - search_start
        timing_payload["total_seconds"] = time.perf_counter() - total_start
        timing_payload["search_seconds"] = max(
            0.0,
            float(timing_payload["total_seconds"]) - float(timing_payload.get("analysis_seconds", 0.0)),
        )
        timing_payload["overhead_seconds"] = 0.0
        save_timing_payload(timing_payload)
        mbq_results["glmi_policy"] = glmi_policy
        mbq_results["glmi_enable_attention_token_weight"] = use_attention_token_weight

        dirpath = os.path.dirname(scale_path)
        os.makedirs(dirpath, exist_ok=True)
        torch.save(mbq_results, scale_path)
        _save_glmi_policy_artifacts(scale_path, glmi_policy)
        print("GLMI results saved at", scale_path)

    if pseudo_quant:
        mbq_results = torch.load(scale_path, map_location="cpu")
        selected_apply_mbq(model.model, mbq_results)
        if not wa_quant:
            pseudo_quantize_model_weight(model.model, w_bit=w_bit, q_config=q_config)
        else:
            pseudo_quantize_model_weight_act(model.model, w_bit=w_bit, a_bit=a_bit)

    model.to_cuda()
    return model
