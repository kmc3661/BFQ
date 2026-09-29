import csv
import json
import os
import time
import torch

from qmllm.methods.bfq.policy import build_bfq_policy
from qmllm.methods.mbq.quantize.pre_quant import run_mbq as run_mbq_base, apply_mbq as apply_mbq_base
from qmllm.methods.mbq_fg.quantize.pre_quant import run_mbq as run_mbq_fg, apply_mbq as apply_mbq_fg
from qmllm.methods.mbq.quantize.quantizer import pseudo_quantize_model_weight, pseudo_quantize_model_weight_act
from qmllm.utils.timing import save_timing_payload


def _save_bfq_policy_artifacts(scale_path: str, bfq_policy: dict) -> None:
    base, _ = os.path.splitext(scale_path)
    json_path = f"{base}_policy.json"
    csv_path = f"{base}_policy.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(bfq_policy, f, indent=2)

    layers = bfq_policy.get("layers", {})
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


def bfq_entry(
    model,
    prompt_inputs,
    prompt_kwargs,
    run_bfq_process: bool,
    pseudo_quant: bool,
    scale_path: str = None,
    zero_point: bool = True,
    q_group_size: int = 128,
    w_bit: int = 4,
    a_bit: int = 16,
    wa_quant: bool = False,
    distort: bool = False,
    loss_mode: str = "mae",
    bfq_policy_n_samples: int = 16,
    bfq_micro_batch_size: int = 4,
    bfq_probe_a_bit: int = 8,
    bfq_beta: float = 0.5,
    bfq_n_grid_max: int = 48,
    bfq_n_grid_target_mean: float = 20.0,
    bfq_budget_conflict_alpha: float = 0.5,
    bfq_budget_temperature: float = 8.0,
    bfq_aggressiveness_temperature: float = 3.0,
    bfq_ratio_upper_min: float = 0.5,
    bfq_ratio_upper_max: float = 1.0,
    bfq_modality_ratio_min: float = 0.25,
    bfq_modality_ratio_max: float = 4.0,
    bfq_weight_signed_clip_percentile: float = 100.0,
    bfq_fixed_budget: bool = False,
    bfq_fixed_aggressiveness: bool = False,
    bfq_upper_tail_budget: bool = False,
    bfq_budget_base_grid: int = 16,
    bfq_budget_bonus_low_percentile: float = 50.0,
    bfq_budget_bonus_high_percentile: float = 95.0,
    bfq_budget_bonus_gamma: float = 2.0,
    bfq_budget_bonus_threshold_mode: str = "percentile",
    bfq_budget_bonus_threshold_alpha: float = 1.0,
    bfq_lower_tail_penalty: bool = False,
    bfq_budget_penalty_low_percentile: float = 5.0,
    bfq_budget_penalty_high_percentile: float = 20.0,
    bfq_budget_penalty_gamma: float = 2.0,
    bfq_conflict_mode: str = "shared",
    bfq_budget_target_mode: str = "both",
    bfq_enable_attention_token_weight: bool = False,
    bfq_importance_cache_path: str = None,
    bfq_disable_importance_cache: bool = False,
    bfq_importance_only: bool = False,
    bfq_policy_override_path: str = None,
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

    use_attention_token_weight = bool(bfq_enable_attention_token_weight)
    selected_run_mbq = run_mbq_fg if use_attention_token_weight else run_mbq_base
    selected_apply_mbq = apply_mbq_fg if use_attention_token_weight else apply_mbq_base

    if run_bfq_process and not scale_exist:
        model.to_cuda()
        timing_payload = {
            "method": "ours",
            "analysis_label": "quantization_effect_policy",
            "search_label": "cwe_scale_search",
            "analysis_seconds": 0.0,
            "search_seconds": 0.0,
        }
        total_start = time.perf_counter()
        if bfq_policy_override_path:
            with open(bfq_policy_override_path, "r", encoding="utf-8") as f:
                bfq_policy = json.load(f)
            print(f"[BFQ] using policy override: {bfq_policy_override_path}")
        else:
            analysis_start = time.perf_counter()
            bfq_policy = build_bfq_policy(
                process_model=model,
                prompt_inputs=prompt_inputs,
                prompt_kwargs=prompt_kwargs,
                w_bit=w_bit,
                a_bit=bfq_probe_a_bit,
                w_group=q_group_size,
                wa_quant=wa_quant,
                policy_n_samples=bfq_policy_n_samples,
                micro_batch_size=bfq_micro_batch_size,
                beta=bfq_beta,
                n_grid_max=bfq_n_grid_max,
                n_grid_target_mean=bfq_n_grid_target_mean,
                budget_conflict_alpha=bfq_budget_conflict_alpha,
                budget_temperature=bfq_budget_temperature,
                aggressiveness_temperature=bfq_aggressiveness_temperature,
                ratio_upper_min=bfq_ratio_upper_min,
                ratio_upper_max=bfq_ratio_upper_max,
                modality_ratio_min=bfq_modality_ratio_min,
                modality_ratio_max=bfq_modality_ratio_max,
                weight_signed_clip_percentile=bfq_weight_signed_clip_percentile,
                fixed_budget=bfq_fixed_budget,
                fixed_aggressiveness=bfq_fixed_aggressiveness,
                upper_tail_budget=bfq_upper_tail_budget,
                budget_base_grid=bfq_budget_base_grid,
                budget_bonus_low_percentile=bfq_budget_bonus_low_percentile,
                budget_bonus_high_percentile=bfq_budget_bonus_high_percentile,
                budget_bonus_gamma=bfq_budget_bonus_gamma,
                budget_bonus_threshold_mode=bfq_budget_bonus_threshold_mode,
                budget_bonus_threshold_alpha=bfq_budget_bonus_threshold_alpha,
                lower_tail_penalty=bfq_lower_tail_penalty,
                budget_penalty_low_percentile=bfq_budget_penalty_low_percentile,
                budget_penalty_high_percentile=bfq_budget_penalty_high_percentile,
                budget_penalty_gamma=bfq_budget_penalty_gamma,
                conflict_mode=bfq_conflict_mode,
                budget_target_mode=bfq_budget_target_mode,
                importance_cache_path=bfq_importance_cache_path,
                disable_importance_cache=bfq_disable_importance_cache,
            )
            timing_payload["analysis_seconds"] += time.perf_counter() - analysis_start
        try:
            model.to_cpu()
        except NotImplementedError:
            pass

        if bfq_importance_only:
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
            bfq_policy=bfq_policy,
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
        mbq_results["bfq_policy"] = bfq_policy
        mbq_results["bfq_enable_attention_token_weight"] = use_attention_token_weight

        dirpath = os.path.dirname(scale_path)
        os.makedirs(dirpath, exist_ok=True)
        torch.save(mbq_results, scale_path)
        _save_bfq_policy_artifacts(scale_path, bfq_policy)
        print("BFQ results saved at", scale_path)

    if pseudo_quant:
        mbq_results = torch.load(scale_path, map_location="cpu")
        selected_apply_mbq(model.model, mbq_results)
        if not wa_quant:
            pseudo_quantize_model_weight(model.model, w_bit=w_bit, q_config=q_config)
        else:
            pseudo_quantize_model_weight_act(model.model, w_bit=w_bit, a_bit=a_bit)

    model.to_cuda()
    return model
