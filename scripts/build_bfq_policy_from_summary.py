#!/usr/bin/env python
import argparse
import csv
import json
from pathlib import Path

import torch

from qmllm.methods.glmi.policy import (
    _allocate_budget,
    _allocate_budget_upper_tail,
    _allocate_budget_upper_tail_with_low_penalty,
    _clip_signed_scores,
    _normalize_list,
    _signed_scores_to_unit,
)


def parse_args():
    p = argparse.ArgumentParser(description="Build BFQ policy JSON from precomputed quantization-effect summary.")
    p.add_argument("--summary_csv", required=True)
    p.add_argument("--quant_mode", required=True, choices=["w3a16", "w4a8"])
    p.add_argument("--output_json", required=True)
    p.add_argument("--glmi_beta", type=float, required=True)
    p.add_argument("--glmi_n_grid_max", type=int, required=True)
    p.add_argument("--glmi_n_grid_target_mean", type=float, required=True)
    p.add_argument("--glmi_budget_conflict_alpha", type=float, required=True)
    p.add_argument("--glmi_budget_temperature", type=float, required=True)
    p.add_argument("--glmi_aggressiveness_temperature", type=float, required=True)
    p.add_argument("--glmi_ratio_upper_min", type=float, required=True)
    p.add_argument("--glmi_ratio_upper_max", type=float, required=True)
    p.add_argument("--glmi_modality_ratio_min", type=float, required=True)
    p.add_argument("--glmi_modality_ratio_max", type=float, required=True)
    p.add_argument("--glmi_weight_signed_clip_percentile", type=float, required=True)
    p.add_argument("--glmi_budget_base_grid", type=int, required=True)
    p.add_argument("--glmi_budget_bonus_low_percentile", type=float, required=True)
    p.add_argument("--glmi_budget_bonus_high_percentile", type=float, required=True)
    p.add_argument("--glmi_budget_bonus_gamma", type=float, required=True)
    p.add_argument("--glmi_budget_bonus_threshold_mode", required=True)
    p.add_argument("--glmi_budget_bonus_threshold_alpha", type=float, required=True)
    p.add_argument("--glmi_budget_penalty_low_percentile", type=float, required=True)
    p.add_argument("--glmi_budget_penalty_high_percentile", type=float, required=True)
    p.add_argument("--glmi_budget_penalty_gamma", type=float, required=True)
    p.add_argument("--glmi_conflict_mode", required=True)
    p.add_argument("--glmi_budget_target_mode", required=True)
    p.add_argument("--glmi_fixed_budget", action="store_true")
    p.add_argument("--glmi_fixed_aggressiveness", action="store_true")
    p.add_argument("--glmi_upper_tail_budget", action="store_true")
    p.add_argument("--glmi_lower_tail_penalty", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    rows = list(csv.DictReader(Path(args.summary_csv).open()))
    if not rows:
        raise RuntimeError(f"empty summary: {args.summary_csv}")

    baseline_loss = None
    records = {}
    max_layer_id = -1
    for row in rows:
        if row["variant"] == "baseline_fp":
            baseline_loss = float(row["baseline_loss"])
            continue
        comp = row["component"]
        if not comp:
            continue
        layer_id = int(row["layer_id"])
        max_layer_id = max(max_layer_id, layer_id)
        delta_loss = float(row["delta_loss"])
        harmful = float(row["harmful_score"])
        beneficial = float(row["beneficial_score"])
        records[(layer_id, comp)] = {
            "quant_loss": float(baseline_loss + delta_loss) if baseline_loss is not None else None,
            "delta_loss": delta_loss,
            "harmful_score": harmful,
            "beneficial_score": beneficial,
            "risk_score": max(harmful - args.glmi_beta * beneficial, 1e-6),
        }

    if baseline_loss is None:
        raise RuntimeError(f"baseline_fp row missing in {args.summary_csv}")
    if max_layer_id < 0:
        raise RuntimeError(f"no component rows found in {args.summary_csv}")

    num_layers = max_layer_id + 1
    wa_quant = args.quant_mode == "w4a8"
    weight_only_policy = args.quant_mode == "w3a16"

    for layer_id in range(num_layers):
        if (layer_id, "weight") not in records:
            raise RuntimeError(f"missing weight row for layer {layer_id} in {args.summary_csv}")
        if weight_only_policy:
            for comp in ("vision", "text"):
                if (layer_id, comp) not in records:
                    records[(layer_id, comp)] = {
                        "quant_loss": baseline_loss,
                        "delta_loss": 0.0,
                        "harmful_score": 0.0,
                        "beneficial_score": 0.0,
                        "risk_score": 1.0,
                    }
        else:
            for comp in ("vision", "text"):
                if (layer_id, comp) not in records:
                    raise RuntimeError(f"missing {comp} row for layer {layer_id} in {args.summary_csv}")

    weight_scores = []
    activation_scores = []
    signed_weight_scores = []
    signed_activation_scores = []
    weight_budget_scores = []
    activation_budget_scores = []
    conflict_scores = []
    weight_conflict_scores = []
    activation_conflict_scores = []

    for layer_id in range(num_layers):
        delta_w = records[(layer_id, "weight")]["delta_loss"]
        delta_v = records[(layer_id, "vision")]["delta_loss"]
        delta_t = records[(layer_id, "text")]["delta_loss"]

        weight_score = records[(layer_id, "weight")]["risk_score"]
        activation_score = (
            records[(layer_id, "vision")]["risk_score"] + records[(layer_id, "text")]["risk_score"]
        )
        signed_weight_score = (
            records[(layer_id, "weight")]["harmful_score"]
            - args.glmi_beta * records[(layer_id, "weight")]["beneficial_score"]
        )

        if weight_only_policy:
            signed_activation_score = 0.0
            activation_score = 0.0
            conflict_score = 0.0
            vt_conflict_score = 0.0
        else:
            signed_activation_score = (
                records[(layer_id, "vision")]["harmful_score"]
                - args.glmi_beta * records[(layer_id, "vision")]["beneficial_score"]
                + records[(layer_id, "text")]["harmful_score"]
                - args.glmi_beta * records[(layer_id, "text")]["beneficial_score"]
            )
            conflict_score = float(
                torch.tensor([delta_w, delta_v, delta_t], dtype=torch.float32).std(unbiased=False).item()
            )
            vt_conflict_score = float(
                torch.tensor([delta_v, delta_t], dtype=torch.float32).std(unbiased=False).item()
            )

        if args.glmi_conflict_mode == "shared":
            weight_conflict_score = conflict_score
            activation_conflict_score = conflict_score
        elif args.glmi_conflict_mode == "split_vt":
            weight_conflict_score = 0.0
            activation_conflict_score = vt_conflict_score
        elif args.glmi_conflict_mode == "none":
            weight_conflict_score = 0.0
            activation_conflict_score = 0.0
        else:
            raise ValueError(f"Unsupported GLMI conflict_mode: {args.glmi_conflict_mode}")

        weight_scores.append(weight_score)
        activation_scores.append(activation_score)
        signed_weight_scores.append(signed_weight_score)
        signed_activation_scores.append(signed_activation_score)
        weight_budget_scores.append(signed_weight_score + args.glmi_budget_conflict_alpha * weight_conflict_score)
        activation_budget_scores.append(
            signed_activation_score + args.glmi_budget_conflict_alpha * activation_conflict_score
        )
        conflict_scores.append(conflict_score)
        weight_conflict_scores.append(weight_conflict_score)
        activation_conflict_scores.append(activation_conflict_score)

    clipped_signed_weight_scores = _clip_signed_scores(
        signed_weight_scores, percentile=args.glmi_weight_signed_clip_percentile
    )
    norm_weight_scores = _normalize_list(weight_scores)
    norm_activation_scores = _normalize_list(activation_scores)
    norm_signed_weight_scores = _signed_scores_to_unit(
        clipped_signed_weight_scores, temperature=args.glmi_aggressiveness_temperature
    )
    norm_signed_activation_scores = _signed_scores_to_unit(
        signed_activation_scores, temperature=args.glmi_aggressiveness_temperature
    )
    clipped_weight_budget_scores = [
        clipped_signed_weight_scores[i] + args.glmi_budget_conflict_alpha * weight_conflict_scores[i]
        for i in range(num_layers)
    ]
    norm_weight_budget_scores = _normalize_list(clipped_weight_budget_scores)
    norm_activation_budget_scores = _normalize_list(activation_budget_scores)
    norm_conflict_scores = _normalize_list(conflict_scores)

    floor_grid = int(args.glmi_budget_base_grid)
    fixed_grid = int(round(args.glmi_n_grid_target_mean))
    fixed_grid = max(floor_grid, min(args.glmi_n_grid_max, fixed_grid))

    if args.glmi_fixed_budget:
        weight_budgets = [fixed_grid for _ in range(num_layers)]
        activation_budgets = [fixed_grid for _ in range(num_layers)]
    else:
        if args.glmi_upper_tail_budget and args.glmi_lower_tail_penalty:
            adaptive_weight_budgets = _allocate_budget_upper_tail_with_low_penalty(
                clipped_weight_budget_scores,
                floor_grid=floor_grid,
                max_grid=args.glmi_n_grid_max,
                target_mean=args.glmi_n_grid_target_mean,
                base_grid=args.glmi_budget_base_grid,
                bonus_low_percentile=args.glmi_budget_bonus_low_percentile,
                bonus_high_percentile=args.glmi_budget_bonus_high_percentile,
                bonus_gamma=args.glmi_budget_bonus_gamma,
                bonus_threshold_mode=args.glmi_budget_bonus_threshold_mode,
                bonus_threshold_alpha=args.glmi_budget_bonus_threshold_alpha,
                penalty_low_percentile=args.glmi_budget_penalty_low_percentile,
                penalty_high_percentile=args.glmi_budget_penalty_high_percentile,
                penalty_gamma=args.glmi_budget_penalty_gamma,
            )
            adaptive_activation_budgets = _allocate_budget_upper_tail_with_low_penalty(
                activation_budget_scores,
                floor_grid=floor_grid,
                max_grid=args.glmi_n_grid_max,
                target_mean=args.glmi_n_grid_target_mean,
                base_grid=args.glmi_budget_base_grid,
                bonus_low_percentile=args.glmi_budget_bonus_low_percentile,
                bonus_high_percentile=args.glmi_budget_bonus_high_percentile,
                bonus_gamma=args.glmi_budget_bonus_gamma,
                bonus_threshold_mode=args.glmi_budget_bonus_threshold_mode,
                bonus_threshold_alpha=args.glmi_budget_bonus_threshold_alpha,
                penalty_low_percentile=args.glmi_budget_penalty_low_percentile,
                penalty_high_percentile=args.glmi_budget_penalty_high_percentile,
                penalty_gamma=args.glmi_budget_penalty_gamma,
            )
        elif args.glmi_upper_tail_budget:
            adaptive_weight_budgets = _allocate_budget_upper_tail(
                clipped_weight_budget_scores,
                floor_grid=floor_grid,
                max_grid=args.glmi_n_grid_max,
                target_mean=args.glmi_n_grid_target_mean,
                base_grid=args.glmi_budget_base_grid,
                low_percentile=args.glmi_budget_bonus_low_percentile,
                high_percentile=args.glmi_budget_bonus_high_percentile,
                gamma=args.glmi_budget_bonus_gamma,
                threshold_mode=args.glmi_budget_bonus_threshold_mode,
                threshold_alpha=args.glmi_budget_bonus_threshold_alpha,
            )
            adaptive_activation_budgets = _allocate_budget_upper_tail(
                activation_budget_scores,
                floor_grid=floor_grid,
                max_grid=args.glmi_n_grid_max,
                target_mean=args.glmi_n_grid_target_mean,
                base_grid=args.glmi_budget_base_grid,
                low_percentile=args.glmi_budget_bonus_low_percentile,
                high_percentile=args.glmi_budget_bonus_high_percentile,
                gamma=args.glmi_budget_bonus_gamma,
                threshold_mode=args.glmi_budget_bonus_threshold_mode,
                threshold_alpha=args.glmi_budget_bonus_threshold_alpha,
            )
        else:
            adaptive_weight_budgets = _allocate_budget(
                clipped_weight_budget_scores,
                floor_grid=floor_grid,
                max_grid=args.glmi_n_grid_max,
                target_mean=args.glmi_n_grid_target_mean,
                temperature=args.glmi_budget_temperature,
            )
            adaptive_activation_budgets = _allocate_budget(
                activation_budget_scores,
                floor_grid=floor_grid,
                max_grid=args.glmi_n_grid_max,
                target_mean=args.glmi_n_grid_target_mean,
                temperature=args.glmi_budget_temperature,
            )

        if args.glmi_budget_target_mode == "both":
            weight_budgets = adaptive_weight_budgets
            activation_budgets = adaptive_activation_budgets
        elif args.glmi_budget_target_mode == "weight_only":
            weight_budgets = adaptive_weight_budgets
            activation_budgets = [fixed_grid for _ in range(num_layers)]
        elif args.glmi_budget_target_mode == "act_only":
            weight_budgets = [fixed_grid for _ in range(num_layers)]
            activation_budgets = adaptive_activation_budgets
        else:
            raise ValueError(f"Unsupported GLMI budget_target_mode: {args.glmi_budget_target_mode}")

    positive_weight_total = float(sum(max(v, 0.0) for v in clipped_weight_budget_scores))
    positive_activation_total = float(sum(max(v, 0.0) for v in activation_budget_scores))
    if wa_quant:
        denom = positive_weight_total + positive_activation_total
        if denom <= 1e-12:
            search_weight_share = 0.5
            search_activation_share = 0.5
        else:
            search_weight_share = positive_weight_total / denom
            search_activation_share = positive_activation_total / denom
    else:
        search_weight_share = 1.0
        search_activation_share = 0.0

    policy_layers = {}
    for layer_id in range(num_layers):
        norm_weight = float(norm_weight_scores[layer_id])
        norm_activation = float(norm_activation_scores[layer_id])
        norm_signed_weight = float(norm_signed_weight_scores[layer_id])
        norm_signed_activation = float(norm_signed_activation_scores[layer_id])
        norm_weight_budget = float(norm_weight_budget_scores[layer_id])
        norm_activation_budget = float(norm_activation_budget_scores[layer_id])
        norm_conflict = float(norm_conflict_scores[layer_id])

        weight_grid = int(weight_budgets[layer_id])
        activation_grid = int(activation_budgets[layer_id])

        if args.glmi_fixed_aggressiveness:
            ratio_max_weight = float(args.glmi_ratio_upper_max)
            ratio_max_activation = float(args.glmi_ratio_upper_max)
        else:
            ratio_max_weight = float(
                args.glmi_ratio_upper_max
                - (args.glmi_ratio_upper_max - args.glmi_ratio_upper_min) * norm_signed_weight
            )
            ratio_max_activation = float(
                args.glmi_ratio_upper_max
                - (args.glmi_ratio_upper_max - args.glmi_ratio_upper_min) * norm_signed_activation
            )

        if wa_quant:
            search_budget_score = (
                search_weight_share * clipped_weight_budget_scores[layer_id]
                + search_activation_share * activation_budget_scores[layer_id]
            )
            search_budget_norm_score = (
                search_weight_share * norm_weight_budget
                + search_activation_share * norm_activation_budget
            )
            search_signed_score = (
                search_weight_share * clipped_signed_weight_scores[layer_id]
                + search_activation_share * signed_activation_scores[layer_id]
            )
            search_signed_norm_score = (
                search_weight_share * norm_signed_weight
                + search_activation_share * norm_signed_activation
            )
            search_grid = int(round(search_weight_share * weight_grid + search_activation_share * activation_grid))
            search_grid = max(floor_grid, min(args.glmi_n_grid_max, search_grid))
            search_ratio_max = float(
                search_weight_share * ratio_max_weight + search_activation_share * ratio_max_activation
            )
        else:
            search_budget_score = clipped_weight_budget_scores[layer_id]
            search_budget_norm_score = norm_weight_budget
            search_signed_score = clipped_signed_weight_scores[layer_id]
            search_signed_norm_score = norm_signed_weight
            search_grid = int(weight_grid)
            search_ratio_max = float(ratio_max_weight)

        if weight_only_policy:
            reweight_ratio = 1.0
        else:
            risk_vis = records[(layer_id, "vision")]["risk_score"]
            risk_text = records[(layer_id, "text")]["risk_score"]
            reweight_ratio = float(risk_vis / max(risk_text, 1e-6))
            reweight_ratio = max(args.glmi_modality_ratio_min, min(args.glmi_modality_ratio_max, reweight_ratio))

        policy_layers[str(layer_id)] = {
            "weight_policy": {
                "layer_score": weight_scores[layer_id],
                "norm_score": norm_weight,
                "signed_score": signed_weight_scores[layer_id],
                "signed_score_raw": signed_weight_scores[layer_id],
                "signed_score_clipped": clipped_signed_weight_scores[layer_id],
                "signed_norm_score": norm_signed_weight,
                "budget_score": clipped_weight_budget_scores[layer_id],
                "budget_norm_score": norm_weight_budget,
                "n_grid": int(weight_grid),
                "ratio_max": ratio_max_weight,
            },
            "activation_policy": {
                "layer_score": activation_scores[layer_id],
                "norm_score": norm_activation,
                "signed_score": signed_activation_scores[layer_id],
                "signed_norm_score": norm_signed_activation,
                "budget_score": activation_budget_scores[layer_id],
                "budget_norm_score": norm_activation_budget,
                "n_grid": int(activation_grid),
                "ratio_max": ratio_max_activation,
            },
            "conflict_score": conflict_scores[layer_id],
            "weight_conflict_score": weight_conflict_scores[layer_id],
            "activation_conflict_score": activation_conflict_scores[layer_id],
            "conflict_norm_score": norm_conflict,
            "reweight_ratio": reweight_ratio,
            "search_policy": {
                "weight_share": float(search_weight_share),
                "activation_share": float(search_activation_share),
                "signed_score": float(search_signed_score),
                "signed_norm_score": float(search_signed_norm_score),
                "budget_score": float(search_budget_score),
                "budget_norm_score": float(search_budget_norm_score),
                "n_grid": int(search_grid),
                "ratio_max": float(search_ratio_max),
            },
            "components": {
                comp: dict(records[(layer_id, comp)])
                for comp in ("weight", "vision", "text")
            },
        }

    payload = {
        "baseline_loss": baseline_loss,
        "beta": float(args.glmi_beta),
        "n_grid_max": int(args.glmi_n_grid_max),
        "n_grid_target_mean": float(args.glmi_n_grid_target_mean),
        "budget_conflict_alpha": float(args.glmi_budget_conflict_alpha),
        "budget_temperature": float(args.glmi_budget_temperature),
        "aggressiveness_temperature": float(args.glmi_aggressiveness_temperature),
        "search_weight_share": float(search_weight_share),
        "search_activation_share": float(search_activation_share),
        "ratio_upper_min": float(args.glmi_ratio_upper_min),
        "ratio_upper_max": float(args.glmi_ratio_upper_max),
        "modality_ratio_min": float(args.glmi_modality_ratio_min),
        "modality_ratio_max": float(args.glmi_modality_ratio_max),
        "weight_signed_clip_percentile": float(args.glmi_weight_signed_clip_percentile),
        "fixed_budget": bool(args.glmi_fixed_budget),
        "fixed_aggressiveness": bool(args.glmi_fixed_aggressiveness),
        "upper_tail_budget": bool(args.glmi_upper_tail_budget),
        "budget_base_grid": int(args.glmi_budget_base_grid),
        "budget_bonus_low_percentile": float(args.glmi_budget_bonus_low_percentile),
        "budget_bonus_high_percentile": float(args.glmi_budget_bonus_high_percentile),
        "budget_bonus_gamma": float(args.glmi_budget_bonus_gamma),
        "budget_bonus_threshold_mode": str(args.glmi_budget_bonus_threshold_mode),
        "budget_bonus_threshold_alpha": float(args.glmi_budget_bonus_threshold_alpha),
        "lower_tail_penalty": bool(args.glmi_lower_tail_penalty),
        "budget_penalty_low_percentile": float(args.glmi_budget_penalty_low_percentile),
        "budget_penalty_high_percentile": float(args.glmi_budget_penalty_high_percentile),
        "budget_penalty_gamma": float(args.glmi_budget_penalty_gamma),
        "conflict_mode": str(args.glmi_conflict_mode),
        "budget_target_mode": str(args.glmi_budget_target_mode),
        "layers": policy_layers,
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
