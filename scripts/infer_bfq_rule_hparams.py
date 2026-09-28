#!/usr/bin/env python
import argparse
import shlex
import csv
import json
from collections import defaultdict
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Infer simple BFQ allocation-rule hyperparameters from calibration quantization-effect summary.")
    p.add_argument("--summary_csv", required=True)
    p.add_argument("--quant_mode", required=True, choices=["w3a16", "w4a8"])
    p.add_argument("--format", default="env", choices=["env", "json"])
    p.add_argument("--rule_version", choices=["coco_v3"], default="coco_v3")
    return p.parse_args()


def _topk_share(vals, k):
    vals = sorted((float(v) for v in vals if float(v) > 0), reverse=True)
    total = sum(vals)
    if total <= 0:
        return 0.0
    return sum(vals[:k]) / total


def _layers_for_mass(vals, target_mass=0.9):
    vals = sorted((float(v) for v in vals if float(v) > 0), reverse=True)
    total = sum(vals)
    if total <= 0:
        return 0
    acc = 0.0
    for idx, v in enumerate(vals, start=1):
        acc += v
        if acc / total >= target_mass:
            return idx
    return len(vals)


def _safe_div(num, den):
    return float(num) / float(den) if float(den) > 0 else 0.0


def _top_fraction_share(vals, fraction):
    vals = sorted((float(v) for v in vals if float(v) > 0), reverse=True)
    total = sum(vals)
    if total <= 0 or len(vals) == 0:
        return 0.0
    k = max(1, int(__import__("math").ceil(float(fraction) * len(vals))))
    return sum(vals[:k]) / total


def _clip_unit(x, lo, hi):
    if hi <= lo:
        return 0.0
    y = (float(x) - float(lo)) / (float(hi) - float(lo))
    return max(0.0, min(1.0, y))


def _round_step(x, step):
    return round(float(x) / float(step)) * float(step)


def main():
    args = parse_args()
    rows = list(csv.DictReader(Path(args.summary_csv).open()))
    if not rows:
        raise RuntimeError(f"empty summary: {args.summary_csv}")

    baseline_loss = None
    harmful_by_comp = defaultdict(dict)
    total_by_comp = defaultdict(float)
    layer_total = defaultdict(float)

    for row in rows:
        if row["variant"] == "baseline_fp":
            baseline_loss = float(row["baseline_loss"])
            continue
        comp = row["component"]
        if not comp:
            continue
        lid = int(row["layer_id"])
        harmful = float(row["harmful_score"])
        harmful_by_comp[comp][lid] = harmful
        total_by_comp[comp] += harmful
        layer_total[lid] += harmful

    if baseline_loss is None:
        raise RuntimeError(f"baseline_fp row missing in {args.summary_csv}")

    stats = {
        "baseline_loss": baseline_loss,
        "total_harmful_weight": total_by_comp.get("weight", 0.0),
        "total_harmful_vision": total_by_comp.get("vision", 0.0),
        "total_harmful_text": total_by_comp.get("text", 0.0),
    }
    total_harmful_all = sum(total_by_comp.values())
    stats["total_harmful_all"] = total_harmful_all

    beneficial_by_comp = defaultdict(dict)
    total_beneficial_by_comp = defaultdict(float)
    layer_beneficial_total = defaultdict(float)
    all_delta_by_layer = defaultdict(list)
    for row in rows:
        if row["variant"] == "baseline_fp":
            continue
        comp = row["component"]
        if not comp:
            continue
        lid = int(row["layer_id"])
        beneficial = float(row["beneficial_score"])
        beneficial_by_comp[comp][lid] = beneficial
        total_beneficial_by_comp[comp] += beneficial
        layer_beneficial_total[lid] += beneficial
        all_delta_by_layer[lid].append(float(row["delta_loss"]))

    total_beneficial_all = sum(total_beneficial_by_comp.values())
    stats["total_beneficial_all"] = total_beneficial_all

    conflict_by_layer = {}
    for lid, deltas in all_delta_by_layer.items():
        if len(deltas) <= 1:
            conflict_by_layer[lid] = 0.0
            continue
        mean_delta = sum(deltas) / len(deltas)
        conflict_by_layer[lid] = (sum((d - mean_delta) ** 2 for d in deltas) / len(deltas)) ** 0.5
    mean_conflict = _safe_div(sum(conflict_by_layer.values()), len(conflict_by_layer))
    stats["mean_conflict"] = mean_conflict

    if args.quant_mode == "w3a16":
        total_harmful_mode = total_by_comp.get("weight", 0.0)
        total_beneficial_mode = total_beneficial_by_comp.get("weight", 0.0)
        harmful_ratio = (total_harmful_mode / baseline_loss) if baseline_loss > 0 else 0.0
        beneficial_ratio = _safe_div(total_beneficial_mode, total_harmful_mode + total_beneficial_mode)
        stats["total_harmful_mode"] = total_harmful_mode
        stats["total_beneficial_mode"] = total_beneficial_mode
        stats["harmful_ratio"] = harmful_ratio
        stats["beneficial_ratio"] = beneficial_ratio

        vals = list(harmful_by_comp.get("weight", {}).values())
        top1_share = _topk_share(vals, 1)
        top3_share = _topk_share(vals, 3)
        top10_share = _top_fraction_share(vals, 0.1)
        layers90 = _layers_for_mass(vals, 0.9)
        support90_frac = _safe_div(layers90, len(vals))

        result = {
            "top1_share_weight": top1_share,
            "top3_share_weight": top3_share,
            "top10_share_weight": top10_share,
            "layers_for_90_weight": layers90,
            "support90_frac_weight": support90_frac,
        }
    else:
        total_harmful_mode = total_harmful_all
        total_beneficial_mode = total_beneficial_all
        harmful_ratio = (total_harmful_mode / baseline_loss) if baseline_loss > 0 else 0.0
        beneficial_ratio = _safe_div(total_beneficial_mode, total_harmful_mode + total_beneficial_mode)
        stats["total_harmful_mode"] = total_harmful_mode
        stats["total_beneficial_mode"] = total_beneficial_mode
        stats["harmful_ratio"] = harmful_ratio
        stats["beneficial_ratio"] = beneficial_ratio

        vals = list(layer_total.values())
        top1_share = _topk_share(vals, 1)
        top3_share = _topk_share(vals, 3)
        top10_share = _top_fraction_share(vals, 0.1)
        layers90 = _layers_for_mass(vals, 0.9)
        support90_frac = _safe_div(layers90, len(vals))
        weight_share = (total_by_comp.get("weight", 0.0) / total_harmful_all) if total_harmful_all > 0 else 0.0
        vision_share = (total_by_comp.get("vision", 0.0) / total_harmful_all) if total_harmful_all > 0 else 0.0
        text_share = (total_by_comp.get("text", 0.0) / total_harmful_all) if total_harmful_all > 0 else 0.0

        activation_share = vision_share + text_share
        result = {
            "top1_share_total": top1_share,
            "top3_share_total": top3_share,
            "top10_share_total": top10_share,
            "layers_for_90_total": layers90,
            "support90_frac_total": support90_frac,
            "weight_share": weight_share,
            "vision_share": vision_share,
            "text_share": text_share,
            "activation_share": activation_share,
        }
    result.update(stats)
    from bfq_auto_rule import predict
    result.update(predict(result, args.quant_mode))

    if args.format == "json":
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        for key, value in result.items():
            print(f"{key}={shlex.quote(str(value))}")


if __name__ == "__main__":
    main()
