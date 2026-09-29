import copy
import gc
import json
import os
import weakref
from typing import Dict, Optional

import torch
import torch.nn.functional as F
import tqdm

from qmllm.methods.mbq.quantize.pre_quant import get_blocks, get_named_linears
from qmllm.methods.mbq.quantize.quantizer import get_module_by_name_suffix
from qmllm.quantization.qlinear import WALinear
from qmllm.quantization.quant_funcs import pseudo_quantize_tensor


def _infer_batch_size(prompt_inputs: Dict, prompt_kwargs: Dict) -> int:
    for key in ("inputs_embeds", "input_ids"):
        val = prompt_inputs.get(key)
        if torch.is_tensor(val) and val.dim() >= 1:
            return int(val.shape[0])
    for key in ("labels", "attention_mask", "vision_mask", "caption_mask"):
        val = prompt_kwargs.get(key)
        if torch.is_tensor(val):
            if val.dim() >= 1:
                return int(val.shape[0] if val.dim() == 1 else val.shape[0])
    raise ValueError("Cannot infer batch size from prompt inputs/kwargs.")


def _slice_tensor(v, start: int, end: int, batch_size: int):
    if not torch.is_tensor(v):
        return v
    if v.dim() >= 1 and v.shape[0] == batch_size:
        return v[start:end]
    if v.dim() >= 2 and v.shape[1] == batch_size:
        return v[:, start:end]
    return v


def _slice_value(v, start: int, end: int, batch_size: int):
    if torch.is_tensor(v):
        return _slice_tensor(v, start, end, batch_size)
    if isinstance(v, list) and len(v) == batch_size:
        return v[start:end]
    if isinstance(v, tuple) and len(v) == batch_size:
        return v[start:end]
    return v


def _slice_batch(prompt_inputs: Dict, prompt_kwargs: Dict, start: int, end: int, batch_size: int):
    cur_inputs = {k: _slice_value(v, start, end, batch_size) for k, v in prompt_inputs.items()}
    cur_kwargs = {k: _slice_value(v, start, end, batch_size) for k, v in prompt_kwargs.items()}
    return cur_inputs, cur_kwargs


def _fit_mask_2d(mask: Optional[torch.Tensor], batch_size: int, seq_len: int, device: torch.device) -> Optional[torch.Tensor]:
    if not torch.is_tensor(mask):
        return None
    mask = mask.to(device=device, dtype=torch.bool)
    while mask.dim() > 2:
        if mask.shape[-1] == 1:
            mask = mask.squeeze(-1)
        else:
            mask = (mask != 0).any(dim=-1)
    if mask.dim() == 1:
        mask = mask.unsqueeze(0)
    if mask.dim() == 2 and mask.shape[0] == seq_len and mask.shape[1] != seq_len:
        mask = (mask != 0).any(dim=-1, keepdim=True).transpose(0, 1)
    if mask.dim() != 2:
        return None
    if mask.shape[0] < batch_size:
        if mask.shape[0] == 1:
            mask = mask.expand(batch_size, -1)
        else:
            pad = torch.zeros((batch_size - mask.shape[0], mask.shape[1]), dtype=torch.bool, device=device)
            mask = torch.cat([mask, pad], dim=0)
    elif mask.shape[0] > batch_size:
        mask = mask[:batch_size]
    if mask.shape[1] < seq_len:
        pad = torch.zeros((mask.shape[0], seq_len - mask.shape[1]), dtype=torch.bool, device=device)
        mask = torch.cat([mask, pad], dim=1)
    elif mask.shape[1] > seq_len:
        mask = mask[:, :seq_len]
    return mask


def _set_runtime_masks(owner_model, vision_mask: Optional[torch.Tensor], attention_mask: Optional[torch.Tensor], ref: torch.Tensor) -> None:
    device = ref.device
    batch_size, seq_len = int(ref.shape[0]), int(ref.shape[1])
    owner_model._bfq_runtime_vision_mask = _fit_mask_2d(vision_mask, batch_size, seq_len, device)
    owner_model._bfq_runtime_attention_mask = _fit_mask_2d(attention_mask, batch_size, seq_len, device)
    if owner_model._bfq_runtime_attention_mask is None:
        owner_model._bfq_runtime_attention_mask = torch.ones((batch_size, seq_len), dtype=torch.bool, device=device)
    if owner_model._bfq_runtime_vision_mask is None:
        owner_model._bfq_runtime_vision_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=device)


class ScopedActOnlyWALinear(WALinear):
    def __init__(self, *args, owner_model=None, token_scope: str = "all", **kwargs):
        super().__init__(*args, **kwargs)
        self.token_scope = token_scope
        object.__setattr__(self, "_owner_model_ref", weakref.ref(owner_model) if owner_model is not None else None)

    @torch.no_grad()
    def forward(self, x):
        owner_model_ref = getattr(self, "_owner_model_ref", None)
        owner_model = owner_model_ref() if owner_model_ref is not None else None
        if self.token_scope == "all" or owner_model is None or x.dim() < 3:
            return super().forward(x)

        vision_mask = getattr(owner_model, "_bfq_runtime_vision_mask", None)
        attention_mask = getattr(owner_model, "_bfq_runtime_attention_mask", None)
        vision_mask = _fit_mask_2d(vision_mask, x.shape[0], x.shape[1], x.device)
        attention_mask = _fit_mask_2d(attention_mask, x.shape[0], x.shape[1], x.device)

        if attention_mask is None:
            attention_mask = torch.ones((x.shape[0], x.shape[1]), dtype=torch.bool, device=x.device)
        if vision_mask is None:
            vision_mask = torch.zeros((x.shape[0], x.shape[1]), dtype=torch.bool, device=x.device)

        if self.token_scope == "vision":
            selected = attention_mask & vision_mask
        else:
            selected = attention_mask & (~vision_mask)

        if not torch.any(selected):
            return F.linear(x, self.weight, self.bias)

        q_x = x.clone()
        q_x[selected] = self.act_quant(q_x[selected])
        y = F.linear(q_x, self.weight, self.bias)
        return self.output_quant(y)


def _replace_layer_with_act_quant(model, layer_id: int, w_bit: int, a_bit: int, token_scope: str) -> None:
    layer = get_blocks(model)[layer_id]
    named_linears = get_named_linears(layer)
    for name, mod in named_linears.items():
        if isinstance(mod, WALinear):
            continue
        new_linear = ScopedActOnlyWALinear(
            mod.in_features,
            mod.out_features,
            mod.bias is not None,
            act_quant="per_token",
            a_bit=a_bit,
            w_bit=w_bit,
            dev=mod.weight.device,
            owner_model=model,
            token_scope=token_scope,
        )
        new_linear.weight = mod.weight
        new_linear.weight_quant_name = "prequantized"
        if mod.bias is not None:
            new_linear.bias = mod.bias
        parent_name = ".".join(name.split(".")[:-1])
        parent = get_module_by_name_suffix(layer, parent_name)
        setattr(parent, name.split(".")[-1], new_linear)


def _quantize_layer_weights(model, layer_id: int, w_bit: int, w_group: int) -> None:
    layer = get_blocks(model)[layer_id]
    named_linears = get_named_linears(layer)
    q_config = {"zero_point": True, "q_group_size": w_group}
    for _, mod in named_linears.items():
        mod.weight.data = pseudo_quantize_tensor(mod.weight.data, n_bits=w_bit, **q_config)


def _snapshot_layer(model, layer_id: int):
    layers = get_blocks(model)
    layer = layers[layer_id]
    return copy.deepcopy(layer).cpu(), next(layer.parameters()).device


def _restore_layer(model, layer_id: int, backup_layer, device: torch.device) -> None:
    get_blocks(model)[layer_id] = backup_layer.to(device)
    gc.collect()
    torch.cuda.empty_cache()


def _evaluate_avg_loss(process_model, prompt_inputs: Dict, prompt_kwargs: Dict, micro_batch_size: int) -> Dict[str, float]:
    total_loss = 0.0
    total_targets = 0
    num_batches = 0
    owner_model = process_model.model
    batch_size = _infer_batch_size(prompt_inputs, prompt_kwargs)

    for start in range(0, batch_size, micro_batch_size):
        end = min(start + micro_batch_size, batch_size)
        cur_inputs, cur_kwargs = _slice_batch(prompt_inputs, prompt_kwargs, start, end, batch_size)
        inputs_embeds = cur_inputs["inputs_embeds"]
        labels = cur_kwargs["labels"]
        attention_mask = cur_kwargs.get("attention_mask")
        vision_mask = cur_kwargs.get("vision_mask")
        _set_runtime_masks(owner_model, vision_mask, attention_mask, ref=inputs_embeds)

        outputs = process_model.forward(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
            return_dict=True,
        )

        valid_targets = (labels[..., 1:] != -100).sum().item()
        if valid_targets > 0:
            total_loss += float(outputs.loss.detach().float().item()) * valid_targets
            total_targets += int(valid_targets)
            num_batches += 1

        del cur_inputs, cur_kwargs, outputs, inputs_embeds, labels, attention_mask, vision_mask
        gc.collect()
        torch.cuda.empty_cache()

    if total_targets <= 0:
        raise RuntimeError("No valid supervised targets found while building BFQ policy.")
    return {
        "avg_loss": total_loss / total_targets,
        "num_targets": total_targets,
        "num_batches": num_batches,
    }


def _normalize_list(values):
    if len(values) == 0:
        return []
    vmin = min(values)
    vmax = max(values)
    if abs(vmax - vmin) < 1e-12:
        return [0.5 for _ in values]
    return [(v - vmin) / (vmax - vmin) for v in values]


def _build_importance_cache_metadata(
    prompt_inputs: Dict,
    prompt_kwargs: Dict,
    *,
    w_bit: int,
    a_bit: int,
    w_group: int,
    wa_quant: bool,
    policy_n_samples: int,
    micro_batch_size: int,
    beta: float,
    num_layers: int,
):
    labels = prompt_kwargs.get("labels")
    attention_mask = prompt_kwargs.get("attention_mask")
    vision_mask = prompt_kwargs.get("vision_mask")
    inputs_embeds = prompt_inputs.get("inputs_embeds")
    sample_ids = prompt_kwargs.get("sample_ids")
    per_sample_target_counts = None
    if torch.is_tensor(labels):
        if labels.dim() >= 2:
            per_sample_target_counts = (labels[..., 1:] != -100).sum(dim=-1).tolist()
        elif labels.dim() == 1:
            per_sample_target_counts = [int((labels[1:] != -100).sum().item())]
    return {
        "w_bit": int(w_bit),
        "a_bit": int(a_bit),
        "w_group": int(w_group),
        "wa_quant": bool(wa_quant),
        "policy_n_samples": int(policy_n_samples),
        "micro_batch_size": int(micro_batch_size),
        "beta": float(beta),
        "num_layers": int(num_layers),
        "batch_size": int(_infer_batch_size(prompt_inputs, prompt_kwargs)),
        "inputs_embeds_shape": list(inputs_embeds.shape) if torch.is_tensor(inputs_embeds) else None,
        "labels_shape": list(labels.shape) if torch.is_tensor(labels) else None,
        "attention_mask_shape": list(attention_mask.shape) if torch.is_tensor(attention_mask) else None,
        "attention_mask_sum": int(attention_mask.sum().item()) if torch.is_tensor(attention_mask) else None,
        "vision_mask_shape": list(vision_mask.shape) if torch.is_tensor(vision_mask) else None,
        "vision_mask_sum": int(vision_mask.sum().item()) if torch.is_tensor(vision_mask) else None,
        "policy_sample_ids": list(sample_ids) if isinstance(sample_ids, (list, tuple)) else sample_ids,
        "per_sample_target_counts": [int(v) for v in per_sample_target_counts] if per_sample_target_counts is not None else None,
    }


def _save_importance_cache(
    cache_path: str,
    *,
    metadata: Dict,
    baseline: Dict,
    records: Dict,
) -> None:
    cache_dir = os.path.dirname(cache_path)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    serializable_records = {}
    for (layer_id, component), rec in records.items():
        layer_key = str(int(layer_id))
        if layer_key not in serializable_records:
            serializable_records[layer_key] = {}
        serializable_records[layer_key][component] = {
            key: float(value) if isinstance(value, (int, float)) else value
            for key, value in rec.items()
        }
    payload = {
        "metadata": metadata,
        "baseline": {
            "avg_loss": float(baseline["avg_loss"]),
            "num_targets": int(baseline["num_targets"]),
            "num_batches": int(baseline["num_batches"]),
        },
        "records": serializable_records,
    }
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _load_importance_cache(cache_path: str):
    with open(cache_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    records = {}
    for layer_key, comps in payload.get("records", {}).items():
        layer_id = int(layer_key)
        for component, rec in comps.items():
            records[(layer_id, component)] = dict(rec)
    return payload.get("metadata", {}), payload["baseline"], records


def _signed_scores_to_unit(values, temperature: float = 3.0):
    if len(values) == 0:
        return []
    score_tensor = torch.tensor(values, dtype=torch.float64)
    centered = score_tensor - score_tensor.mean()
    std = score_tensor.std(unbiased=False)
    if torch.isclose(std, torch.tensor(0.0, dtype=std.dtype)):
        z = centered
    else:
        z = centered / std
    unit = torch.sigmoid(z * float(temperature))
    return [float(v) for v in unit.tolist()]


def _clip_signed_scores(values, percentile: float = 100.0):
    if len(values) == 0:
        return []
    percentile = float(percentile)
    if percentile >= 100.0:
        return [float(v) for v in values]
    percentile = max(0.0, min(100.0, percentile))
    tensor = torch.tensor(values, dtype=torch.float64)
    abs_tensor = tensor.abs()
    threshold = torch.quantile(abs_tensor, percentile / 100.0)
    clipped = torch.clamp(tensor, min=-threshold, max=threshold)
    return [float(v) for v in clipped.tolist()]


def _allocate_budget(
    scores,
    floor_grid: int,
    max_grid: int,
    target_mean: float,
    temperature: float = 4.0,
):
    if len(scores) == 0:
        return []

    num_layers = len(scores)
    min_total = int(num_layers * floor_grid)
    max_total = int(num_layers * max_grid)
    target_total = int(round(target_mean * num_layers))
    target_total = max(min_total, min(max_total, target_total))

    budgets = [int(floor_grid) for _ in range(num_layers)]
    capacities = [int(max_grid - floor_grid) for _ in range(num_layers)]
    remaining = int(target_total - min_total)
    if remaining <= 0:
        return budgets

    score_tensor = torch.tensor(scores, dtype=torch.float64)
    if torch.allclose(score_tensor, score_tensor[:1]):
        probs = torch.full((num_layers,), 1.0 / num_layers, dtype=torch.float64)
    else:
        centered = score_tensor - score_tensor.mean()
        std = score_tensor.std(unbiased=False)
        if torch.isclose(std, torch.tensor(0.0, dtype=std.dtype)):
            scaled = centered
        else:
            scaled = centered / std
        probs = torch.softmax(scaled * float(temperature), dim=0)

    active = torch.ones(num_layers, dtype=torch.bool)
    while remaining > 0 and bool(active.any()):
        active_idx = active.nonzero(as_tuple=True)[0]
        active_probs = probs[active_idx]
        active_probs = active_probs / active_probs.sum()
        raw_extra = active_probs * remaining

        saturated = False
        for local_idx, layer_idx in enumerate(active_idx.tolist()):
            cap = capacities[layer_idx]
            if cap <= 0:
                active[layer_idx] = False
                continue
            if raw_extra[local_idx].item() >= cap - 1e-12:
                budgets[layer_idx] += cap
                remaining -= cap
                capacities[layer_idx] = 0
                active[layer_idx] = False
                saturated = True
        if saturated:
            continue

        floor_extra = torch.floor(raw_extra).to(torch.int64)
        floor_sum = int(floor_extra.sum().item())
        if floor_sum > 0:
            for local_idx, add in enumerate(floor_extra.tolist()):
                if add <= 0:
                    continue
                layer_idx = active_idx[local_idx].item()
                add = min(add, capacities[layer_idx], remaining)
                if add <= 0:
                    continue
                budgets[layer_idx] += add
                capacities[layer_idx] -= add
                remaining -= add
                if capacities[layer_idx] <= 0:
                    active[layer_idx] = False

        if remaining <= 0:
            break

        frac_extra = raw_extra - floor_extra.to(raw_extra.dtype)
        order = torch.argsort(frac_extra, descending=True).tolist()
        progressed = False
        for local_idx in order:
            layer_idx = active_idx[local_idx].item()
            if remaining <= 0:
                break
            if capacities[layer_idx] <= 0:
                active[layer_idx] = False
                continue
            budgets[layer_idx] += 1
            capacities[layer_idx] -= 1
            remaining -= 1
            progressed = True
            if capacities[layer_idx] <= 0:
                active[layer_idx] = False

        if not progressed:
            break

    return budgets


def _allocate_budget_upper_tail(
    scores,
    floor_grid: int,
    max_grid: int,
    target_mean: float,
    base_grid: int,
    low_percentile: float = 50.0,
    high_percentile: float = 95.0,
    gamma: float = 2.0,
    threshold_mode: str = "percentile",
    threshold_alpha: float = 1.0,
):
    if len(scores) == 0:
        return []

    num_layers = len(scores)
    min_total = int(num_layers * floor_grid)
    max_total = int(num_layers * max_grid)
    target_total = int(round(target_mean * num_layers))
    target_total = max(min_total, min(max_total, target_total))

    max_base_grid = max(floor_grid, min(max_grid, int(target_total // num_layers)))
    base_grid = max(floor_grid, min(max_base_grid, int(base_grid)))
    budgets = [int(base_grid) for _ in range(num_layers)]
    capacities = [int(max_grid - base_grid) for _ in range(num_layers)]

    remaining = int(target_total - base_grid * num_layers)
    if remaining <= 0:
        return budgets

    score_tensor = torch.tensor(scores, dtype=torch.float64)
    if threshold_mode == "percentile":
        low_q = torch.quantile(score_tensor, float(low_percentile) / 100.0)
        high_q = torch.quantile(score_tensor, float(high_percentile) / 100.0)
        denom = float(max((high_q - low_q).item(), 1e-12))
        tail = torch.clamp((score_tensor - low_q) / denom, min=0.0)
    elif threshold_mode == "auto":
        median = torch.quantile(score_tensor, 0.5)
        mad = torch.quantile((score_tensor - median).abs(), 0.5)
        robust_scale = float(max((1.4826 * mad).item(), 1e-12))
        threshold = float(median.item()) + float(threshold_alpha) * robust_scale
        max_score = float(score_tensor.max().item())
        denom = max(max_score - threshold, robust_scale, 1e-12)
        tail = torch.clamp((score_tensor - threshold) / denom, min=0.0)
        if torch.allclose(tail, torch.zeros_like(tail)):
            tail = torch.clamp(score_tensor - median, min=0.0)
            fallback_max = float(max(tail.max().item(), 1e-12))
            tail = tail / fallback_max
    else:
        raise ValueError(f"Unsupported BFQ upper-tail threshold_mode: {threshold_mode}")
    tail = torch.pow(tail, float(gamma))

    if torch.allclose(tail, torch.zeros_like(tail)):
        probs = torch.full((num_layers,), 1.0 / num_layers, dtype=torch.float64)
    else:
        probs = tail / tail.sum()

    active = torch.ones(num_layers, dtype=torch.bool)
    while remaining > 0 and bool(active.any()):
        active_idx = active.nonzero(as_tuple=True)[0]
        active_probs = probs[active_idx]
        if torch.allclose(active_probs, torch.zeros_like(active_probs)):
            active_probs = torch.full_like(active_probs, 1.0 / len(active_idx))
        else:
            active_probs = active_probs / active_probs.sum()
        raw_extra = active_probs * remaining

        saturated = False
        for local_idx, layer_idx in enumerate(active_idx.tolist()):
            cap = capacities[layer_idx]
            if cap <= 0:
                active[layer_idx] = False
                continue
            if raw_extra[local_idx].item() >= cap - 1e-12:
                budgets[layer_idx] += cap
                remaining -= cap
                capacities[layer_idx] = 0
                active[layer_idx] = False
                saturated = True
        if saturated:
            continue

        floor_extra = torch.floor(raw_extra).to(torch.int64)
        if int(floor_extra.sum().item()) > 0:
            for local_idx, add in enumerate(floor_extra.tolist()):
                if add <= 0:
                    continue
                layer_idx = active_idx[local_idx].item()
                add = min(add, capacities[layer_idx], remaining)
                if add <= 0:
                    continue
                budgets[layer_idx] += add
                capacities[layer_idx] -= add
                remaining -= add
                if capacities[layer_idx] <= 0:
                    active[layer_idx] = False

        if remaining <= 0:
            break

        frac_extra = raw_extra - floor_extra.to(raw_extra.dtype)
        order = torch.argsort(frac_extra, descending=True).tolist()
        progressed = False
        for local_idx in order:
            layer_idx = active_idx[local_idx].item()
            if remaining <= 0:
                break
            if capacities[layer_idx] <= 0:
                active[layer_idx] = False
                continue
            budgets[layer_idx] += 1
            capacities[layer_idx] -= 1
            remaining -= 1
            progressed = True
            if capacities[layer_idx] <= 0:
                active[layer_idx] = False

        if not progressed:
            break

    return budgets


def _allocate_budget_upper_tail_with_low_penalty(
    scores,
    floor_grid: int,
    max_grid: int,
    target_mean: float,
    base_grid: int,
    bonus_low_percentile: float = 50.0,
    bonus_high_percentile: float = 95.0,
    bonus_gamma: float = 2.0,
    bonus_threshold_mode: str = "percentile",
    bonus_threshold_alpha: float = 1.0,
    penalty_low_percentile: float = 5.0,
    penalty_high_percentile: float = 20.0,
    penalty_gamma: float = 2.0,
):
    if len(scores) == 0:
        return []

    num_layers = len(scores)
    min_total = int(num_layers * floor_grid)
    max_total = int(num_layers * max_grid)
    target_total = int(round(target_mean * num_layers))
    target_total = max(min_total, min(max_total, target_total))

    max_base_grid = max(floor_grid, min(max_grid, int(target_total // num_layers)))
    base_grid = max(floor_grid, min(max_base_grid, int(base_grid)))
    budgets = [int(base_grid) for _ in range(num_layers)]

    score_tensor = torch.tensor(scores, dtype=torch.float64)

    # First, reduce only clearly low-score layers below the baseline grid.
    if base_grid > floor_grid:
        low_q = torch.quantile(score_tensor, float(penalty_low_percentile) / 100.0)
        high_q = torch.quantile(score_tensor, float(penalty_high_percentile) / 100.0)
        denom = float(max((high_q - low_q).item(), 1e-12))
        low_tail = torch.clamp((high_q - score_tensor) / denom, min=0.0, max=1.0)
        low_tail = torch.pow(low_tail, float(penalty_gamma))
        penalty_caps = [int(base_grid - floor_grid) for _ in range(num_layers)]
        for idx, frac in enumerate(low_tail.tolist()):
            if penalty_caps[idx] <= 0:
                continue
            penalty = int(round(float(frac) * penalty_caps[idx]))
            penalty = max(0, min(penalty, penalty_caps[idx]))
            budgets[idx] -= penalty

    remaining = int(target_total - sum(budgets))
    if remaining <= 0:
        return budgets

    capacities = [int(max_grid - b) for b in budgets]
    if bonus_threshold_mode == "percentile":
        low_q = torch.quantile(score_tensor, float(bonus_low_percentile) / 100.0)
        high_q = torch.quantile(score_tensor, float(bonus_high_percentile) / 100.0)
        denom = float(max((high_q - low_q).item(), 1e-12))
        tail = torch.clamp((score_tensor - low_q) / denom, min=0.0)
    elif bonus_threshold_mode == "auto":
        median = torch.quantile(score_tensor, 0.5)
        mad = torch.quantile((score_tensor - median).abs(), 0.5)
        robust_scale = float(max((1.4826 * mad).item(), 1e-12))
        threshold = float(median.item()) + float(bonus_threshold_alpha) * robust_scale
        max_score = float(score_tensor.max().item())
        denom = max(max_score - threshold, robust_scale, 1e-12)
        tail = torch.clamp((score_tensor - threshold) / denom, min=0.0)
        if torch.allclose(tail, torch.zeros_like(tail)):
            tail = torch.clamp(score_tensor - median, min=0.0)
            fallback_max = float(max(tail.max().item(), 1e-12))
            tail = tail / fallback_max
    else:
        raise ValueError(f"Unsupported BFQ upper-tail bonus_threshold_mode: {bonus_threshold_mode}")
    tail = torch.pow(tail, float(bonus_gamma))

    if torch.allclose(tail, torch.zeros_like(tail)):
        probs = torch.full((num_layers,), 1.0 / num_layers, dtype=torch.float64)
    else:
        probs = tail / tail.sum()

    active = torch.ones(num_layers, dtype=torch.bool)
    while remaining > 0 and bool(active.any()):
        active_idx = active.nonzero(as_tuple=True)[0]
        active_probs = probs[active_idx]
        prob_sum = float(active_probs.sum().item())
        if prob_sum <= 0:
            active_probs = torch.full_like(active_probs, 1.0 / float(active_idx.numel()), dtype=torch.float64)
        else:
            active_probs = active_probs / prob_sum

        raw_extra = active_probs * float(remaining)
        saturated = False
        for local_idx, layer_idx in enumerate(active_idx.tolist()):
            cap = capacities[layer_idx]
            if cap <= 0:
                active[layer_idx] = False
                continue
            if raw_extra[local_idx].item() >= cap:
                budgets[layer_idx] += cap
                remaining -= cap
                capacities[layer_idx] = 0
                active[layer_idx] = False
                saturated = True
        if saturated:
            continue

        floor_extra = torch.floor(raw_extra).to(torch.int64)
        if int(floor_extra.sum().item()) > 0:
            for local_idx, add in enumerate(floor_extra.tolist()):
                if add <= 0:
                    continue
                layer_idx = active_idx[local_idx].item()
                add = min(add, capacities[layer_idx], remaining)
                if add <= 0:
                    continue
                budgets[layer_idx] += add
                capacities[layer_idx] -= add
                remaining -= add
                if capacities[layer_idx] <= 0:
                    active[layer_idx] = False

        if remaining <= 0:
            break

        frac_extra = raw_extra - floor_extra.to(raw_extra.dtype)
        order = torch.argsort(frac_extra, descending=True).tolist()
        progressed = False
        for local_idx in order:
            layer_idx = active_idx[local_idx].item()
            if remaining <= 0:
                break
            if capacities[layer_idx] <= 0:
                active[layer_idx] = False
                continue
            budgets[layer_idx] += 1
            capacities[layer_idx] -= 1
            remaining -= 1
            progressed = True
            if capacities[layer_idx] <= 0:
                active[layer_idx] = False
        if not progressed:
            break

    return budgets


def build_bfq_policy(
    process_model,
    prompt_inputs: Dict,
    prompt_kwargs: Dict,
    w_bit: int,
    a_bit: int,
    w_group: int,
    wa_quant: bool = False,
    policy_n_samples: int = 16,
    micro_batch_size: int = 4,
    beta: float = 0.5,
    n_grid_max: int = 32,
    n_grid_target_mean: float = 20.0,
    budget_conflict_alpha: float = 0.5,
    budget_temperature: float = 4.0,
    aggressiveness_temperature: float = 3.0,
    ratio_upper_min: float = 0.5,
    ratio_upper_max: float = 1.0,
    modality_ratio_min: float = 0.25,
    modality_ratio_max: float = 4.0,
    weight_signed_clip_percentile: float = 100.0,
    fixed_budget: bool = False,
    fixed_aggressiveness: bool = False,
    upper_tail_budget: bool = False,
    budget_base_grid: int = 16,
    budget_bonus_low_percentile: float = 50.0,
    budget_bonus_high_percentile: float = 95.0,
    budget_bonus_gamma: float = 2.0,
    budget_bonus_threshold_mode: str = "percentile",
    budget_bonus_threshold_alpha: float = 1.0,
    lower_tail_penalty: bool = False,
    budget_penalty_low_percentile: float = 5.0,
    budget_penalty_high_percentile: float = 20.0,
    budget_penalty_gamma: float = 2.0,
    conflict_mode: str = "shared",
    budget_target_mode: str = "both",
    importance_cache_path: Optional[str] = None,
    disable_importance_cache: bool = False,
    eps: float = 1e-6,
):
    process_model.to_cuda()
    layers = get_blocks(process_model.model)
    num_layers = len(layers)
    full_batch_size = _infer_batch_size(prompt_inputs, prompt_kwargs)
    if policy_n_samples > 0 and policy_n_samples < full_batch_size:
        prompt_inputs, prompt_kwargs = _slice_batch(
            prompt_inputs,
            prompt_kwargs,
            0,
            policy_n_samples,
            full_batch_size,
        )
        print(f"[BFQ] using first {policy_n_samples} / {full_batch_size} calibration samples for policy analysis")
    else:
        print(f"[BFQ] using all {full_batch_size} calibration samples for policy analysis")

    cache_metadata = _build_importance_cache_metadata(
        prompt_inputs,
        prompt_kwargs,
        w_bit=w_bit,
        a_bit=a_bit,
        w_group=w_group,
        wa_quant=wa_quant,
        policy_n_samples=policy_n_samples,
        micro_batch_size=micro_batch_size,
        beta=beta,
        num_layers=num_layers,
    )
    print(f"[BFQ] building quantization-effect policy: num_layers={num_layers}")

    weight_only_policy = (
        (not wa_quant)
        and abs(float(modality_ratio_min) - 1.0) < 1e-12
        and abs(float(modality_ratio_max) - 1.0) < 1e-12
    )
    probe_components = ("weight",) if weight_only_policy else ("weight", "vision", "text")
    cache_metadata["modality_ratio_min"] = float(modality_ratio_min)
    cache_metadata["modality_ratio_max"] = float(modality_ratio_max)
    cache_metadata["weight_only_policy"] = bool(weight_only_policy)
    cache_metadata["probe_components"] = list(probe_components)
    if weight_only_policy:
        print("[BFQ] W3A16 with modality reweight disabled: using weight-only importance probes")

    loaded_from_cache = False
    baseline = None
    records = {}
    if importance_cache_path and (not disable_importance_cache) and os.path.exists(importance_cache_path):
        cached_metadata, baseline, records = _load_importance_cache(importance_cache_path)
        if cached_metadata:
            mismatched = []
            required_keys = (
                "w_bit",
                "a_bit",
                "w_group",
                "wa_quant",
                "policy_n_samples",
                "num_layers",
                "modality_ratio_min",
                "modality_ratio_max",
                "weight_only_policy",
                "probe_components",
            )
            for key in required_keys:
                if key not in cached_metadata:
                    mismatched.append(f"{key}: missing in cache")
                    continue
                if key not in cache_metadata:
                    mismatched.append(f"{key}: missing in current metadata")
                    continue
                if cached_metadata[key] != cache_metadata[key]:
                    mismatched.append(f"{key}: cached={cached_metadata[key]} current={cache_metadata[key]}")
            if mismatched:
                print("[BFQ] warning: importance cache metadata differs from current run:")
                for item in mismatched:
                    print(f"[BFQ]   {item}")
            else:
                loaded_from_cache = True
        if loaded_from_cache:
            print(f"[BFQ] reusing quantization-effect importance cache: {importance_cache_path}")
        else:
            print(f"[BFQ] ignoring incompatible quantization-effect importance cache: {importance_cache_path}")

    if not loaded_from_cache:
        baseline = _evaluate_avg_loss(process_model, prompt_inputs, prompt_kwargs, micro_batch_size)
        print(
            f"[BFQ] baseline calibration loss={baseline['avg_loss']:.6f} "
            f"(targets={baseline['num_targets']}, batches={baseline['num_batches']})"
        )

        for layer_id in tqdm.tqdm(range(num_layers), desc="BFQ policy analysis"):
            print(f"[BFQ] layer {layer_id + 1}/{num_layers}")
            for component in probe_components:
                backup_layer, backup_device = _snapshot_layer(process_model.model, layer_id)
                try:
                    print(f"[BFQ] probing layer={layer_id} component={component}")
                    if component == "weight":
                        _quantize_layer_weights(process_model.model, layer_id, w_bit=w_bit, w_group=w_group)
                    else:
                        _replace_layer_with_act_quant(
                            process_model.model,
                            layer_id=layer_id,
                            w_bit=w_bit,
                            a_bit=a_bit,
                            token_scope=component,
                        )

                    result = _evaluate_avg_loss(process_model, prompt_inputs, prompt_kwargs, micro_batch_size)
                    delta_loss = result["avg_loss"] - baseline["avg_loss"]
                    harmful = max(delta_loss, 0.0)
                    beneficial = max(-delta_loss, 0.0)
                    records[(layer_id, component)] = {
                        "quant_loss": result["avg_loss"],
                        "delta_loss": delta_loss,
                        "harmful_score": harmful,
                        "beneficial_score": beneficial,
                        "risk_score": max(harmful - beta * beneficial, eps),
                    }
                    print(
                        f"[BFQ] layer={layer_id} component={component} "
                        f"quant_loss={result['avg_loss']:.6f} "
                        f"delta={delta_loss:.6f} "
                        f"harmful={harmful:.6f} "
                        f"beneficial={beneficial:.6f}"
                    )
                finally:
                    _restore_layer(process_model.model, layer_id, backup_layer, backup_device)

            if weight_only_policy:
                for component in ("vision", "text"):
                    records[(layer_id, component)] = {
                        "quant_loss": baseline["avg_loss"],
                        "delta_loss": 0.0,
                        "harmful_score": 0.0,
                        "beneficial_score": 0.0,
                        "risk_score": 1.0,
                    }

        if importance_cache_path and (not disable_importance_cache):
            _save_importance_cache(
                importance_cache_path,
                metadata=cache_metadata,
                baseline=baseline,
                records=records,
            )
            print(f"[BFQ] saved quantization-effect importance cache: {importance_cache_path}")
    else:
        print(
            f"[BFQ] baseline calibration loss={baseline['avg_loss']:.6f} "
            f"(targets={baseline['num_targets']}, batches={baseline['num_batches']})"
        )

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
            records[(layer_id, "vision")]["risk_score"]
            + records[(layer_id, "text")]["risk_score"]
        )
        signed_weight_score = (
            records[(layer_id, "weight")]["harmful_score"]
            - beta * records[(layer_id, "weight")]["beneficial_score"]
        )
        if weight_only_policy:
            signed_activation_score = 0.0
            activation_score = 0.0
            conflict_score = 0.0
            vt_conflict_score = 0.0
        else:
            signed_activation_score = (
                records[(layer_id, "vision")]["harmful_score"]
                - beta * records[(layer_id, "vision")]["beneficial_score"]
                + records[(layer_id, "text")]["harmful_score"]
                - beta * records[(layer_id, "text")]["beneficial_score"]
            )
            conflict_score = float(
                torch.tensor([delta_w, delta_v, delta_t], dtype=torch.float32).std(unbiased=False).item()
            )
            vt_conflict_score = float(
                torch.tensor([delta_v, delta_t], dtype=torch.float32).std(unbiased=False).item()
            )
        if conflict_mode == "shared":
            weight_conflict_score = conflict_score
            activation_conflict_score = conflict_score
        elif conflict_mode == "split_vt":
            weight_conflict_score = 0.0
            activation_conflict_score = vt_conflict_score
        elif conflict_mode == "none":
            weight_conflict_score = 0.0
            activation_conflict_score = 0.0
        else:
            raise ValueError(f"Unsupported BFQ conflict_mode: {conflict_mode}")
        weight_scores.append(weight_score)
        activation_scores.append(activation_score)
        signed_weight_scores.append(signed_weight_score)
        signed_activation_scores.append(signed_activation_score)
        weight_budget_scores.append(signed_weight_score + budget_conflict_alpha * weight_conflict_score)
        activation_budget_scores.append(signed_activation_score + budget_conflict_alpha * activation_conflict_score)
        conflict_scores.append(conflict_score)
        weight_conflict_scores.append(weight_conflict_score)
        activation_conflict_scores.append(activation_conflict_score)

    clipped_signed_weight_scores = _clip_signed_scores(
        signed_weight_scores,
        percentile=weight_signed_clip_percentile,
    )

    norm_weight_scores = _normalize_list(weight_scores)
    norm_activation_scores = _normalize_list(activation_scores)
    norm_signed_weight_scores = _signed_scores_to_unit(
        clipped_signed_weight_scores,
        temperature=aggressiveness_temperature,
    )
    norm_signed_activation_scores = _signed_scores_to_unit(
        signed_activation_scores,
        temperature=aggressiveness_temperature,
    )
    clipped_weight_budget_scores = [
        clipped_signed_weight_scores[i] + budget_conflict_alpha * weight_conflict_scores[i]
        for i in range(num_layers)
    ]

    norm_weight_budget_scores = _normalize_list(clipped_weight_budget_scores)
    norm_activation_budget_scores = _normalize_list(activation_budget_scores)
    norm_conflict_scores = _normalize_list(conflict_scores)
    floor_grid = int(budget_base_grid)
    fixed_grid = int(round(n_grid_target_mean))
    fixed_grid = max(floor_grid, min(n_grid_max, fixed_grid))

    if fixed_budget:
        weight_budgets = [fixed_grid for _ in range(num_layers)]
        activation_budgets = [fixed_grid for _ in range(num_layers)]
    else:
        if upper_tail_budget and lower_tail_penalty:
            adaptive_weight_budgets = _allocate_budget_upper_tail_with_low_penalty(
                clipped_weight_budget_scores,
                floor_grid=floor_grid,
                max_grid=n_grid_max,
                target_mean=n_grid_target_mean,
                base_grid=budget_base_grid,
                bonus_low_percentile=budget_bonus_low_percentile,
                bonus_high_percentile=budget_bonus_high_percentile,
                bonus_gamma=budget_bonus_gamma,
                bonus_threshold_mode=budget_bonus_threshold_mode,
                bonus_threshold_alpha=budget_bonus_threshold_alpha,
                penalty_low_percentile=budget_penalty_low_percentile,
                penalty_high_percentile=budget_penalty_high_percentile,
                penalty_gamma=budget_penalty_gamma,
            )
            adaptive_activation_budgets = _allocate_budget_upper_tail_with_low_penalty(
                activation_budget_scores,
                floor_grid=floor_grid,
                max_grid=n_grid_max,
                target_mean=n_grid_target_mean,
                base_grid=budget_base_grid,
                bonus_low_percentile=budget_bonus_low_percentile,
                bonus_high_percentile=budget_bonus_high_percentile,
                bonus_gamma=budget_bonus_gamma,
                bonus_threshold_mode=budget_bonus_threshold_mode,
                bonus_threshold_alpha=budget_bonus_threshold_alpha,
                penalty_low_percentile=budget_penalty_low_percentile,
                penalty_high_percentile=budget_penalty_high_percentile,
                penalty_gamma=budget_penalty_gamma,
            )
        elif upper_tail_budget:
            adaptive_weight_budgets = _allocate_budget_upper_tail(
                clipped_weight_budget_scores,
                floor_grid=floor_grid,
                max_grid=n_grid_max,
                target_mean=n_grid_target_mean,
                base_grid=budget_base_grid,
                low_percentile=budget_bonus_low_percentile,
                high_percentile=budget_bonus_high_percentile,
                gamma=budget_bonus_gamma,
                threshold_mode=budget_bonus_threshold_mode,
                threshold_alpha=budget_bonus_threshold_alpha,
            )
            adaptive_activation_budgets = _allocate_budget_upper_tail(
                activation_budget_scores,
                floor_grid=floor_grid,
                max_grid=n_grid_max,
                target_mean=n_grid_target_mean,
                base_grid=budget_base_grid,
                low_percentile=budget_bonus_low_percentile,
                high_percentile=budget_bonus_high_percentile,
                gamma=budget_bonus_gamma,
                threshold_mode=budget_bonus_threshold_mode,
                threshold_alpha=budget_bonus_threshold_alpha,
            )
        else:
            adaptive_weight_budgets = _allocate_budget(
                clipped_weight_budget_scores,
                floor_grid=floor_grid,
                max_grid=n_grid_max,
                target_mean=n_grid_target_mean,
                temperature=budget_temperature,
            )
            adaptive_activation_budgets = _allocate_budget(
                activation_budget_scores,
                floor_grid=floor_grid,
                max_grid=n_grid_max,
                target_mean=n_grid_target_mean,
                temperature=budget_temperature,
            )

        if budget_target_mode == "both":
            weight_budgets = adaptive_weight_budgets
            activation_budgets = adaptive_activation_budgets
        elif budget_target_mode == "weight_only":
            weight_budgets = adaptive_weight_budgets
            activation_budgets = [fixed_grid for _ in range(num_layers)]
        elif budget_target_mode == "act_only":
            weight_budgets = [fixed_grid for _ in range(num_layers)]
            activation_budgets = adaptive_activation_budgets
        else:
            raise ValueError(f"Unsupported BFQ budget_target_mode: {budget_target_mode}")

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

        if fixed_aggressiveness:
            ratio_max_weight = float(ratio_upper_max)
            ratio_max_activation = float(ratio_upper_max)
        else:
            ratio_max_weight = float(ratio_upper_max - (ratio_upper_max - ratio_upper_min) * norm_signed_weight)
            ratio_max_activation = float(ratio_upper_max - (ratio_upper_max - ratio_upper_min) * norm_signed_activation)

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
            search_grid = int(round(
                search_weight_share * weight_grid + search_activation_share * activation_grid
            ))
            search_grid = max(floor_grid, min(n_grid_max, search_grid))
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
            reweight_ratio = float(risk_vis / max(risk_text, eps))
            reweight_ratio = max(modality_ratio_min, min(modality_ratio_max, reweight_ratio))
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
        log_parts = [
            f"[BFQ][policy] layer={layer_id}",
            f"weight_grid={int(weight_grid)}",
        ]
        if wa_quant:
            log_parts.append(f"act_grid={int(activation_grid)}")
        log_parts.append(f"search_grid={int(search_grid)}")
        if not fixed_aggressiveness:
            log_parts.append(f"weight_ratio_max={ratio_max_weight:.4f}")
            if wa_quant:
                log_parts.append(f"act_ratio_max={ratio_max_activation:.4f}")
            log_parts.append(f"search_ratio_max={search_ratio_max:.4f}")
        if not weight_only_policy:
            log_parts.append(f"reweight_ratio={reweight_ratio:.4f}")
        log_parts.append(f"conflict={conflict_scores[layer_id]:.6f}")
        if conflict_mode != "shared":
            log_parts.extend(
                [
                    f"w_conflict={weight_conflict_scores[layer_id]:.6f}",
                    f"a_conflict={activation_conflict_scores[layer_id]:.6f}",
                ]
            )
        print(" ".join(log_parts))

    weight_grid_values = [layer["weight_policy"]["n_grid"] for layer in policy_layers.values()]
    act_grid_values = [layer["activation_policy"]["n_grid"] for layer in policy_layers.values()]
    weight_ratio_values = [layer["weight_policy"]["ratio_max"] for layer in policy_layers.values()]
    act_ratio_values = [layer["activation_policy"]["ratio_max"] for layer in policy_layers.values()]
    summary_parts = [
        "[BFQ][summary]",
        f"search_type_share(weight/act)={search_weight_share:.4f}/{search_activation_share:.4f}",
        f"upper_tail_budget={int(bool(upper_tail_budget))}",
        f"bonus_threshold_mode={budget_bonus_threshold_mode}",
        f"bonus_threshold_alpha={budget_bonus_threshold_alpha:.4f}",
        f"lower_tail_penalty={int(bool(lower_tail_penalty))}",
        f"conflict_mode={conflict_mode}",
        f"budget_target_mode={budget_target_mode}",
        f"weight_grid(mean/min/max)={sum(weight_grid_values)/len(weight_grid_values):.2f}/"
        f"{min(weight_grid_values)}/{max(weight_grid_values)}",
    ]
    if wa_quant:
        summary_parts.append(
            f"act_grid(mean/min/max)={sum(act_grid_values)/len(act_grid_values):.2f}/"
            f"{min(act_grid_values)}/{max(act_grid_values)}"
        )
    summary_parts.append(
        f"weight_ratio(mean)={sum(weight_ratio_values)/len(weight_ratio_values):.4f}"
    )
    if wa_quant:
        summary_parts.append(
            f"act_ratio(mean)={sum(act_ratio_values)/len(act_ratio_values):.4f}"
        )
    print(" ".join(summary_parts))

    return {
        "baseline_loss": baseline["avg_loss"],
        "beta": beta,
        "n_grid_max": int(n_grid_max),
        "n_grid_target_mean": float(n_grid_target_mean),
        "budget_conflict_alpha": float(budget_conflict_alpha),
        "budget_temperature": float(budget_temperature),
        "aggressiveness_temperature": float(aggressiveness_temperature),
        "search_weight_share": float(search_weight_share),
        "search_activation_share": float(search_activation_share),
        "ratio_upper_min": float(ratio_upper_min),
        "ratio_upper_max": float(ratio_upper_max),
        "modality_ratio_min": float(modality_ratio_min),
        "modality_ratio_max": float(modality_ratio_max),
        "weight_signed_clip_percentile": float(weight_signed_clip_percentile),
        "fixed_budget": bool(fixed_budget),
        "fixed_aggressiveness": bool(fixed_aggressiveness),
        "upper_tail_budget": bool(upper_tail_budget),
        "budget_base_grid": int(budget_base_grid),
        "budget_bonus_low_percentile": float(budget_bonus_low_percentile),
        "budget_bonus_high_percentile": float(budget_bonus_high_percentile),
        "budget_bonus_gamma": float(budget_bonus_gamma),
        "budget_bonus_threshold_mode": str(budget_bonus_threshold_mode),
        "budget_bonus_threshold_alpha": float(budget_bonus_threshold_alpha),
        "lower_tail_penalty": bool(lower_tail_penalty),
        "budget_penalty_low_percentile": float(budget_penalty_low_percentile),
        "budget_penalty_high_percentile": float(budget_penalty_high_percentile),
        "budget_penalty_gamma": float(budget_penalty_gamma),
        "conflict_mode": str(conflict_mode),
        "budget_target_mode": str(budget_target_mode),
        "layers": policy_layers,
    }
