#!/usr/bin/env python
import argparse
import copy
import csv
import gc
import json
import os
import sys
import weakref
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn


def _inject_local_sources(root: Path) -> None:
    candidates = [
        os.environ.get("LLAVA_SRC_PATH"),
        str(root / "3rdparty" / "LLaVA-NeXT"),
        os.environ.get("LMMS_EVAL_SRC_PATH"),
        str(root / "3rdparty" / "lmms-eval"),
        str(root),
    ]
    for cand in candidates:
        if cand and os.path.isdir(cand) and cand not in sys.path:
            sys.path.insert(0, cand)


ROOT = Path(__file__).resolve().parents[1]
_inject_local_sources(ROOT)

from lmms_eval.models import get_model  # noqa: E402
from qmllm.calibration.coco_vl import load_image  # noqa: E402
from qmllm.methods.bfq.policy import _infer_batch_size, _slice_batch  # noqa: E402
from qmllm.methods.mbq.quantize.pre_quant import get_blocks, get_named_linears  # noqa: E402
from qmllm.methods.mbq.quantize.quantizer import get_module_by_name_suffix  # noqa: E402
from qmllm.models import get_process_model  # noqa: E402
from qmllm.quantization.qlinear import WALinear  # noqa: E402
from qmllm.quantization.quant_funcs import pseudo_quantize_tensor  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibration-set quantization-effect analysis for layer-wise weight/vision/text perturbations.")
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--model", required=True, choices=["qwen2_vl", "qwen2_5_vl", "llava_onevision", "internvl2"])
    parser.add_argument("--model_args", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--data_json", required=True, help="Path to the calibration JSON or JSONL file.")
    parser.add_argument("--image_root", required=True, help="Root directory for calibration images.")
    parser.add_argument("--n_samples", type=int, default=64)
    parser.add_argument("--no_shuffle", action="store_true", help="Read an ordered calibration manifest without resampling")
    parser.add_argument("--micro_batch_size", type=int, default=4)
    parser.add_argument("--w_bit", type=int, default=4)
    parser.add_argument("--act_a_bit", type=int, default=8)
    parser.add_argument("--w_group", type=int, default=128)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timing_json", default=None)
    parser.add_argument(
        "--components",
        default="weight,vision,text",
        help="Comma-separated components to evaluate: weight,vision,text",
    )
    return parser.parse_args()


def _process_model_name(model_name: str) -> str:
    return "qwen2_vl" if model_name == "qwen2_5_vl" else model_name


def _build_process_model(args: argparse.Namespace):
    ModelClass = get_model(args.model)
    lm = ModelClass.create_from_arg_string(
        args.model_args,
        {
            "batch_size": "1",
            "device": args.device,
        },
    )
    ProcessModelClass = get_process_model(_process_model_name(args.model))
    process_model = ProcessModelClass(
        lm._model,
        lm._tokenizer,
        lm.processor if hasattr(lm, "processor") else None,
    )
    process_model.to_cuda()
    return lm, process_model


def _load_calibration_examples(process_model, data_json: str, image_root: str, n_samples: int, seed: int, shuffle: bool = True) -> List[dict]:
    if data_json.endswith(".jsonl"):
        dataset = []
        with open(data_json, "r") as f:
            for line in f:
                dataset.append(json.loads(line.strip()))
    elif data_json.endswith(".json"):
        with open(data_json, "r") as f:
            dataset = json.load(f)
    else:
        raise ValueError(f"Unsupported data file: {data_json}")

    if not dataset:
        raise ValueError(f"Empty calibration data: {data_json}")
    if shuffle:
        rng = torch.Generator().manual_seed(seed)
        perm = torch.randperm(len(dataset), generator=rng).tolist()
        selected = [dataset[i] for i in perm]
    else:
        if n_samples != len(dataset):
            raise ValueError(f"Ordered BFQ manifest has {len(dataset)} records, expected {n_samples}")
        selected = dataset

    examples = []
    for i in range(n_samples):
        item = selected[i % len(selected)]
        if "image" in item and item["image"]:
            if isinstance(item["image"], list):
                images = [load_image(os.path.join(image_root, p)) for p in item["image"]]
            else:
                images = [load_image(os.path.join(image_root, item["image"]))]
        else:
            images = None
        examples.append(process_model.preprocess_data(images, item))
    return examples


def _fit_mask_2d(mask: Optional[torch.Tensor], batch_size: int, seq_len: int, device: torch.device) -> Optional[torch.Tensor]:
    if not torch.is_tensor(mask):
        return None
    mask = mask.to(device=device, dtype=torch.bool)
    if mask.dim() == 1:
        mask = mask.unsqueeze(0)
    elif mask.dim() > 2:
        mask = mask.reshape(mask.shape[0], -1)
    if mask.shape[0] < batch_size:
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


def _set_runtime_masks(owner_model, vision_mask: Optional[torch.Tensor], attention_mask: Optional[torch.Tensor], ref: Optional[torch.Tensor]) -> None:
    device = ref.device if torch.is_tensor(ref) else None
    if device is None:
        owner_model._signed_runtime_vision_mask = None
        owner_model._signed_runtime_attention_mask = None
        return
    batch_size, seq_len = ref.shape[0], ref.shape[1]
    owner_model._signed_runtime_vision_mask = _fit_mask_2d(vision_mask, batch_size, seq_len, device)
    owner_model._signed_runtime_attention_mask = _fit_mask_2d(attention_mask, batch_size, seq_len, device)
    if owner_model._signed_runtime_attention_mask is None:
        owner_model._signed_runtime_attention_mask = torch.ones((batch_size, seq_len), dtype=torch.bool, device=device)
    if owner_model._signed_runtime_vision_mask is None:
        owner_model._signed_runtime_vision_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=device)


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

        vision_mask = getattr(owner_model, "_signed_runtime_vision_mask", None)
        attention_mask = getattr(owner_model, "_signed_runtime_attention_mask", None)
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
            return torch.functional.F.linear(x, self.weight, self.bias)

        q_x = x.clone()
        q_x[selected] = self.act_quant(q_x[selected])
        y = torch.functional.F.linear(q_x, self.weight, self.bias)
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


def _evaluate_avg_loss(
    process_model,
    prompt_inputs: Dict,
    prompt_kwargs: Dict,
    micro_batch_size: int,
) -> Dict[str, float]:
    total_loss = 0.0
    total_targets = 0
    num_batches = 0
    owner_model = process_model.model

    batch_size = _infer_batch_size(prompt_inputs, prompt_kwargs)
    for start in range(0, batch_size, micro_batch_size):
        cur_inputs, cur_kwargs = _slice_batch(
            prompt_inputs, prompt_kwargs, start, min(start + micro_batch_size, batch_size), batch_size
        )
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
        if valid_targets <= 0:
            continue
        total_loss += float(outputs.loss.detach().float().item()) * valid_targets
        total_targets += int(valid_targets)
        num_batches += 1

        del cur_inputs, cur_kwargs, outputs, inputs_embeds, labels, attention_mask, vision_mask
        gc.collect()
        torch.cuda.empty_cache()

    if total_targets == 0:
        raise RuntimeError("No valid supervised targets found in calibration examples.")
    return {
        "avg_loss": total_loss / total_targets,
        "num_targets": total_targets,
        "num_batches": num_batches,
    }


def _plot_analysis(records: List[dict], output_dir: Path, num_layers: int, components: List[str]) -> None:
    if not records:
        return
    metric_specs = [
        ("delta_loss", "Quantization Effect", "coolwarm"),
        ("harmful_score", "Harmful Score", "Reds"),
        ("beneficial_score", "Beneficial Score", "Blues"),
    ]

    component_to_idx = {comp: idx for idx, comp in enumerate(components)}
    metric_maps = {
        key: torch.zeros((len(components), num_layers), dtype=torch.float32)
        for key, _, _ in metric_specs
    }

    for rec in records:
        comp_idx = component_to_idx[rec["component"]]
        layer_id = int(rec["layer_id"])
        for key, _, _ in metric_specs:
            metric_maps[key][comp_idx, layer_id] = float(rec[key])

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    for key, title, cmap in metric_specs:
        arr = metric_maps[key].numpy()
        fig_w = max(10.0, num_layers * 0.45)
        fig, ax = plt.subplots(figsize=(fig_w, 3.8))
        im = ax.imshow(arr, aspect="auto", cmap=cmap)
        ax.set_title(f"{title} Heatmap")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Component")
        ax.set_xticks(list(range(num_layers)))
        ax.set_xticklabels(list(range(num_layers)), fontsize=7)
        ax.set_yticks(list(range(len(components))))
        ax.set_yticklabels(components)
        cbar = fig.colorbar(im, ax=ax)
        cbar.ax.set_ylabel(key, rotation=270, labelpad=14)
        fig.tight_layout()
        fig.savefig(plots_dir / f"{key}_heatmap.png", dpi=200)
        plt.close(fig)

    fig_w = max(11.0, num_layers * 0.5)
    fig, axes = plt.subplots(3, 1, figsize=(fig_w, 10), sharex=True)
    line_specs = [
        ("delta_loss", "Quantization Effect"),
        ("harmful_score", "Harmful Score"),
        ("beneficial_score", "Beneficial Score"),
    ]
    colors = {"weight": "#1f77b4", "vision": "#d62728", "text": "#2ca02c"}
    xs = list(range(num_layers))

    for ax, (key, title) in zip(axes, line_specs):
        for comp in components:
            ys = [float(metric_maps[key][component_to_idx[comp], i].item()) for i in xs]
            ax.plot(xs, ys, marker="o", markersize=3, linewidth=1.5, label=comp, color=colors[comp])
        if key == "delta_loss":
            ax.axhline(0.0, linestyle="--", linewidth=1.0, color="black", alpha=0.7)
        ax.set_ylabel(key)
        ax.set_title(title)
        ax.grid(alpha=0.25)

    axes[-1].set_xlabel("Layer")
    axes[0].legend(loc="best")
    fig.tight_layout()
    fig.savefig(plots_dir / "component_lineplots.png", dpi=200)
    plt.close(fig)


def main() -> None:
    import time
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _, process_model = _build_process_model(args)
    examples = _load_calibration_examples(
        process_model=process_model,
        data_json=args.data_json,
        image_root=args.image_root,
        n_samples=args.n_samples,
        seed=args.seed,
        shuffle=not args.no_shuffle,
    )

    layers = get_blocks(process_model.model)
    num_layers = len(layers)
    print(f"[info] model={args.model} n_samples={len(examples)} num_layers={num_layers}")

    # Match the paper protocol: pad the full calibration set once, then slice
    # that fixed batch for micro-batched forward passes.
    collated = process_model.data_collator(examples)
    prompt_inputs, prompt_kwargs = process_model.generate_input(collated)

    summary_path = output_dir / "summary.csv"
    details_dir = output_dir / "details"
    details_dir.mkdir(parents=True, exist_ok=True)
    components = [c.strip() for c in args.components.split(",") if c.strip()]
    valid_components = {"weight", "vision", "text"}
    invalid = [c for c in components if c not in valid_components]
    if invalid:
        raise ValueError(f"Unsupported components: {invalid}")
    if not components:
        raise ValueError("No components selected.")

    analysis_start = time.perf_counter()
    baseline = _evaluate_avg_loss(process_model, prompt_inputs, prompt_kwargs, args.micro_batch_size)
    baseline_json = details_dir / "baseline.json"
    baseline_json.write_text(json.dumps(baseline, indent=2))
    records: List[dict] = []

    with summary_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "variant",
                "layer_id",
                "component",
                "baseline_loss",
                "quant_loss",
                "delta_loss",
                "harmful_score",
                "beneficial_score",
                "num_targets",
                "details_json",
            ]
        )
        writer.writerow(
            [
                "baseline_fp",
                "",
                "",
                baseline["avg_loss"],
                baseline["avg_loss"],
                0.0,
                0.0,
                0.0,
                baseline["num_targets"],
                str(baseline_json),
            ]
        )
        f.flush()

        for layer_id in range(num_layers):
            for component in components:
                print(f"[layer] evaluating layer={layer_id} component={component}")
                backup_layer, backup_device = _snapshot_layer(process_model.model, layer_id)
                try:
                    if component == "weight":
                        _quantize_layer_weights(process_model.model, layer_id, args.w_bit, args.w_group)
                    else:
                        _replace_layer_with_act_quant(
                            process_model.model,
                            layer_id=layer_id,
                            w_bit=args.w_bit,
                            a_bit=args.act_a_bit,
                            token_scope=component,
                        )

                    result = _evaluate_avg_loss(process_model, prompt_inputs, prompt_kwargs, args.micro_batch_size)
                    delta_loss = result["avg_loss"] - baseline["avg_loss"]
                    harmful = max(delta_loss, 0.0)
                    beneficial = max(-delta_loss, 0.0)
                    result.update(
                        {
                            "baseline_loss": baseline["avg_loss"],
                            "delta_loss": delta_loss,
                            "harmful_score": harmful,
                            "beneficial_score": beneficial,
                            "component": component,
                            "layer_id": layer_id,
                        }
                    )
                    result_json = details_dir / f"layer_{layer_id:02d}_{component}.json"
                    result_json.write_text(json.dumps(result, indent=2))
                    records.append(
                        {
                            "layer_id": layer_id,
                            "component": component,
                            "delta_loss": delta_loss,
                            "harmful_score": harmful,
                            "beneficial_score": beneficial,
                        }
                    )
                    writer.writerow(
                        [
                            f"layer_{layer_id:02d}_{component}",
                            layer_id,
                            component,
                            baseline["avg_loss"],
                            result["avg_loss"],
                            delta_loss,
                            harmful,
                            beneficial,
                            result["num_targets"],
                            str(result_json),
                        ]
                    )
                    f.flush()
                    print(
                        f"[layer:{layer_id}:{component}] "
                        f"baseline={baseline['avg_loss']:.6f} "
                        f"quant={result['avg_loss']:.6f} "
                        f"delta={delta_loss:.6f}"
                    )
                finally:
                    _restore_layer(process_model.model, layer_id, backup_layer, backup_device)
    analysis_end = time.perf_counter()

    if args.timing_json:
        timing_payload = {
            "method": "quantization_effect_analysis",
            "analysis_label": "quantization_effect_analysis",
            "search_label": "",
            "analysis_seconds": analysis_end - analysis_start,
            "search_seconds": 0.0,
            "total_seconds": analysis_end - analysis_start,
            "overhead_seconds": 0.0,
        }
        Path(args.timing_json).write_text(json.dumps(timing_payload, indent=2, sort_keys=True))

    _plot_analysis(records, output_dir, num_layers, components)
    del collated, prompt_inputs, prompt_kwargs, process_model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[done] summary={summary_path}")


if __name__ == "__main__":
    main()
