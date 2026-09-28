import torch
import torch.nn as nn
import tqdm
import copy
import gc
import functools
import time
from collections import defaultdict
from typing import List, Tuple

import numpy as np
from torch.nn import CrossEntropyLoss
from transformers.models.bloom.modeling_bloom import BloomForCausalLM
from transformers.models.opt.modeling_opt import OPTForCausalLM
from transformers.models.llama.modeling_llama import LlamaForCausalLM

from qmllm.utils.search import append_str_prefix, get_op_name
from qmllm.methods.mbq_fg.quantize.auto_scale import auto_scale_block, apply_scale
from qmllm.quantization.qlinear import WALinear
from qmllm.quantization.quant_funcs import pseudo_quantize_tensor
from qmllm.methods.mbq_fg.quantize.quantizer import get_module_by_name_suffix

__all__ = ["run_qvlm"]

# ---------- helpers copied from MBQ with minor tweaks ----------

class GradCacheHook:
    def __init__(self, vis_masks, cap_masks):
        if vis_masks is None or cap_masks is None:
            raise ValueError
        self.hooks = []
        self.vis_masks = vis_masks.cpu()
        self.cap_masks = cap_masks.cpu()
        self.steps = {}
        self.grad_dict = {}

    def cache_grad_hook(self, module, inp, out, name):
        if name not in self.steps:
            self.steps[name] = 0
        if name not in self.grad_dict:
            self.grad_dict[name] = {"vis_grad": [], "cap_grad": []}
        output_grad = out[0].float()
        step = self.steps[name]
        B, N, C = output_grad.shape
        for batch_idx in range(B):
            vis_mask = self.vis_masks[step]
            cap_mask = self.cap_masks[step]
            vis_grad = output_grad[batch_idx][vis_mask]
            cap_grad = output_grad[batch_idx][cap_mask]
            vis_grad_avg = vis_grad.abs().mean()
            cap_grad_avg = cap_grad.abs().mean()
            self.grad_dict[name]["vis_grad"].append(vis_grad_avg.detach().cpu())
            self.grad_dict[name]["cap_grad"].append(cap_grad_avg.detach().cpu())
            step = step + 1
        self.steps[name] = step

    def register_hooks(self, layers):
        for n, m in layers.named_modules():
            if isinstance(m, nn.Linear) and any([_ in n for _ in ["wo", "w2", "down_proj", "o_proj", "v_proj", "gate_proj", "up_proj", "w1", "w3"]]):
                self.hooks.append(
                    m.register_full_backward_hook(
                        functools.partial(self.cache_grad_hook, name=f"layers.{n}")
                    )
                )

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def get_avg_grad_dict(self):
        avg_grad_dict = {}
        for name, grad_values in self.grad_dict.items():
            mean_vis = torch.mean(torch.stack(grad_values["vis_grad"]))
            mean_cap = torch.mean(torch.stack(grad_values["cap_grad"]))
            avg_grad_dict[name] = {
                "vis_avg_grad": mean_vis.item(),
                "cap_avg_grad": mean_cap.item()
            }
        return avg_grad_dict


def get_named_linears(module):
    return {name: m for name, m in module.named_modules() if isinstance(m, nn.Linear)}


def get_blocks(model):
    cname = model.__class__.__name__
    clower = cname.lower()
    def _find_by_classnames(root, classnames):
        seen = set()
        out = []
        for m in root.modules():
            if m.__class__.__name__ in classnames and id(m) not in seen:
                out.append(m)
                seen.add(id(m))
        return out

    if cname == "LlamaForCausalLM":
        layers = model.model.layers
    elif cname == "LlavaLlamaForCausalLM":
        layers = model.model.layers
    elif cname == "LlavaQwenForCausalLM":
        layers = model.model.layers
    elif cname == "InternLM2ForCausalLM":
        layers = model.model.layers
    elif cname == "InternVLChatModel":
        layers = model.language_model.model.layers
    elif cname == "Qwen2VLForConditionalGeneration" or "qwen2" in clower:
        if hasattr(model, "model") and hasattr(model.model, "language_model") and hasattr(model.model.language_model, "layers"):
            layers = model.model.language_model.layers
        elif hasattr(model, "model") and hasattr(model.model, "layers"):
            layers = model.model.layers
        else:
            layers = None
    elif "llavaonevision" in clower:
        candidates = [
            getattr(model, "language_model", None),
            getattr(model, "model", None),
            model,
        ]
        layers = None
        for cand in candidates:
            if cand is None:
                continue
            if hasattr(cand, "model") and hasattr(cand.model, "layers"):
                layers = cand.model.layers
                break
            if hasattr(cand, "layers"):
                layers = cand.layers
                break
            if hasattr(cand, "decoder") and hasattr(cand.decoder, "layers"):
                layers = cand.decoder.layers
                break
        if layers is None:
            layers = _find_by_classnames(model, {"Qwen2DecoderLayer", "Qwen2Block"})
        if layers is None or len(layers) == 0:
            raise NotImplementedError(f"Cannot find decoder layers for {type(model)}")
    elif cname == "LlavaLlamaModel":
        layers = model.llm.model.layers
    elif isinstance(model, OPTForCausalLM):
        layers = model.model.decoder.layers
    elif isinstance(model, BloomForCausalLM):
        layers = model.transformer.h
    elif "mpt" in str(model.__class__).lower():
        layers = model.transformer.blocks
    elif "falcon" in str(model.__class__).lower():
        layers = model.transformer.h
    elif "bigcode" in str(model.__class__).lower():
        layers = model.transformer.h
    elif "neox" in str(model.__class__).lower():
        layers = model.gpt_neox.layers
    else:
        raise NotImplementedError(type(model))
    return layers


def move_embed(model, device):
    def _move_if_has_module(obj, attr_name):
        if obj is not None and hasattr(obj, attr_name):
            mod = getattr(obj, attr_name)
            if mod is not None and hasattr(mod, "to"):
                setattr(obj, attr_name, mod.to(device))
                return True
        return False

    if isinstance(model, LlamaForCausalLM):
        model.model.embed_tokens = model.model.embed_tokens.to(device)
        model.model.rotary_emb = model.model.rotary_emb.to(device)
    elif isinstance(model, OPTForCausalLM):
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(device)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(device)
    elif isinstance(model, BloomForCausalLM):
        model.transformer.word_embeddings = model.transformer.word_embeddings.to(device)
        model.transformer.word_embeddings_layernorm = (
            model.transformer.word_embeddings_layernorm.to(device)
        )
    elif "mpt" in str(model.__class__).lower():
        model.transformer.wte = model.transformer.wte.to(device)
        model.transformer.emb_drop = model.transformer.emb_drop.to(device)
    elif "falcon" in str(model.__class__).lower():
        model.transformer.word_embeddings = model.transformer.word_embeddings.to(device)
    elif "bigcode" in str(model.__class__).lower():
        model.transformer.wte = model.transformer.wte.to(device)
        model.transformer.wpe = model.transformer.wpe.to(device)
        model.transformer.drop = model.transformer.drop.to(device)
    elif "neox" in str(model.__class__).lower():
        model.gpt_neox.embed_in = model.gpt_neox.embed_in.to(device)
        model.gpt_neox.emb_dropout = model.gpt_neox.emb_dropout.to(device)
        model.embed_out = model.embed_out.to(device)
    elif model.__class__.__name__ == "LlavaLlamaForCausalLM":
        model.model.embed_tokens = model.model.embed_tokens.to(device)
        model.model.vision_tower.vision_tower.vision_model.embeddings.to(device)
    elif model.__class__.__name__ == "LlavaQwenForCausalLM":
        model.model.embed_tokens = model.model.embed_tokens.to(device)
    elif model.__class__.__name__ == "InternLM2ForCausalLM":
        model.model.tok_embeddings = model.model.tok_embeddings.to(device)
    elif "llavaonevision" in str(model.__class__).lower():
        root = getattr(model, "language_model", None)
        if root is None:
            root = getattr(model, "model", None)
        if root is not None and hasattr(root, "embed_tokens"):
            root.embed_tokens = root.embed_tokens.to(device)
    elif model.__class__.__name__ == "InternVLChatModel":
        model.language_model.model.tok_embeddings = model.language_model.model.tok_embeddings.to(device)  
    elif model.__class__.__name__ == "Qwen2VLForConditionalGeneration":
        moved = False
        if hasattr(model, "get_input_embeddings"):
            try:
                emb = model.get_input_embeddings()
                if emb is not None and hasattr(emb, "to"):
                    emb.to(device)
                    moved = True
            except Exception:
                pass
        moved = _move_if_has_module(getattr(model, "model", None), "embed_tokens") or moved
        moved = _move_if_has_module(getattr(getattr(model, "model", None), "language_model", None), "embed_tokens") or moved
        if not moved:
            raise AttributeError(f"Cannot find embed_tokens/get_input_embeddings on {type(model)}")
    elif "qwen2" in str(model.__class__).lower() and hasattr(model, "model"):
        moved = False
        moved = _move_if_has_module(getattr(model, "model", None), "embed_tokens") or moved
        moved = _move_if_has_module(getattr(getattr(model, "model", None), "language_model", None), "embed_tokens") or moved
        if hasattr(model, "get_input_embeddings"):
            try:
                emb = model.get_input_embeddings()
                if emb is not None and hasattr(emb, "to"):
                    emb.to(device)
                    moved = True
            except Exception:
                pass
        if not moved:
            raise AttributeError(f"Cannot find Qwen2 embedding module on {type(model)}")
    elif model.__class__.__name__ == "LlavaLlamaModel":
        model.llm.model.embed_tokens = model.llm.model.embed_tokens.to(device)
    else:
        raise NotImplementedError(type(model))


def process_input(prompt_inputs, prompt_kwargs):
    inputs = {**prompt_inputs, **prompt_kwargs}
    inputs["use_cache"] = False
    vision_mask = inputs.pop("vision_mask", None)
    caption_mask = inputs.pop("caption_mask", None)
    return inputs, vision_mask, caption_mask


# ---------- QVLM-specific helpers ----------

def measure_layer_entropies(model, prompt_inputs, prompt_kwargs, max_samples: int = 16, bins: int = 256, micro_batch: int = None):
    backbone = getattr(model, "model", model)
    layers = get_blocks(backbone)
    entropies = [None] * len(layers)

    def make_hook(idx):
        def hook(module, inputs):
            x = inputs[0]
            if x.numel() == 0:
                entropies[idx] = 0.0
                return
            flat = x.detach().float().view(-1)
            if flat.numel() > max_samples * 1000:
                idxs = torch.randperm(flat.numel(), device=flat.device)[: max_samples * 1000]
                flat = flat[idxs]
            flat = flat.cpu()
            min_v = flat.min().item()
            max_v = flat.max().item()
            if min_v == max_v:
                entropies[idx] = 0.0
                return
            hist = torch.histc(flat, bins=bins, min=min_v, max=max_v)
            prob = hist / hist.sum()
            prob = prob[prob > 0]
            ent = -(prob * torch.log(prob)).sum().item()
            entropies[idx] = ent
        return hook

    handles = [layer.register_forward_pre_hook(make_hook(i)) for i, layer in enumerate(layers)]
    try:
        inputs, _, _ = process_input(prompt_inputs, prompt_kwargs)
        model.to_cuda()
        device = next(backbone.parameters()).device
        for k, v in list(inputs.items()):
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(device)
        # micro-batch entropy pass to reduce peak memory
        bsz = None
        for v in inputs.values():
            if isinstance(v, torch.Tensor):
                bsz = v.shape[0]
                break
        if bsz is None:
            bsz = 1
        mb = micro_batch if micro_batch is not None and micro_batch > 0 else bsz
        with torch.no_grad():
            start = 0
            # use first chunk only to keep behavior light
            end = min(start + mb, bsz)
            mini = {}
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    mini[k] = v[start:end]
                else:
                    mini[k] = v
            model(**mini)
    finally:
        for h in handles:
            h.remove()

    valid = [e for e in entropies if e is not None]
    if len(valid) == 0:
        return [0.0] * len(entropies)
    med = float(np.median(valid))
    entropies = [med if e is None else e for e in entropies]
    return entropies


def partition_blocks(entropies, max_block_len: int = 6, percentile: float = 0.6):
    if len(entropies) == 0:
        return []
    thresh = np.quantile(entropies, percentile)
    blocks = []
    start = 0
    cur_len = 0
    for idx, ent in enumerate(entropies):
        if cur_len == 0:
            start = idx
            cur_len = 1
            continue
        if ent > thresh or cur_len >= max_block_len:
            blocks.append((start, idx - 1))
            start = idx
            cur_len = 1
        else:
            cur_len += 1
    blocks.append((start, len(entropies) - 1))
    return blocks


def _get_layer_kwargs(layer_kwargs, device):
    cur_kwargs = {}
    for k, v in layer_kwargs.items():
        if isinstance(v, torch.Tensor):
            cur_kwargs[k] = v.to(device)
        else:
            cur_kwargs[k] = v
    return cur_kwargs


def _block_forward_collect(layer, inps, base_kwargs, reweight=False, distort=False, loss_mode="mae", w_bit=4, a_bit=8, q_config=None, vis_mask=None, cap_mask=None, rotary_emb_fn=None):
    # reuse MBQ auto_scale but restrict to this block
    # collect input features and compute scales for this layer (block-level aggregation is done outside)
    named_linears = get_named_linears(layer)

    def cache_input_hook(m, x, y, name, feat_dict):
        x = x[0]
        x = x.detach().cpu()
        feat_dict[name].append(x)

    input_feat = defaultdict(list)
    handles = []
    for name in named_linears:
        handles.append(
            named_linears[name].register_forward_hook(
                functools.partial(cache_input_hook, name=name, feat_dict=input_feat)
            )
        )
    # ensure OneVision has position_embeddings
    cur_kwargs = dict(base_kwargs)
    is_ov_layer = "onevision" in str(layer.__class__).lower()
    if is_ov_layer:
        cur_kwargs.pop("position_ids", None)
        cur_kwargs.pop("attention_mask", None)
        # regenerate rotary embeddings with explicit position ids (OneVision expects it)
        cur_kwargs.pop("position_embeddings", None)
        cur_kwargs.pop("past_key_value", None)
        cur_kwargs.pop("cache_position", None)
        cur_kwargs["use_cache"] = False
        pos_emb = None
        if inps.dim() == 3:
            bsz, seqlen = inps.shape[0], inps.shape[1]
        elif inps.dim() == 2:  # rare path: no batch dim
            bsz, seqlen = 1, inps.shape[0]
            inps = inps.unsqueeze(0)
        else:
            return inps, [], {}
        pos_ids = torch.arange(seqlen, device=inps.device).unsqueeze(0).expand(bsz, -1)
        if rotary_emb_fn is not None:
            try:
                pos_emb = rotary_emb_fn(inps, pos_ids)
            except TypeError:
                try:
                    pos_emb = rotary_emb_fn(pos_ids)
                except Exception:
                    pos_emb = None
        if pos_emb is None and hasattr(layer, "self_attn") and hasattr(layer.self_attn, "rotary_emb"):
            try:
                pos_emb = layer.self_attn.rotary_emb(inps, pos_ids)
            except TypeError:
                pos_emb = layer.self_attn.rotary_emb(pos_ids)
        if pos_emb is None and hasattr(layer, "rotary_emb"):
            try:
                pos_emb = layer.rotary_emb(inps, pos_ids)
            except TypeError:
                pos_emb = layer.rotary_emb(pos_ids)
        if pos_emb is None:
            return inps, [], {}
        cur_kwargs["position_embeddings"] = pos_emb
    out = layer(inps, **cur_kwargs)[0]
    for h in handles:
        h.remove()
    input_feat = {k: torch.cat(v, dim=0) for k, v in input_feat.items()}

    # block-level: aggregate per-layer scales and return
    # OneVision needs the regenerated position_embeddings during scale search too
    module_kwargs = cur_kwargs if is_ov_layer else base_kwargs
    solver_start = time.perf_counter()
    scales_list = auto_scale_block(
        layer,
        module_kwargs,
        w_bit=w_bit,
        q_config=q_config,
        input_feat=input_feat,
        ans_mask=cap_mask,
        vis_mask=vis_mask,
        reweight_ratio_dict={"attn": None, "mlp": None},
        loss_mode=loss_mode
    )
    solver_seconds = time.perf_counter() - solver_start
    return out, scales_list, input_feat, solver_seconds


def run_qvlm(
    model,
    prompt_inputs,
    prompt_kwargs,
    w_bit,
    a_bit,
    q_config,
    loss_mode="mae",
    wa_quant=False,
    desc="Running QVLM...",
    max_block_len=6,
    entropy_percentile=0.6,
    verbose=True,
    blocks=None,
    micro_batch=1,
    calib_batch=None,
    timing_log=None,
):
    if timing_log is None:
        timing_log = {}
    rotary_emb_fn = None
    cand_rot = [
        getattr(model, "rotary_emb", None),
        getattr(getattr(model, "model", None), "rotary_emb", None),
        getattr(getattr(model, "language_model", None), "rotary_emb", None),
        getattr(getattr(getattr(model, "model", None), "language_model", None), "rotary_emb", None),
        getattr(getattr(getattr(model, "model", None), "text_model", None), "rotary_emb", None),
    ]
    for r in cand_rot:
        if r is not None:
            rotary_emb_fn = r
            break
    # defaults: no reweight/distort; WA not supported in OneVision path
    reweight = False
    distort = False

    backbone = getattr(model, "model", model)
    layers = get_blocks(backbone)
    layer_kwargs = {}
    inps = []

    # Catcher to grab inputs/kwargs to layer0
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def __getattr__(self, name):
            if name == "module":
                return super().__getattr__(name)
            return getattr(self.module, name)

        def forward(self, inp, **kwargs):
            inps.append(inp)
            layer_kwargs.update(kwargs)
            return self.module(inp, **kwargs)

    layers[0] = Catcher(layers[0])

    inputs, vision_mask, caption_mask = process_input(prompt_inputs, prompt_kwargs)

    model.to_cuda()
    device = next(backbone.parameters()).device
    for k, v in list(inputs.items()):
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(device)

    debug_shapes = {k: tuple(v.shape) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
    print(f"[QVLM][debug] input shapes: {debug_shapes}")

    is_onevision = "onevision" in str(type(backbone)).lower()

    if is_onevision:
        image_token_id = getattr(getattr(backbone, "config", None), "image_token_id", None)
        if "labels" in inputs:
            inputs.pop("labels", None)
        pv_flat = inputs.get("pixel_values", None)
        grid = inputs.get("image_grid_thw", None)
        if pv_flat is None or grid is None:
            raise ValueError("pixel_values and image_grid_thw are required for OneVision path")
        if not isinstance(grid, torch.Tensor):
            grid = torch.tensor(grid, device=pv_flat.device)
        batch_sz = inputs["input_ids"].shape[0]
        if grid.dim() != 2 or grid.size(0) != batch_sz:
            raise ValueError(f"Unexpected image_grid_thw shape {grid.shape}")
        onevision_samples = []
        flat_idx = 0
        for b in range(batch_sz):
            need = int(grid[b].prod().item())
            if pv_flat.dim() == 2 and pv_flat.shape[-1] == 588:
                if flat_idx + need > pv_flat.shape[0]:
                    raise ValueError(f"pixel_values patches {pv_flat.shape[0]} < required {flat_idx+need}")
                pv = pv_flat[flat_idx:flat_idx+need].view(1, need, 3, 14, 14)
                flat_idx += need
            elif pv_flat.dim() == 5:
                pv = pv_flat[b:b+1]
                need = pv.shape[1]
            else:
                raise ValueError(f"Unexpected pixel_values dim {pv_flat.dim()}")

            ids0 = inputs["input_ids"][b:b+1]
            attn0 = inputs.get("attention_mask", torch.ones_like(ids0))[b:b+1]
            base_mask = ids0[0] != image_token_id
            base_ids = ids0[0][base_mask]
            base_attn = attn0[0][base_mask]
            num_img_tokens = max(1, need // 4)  # rough; will be padded
            max_full_len = num_img_tokens + int(base_mask.sum().item())
            pad_id = getattr(getattr(backbone, "processor", None), "tokenizer", None)
            pad_id = getattr(pad_id, "pad_token_id", None)
            if pad_id is None:
                pad_id = image_token_id
            img_ids = torch.full((num_img_tokens,), image_token_id, dtype=ids0.dtype, device=ids0.device)
            img_attn = torch.ones_like(img_ids, dtype=base_attn.dtype)
            merged_ids = torch.cat([img_ids, base_ids], dim=0)
            merged_attn = torch.cat([img_attn, base_attn], dim=0)
            if merged_ids.size(0) < max_full_len:
                pad_tokens = torch.full((max_full_len - merged_ids.size(0),), pad_id, dtype=ids0.dtype, device=ids0.device)
                pad_mask = torch.zeros(max_full_len - merged_ids.size(0), dtype=merged_attn.dtype, device=merged_attn.device)
                merged_ids = torch.cat([merged_ids, pad_tokens], dim=0)
                merged_attn = torch.cat([merged_attn, pad_mask], dim=0)
            sample = {
                "input_ids": merged_ids.view(1, -1),
                "attention_mask": merged_attn.view(1, -1),
                "pixel_values": pv,
                "image_grid_thw": grid[b:b+1],
            }
            onevision_samples.append(sample)

        total_samples = len(onevision_samples)
        micro_batch = 1
        inps_chunks = []
        for start in range(0, total_samples, micro_batch):
            chunk_inputs = onevision_samples[start]
            with torch.no_grad():
                model(**chunk_inputs)
            if len(inps) > 0 and len(inps) > len(inps_chunks):
                inps_chunks.append(inps[-1].detach().cpu())
        if len(inps_chunks) == 0:
            raise RuntimeError("Catcher did NOT capture any input in OneVision path.")
        max_len = max(t.shape[1] for t in inps_chunks)
        padded = []
        for t in inps_chunks:
            if t.shape[1] == max_len:
                padded.append(t)
            else:
                pad = torch.zeros(t.shape[0], max_len - t.shape[1], t.shape[2], device=t.device, dtype=t.dtype)
                padded.append(torch.cat([t, pad], dim=1))
        inps = torch.cat(padded, dim=0)
        # for downstream block loop, keep per-sample chunks to avoid full-batch peak memory
        inps = list(inps.chunk(inps.shape[0], dim=0))
    else:
        # non-OneVision: allow micro-batching to reduce peak memory (e.g., Qwen2.5VL)
        # default micro_batch=1 keeps memory low; set to batch size to revert to old behavior
        bsz = None
        for v in inputs.values():
            if isinstance(v, torch.Tensor):
                bsz = v.shape[0]
                break
        if bsz is None:
            bsz = 1
        mb = micro_batch if micro_batch is not None and micro_batch > 0 else bsz
        inps_chunks = []
        vision_chunks = [] if vision_mask is not None and isinstance(vision_mask, torch.Tensor) else None
        caption_chunks = [] if caption_mask is not None and isinstance(caption_mask, torch.Tensor) else None
        for start in range(0, bsz, mb):
            mini = {}
            end = min(start + mb, bsz)
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    mini[k] = v[start:end]
                else:
                    mini[k] = v
            with torch.no_grad():
                model(**mini)
            if len(inps) > 0 and len(inps) > len(inps_chunks):
                inps_chunks.append(inps[-1].detach().cpu())
                if vision_chunks is not None:
                    vision_chunks.append(vision_mask[start:end])
                if caption_chunks is not None:
                    caption_chunks.append(caption_mask[start:end])
        if len(inps_chunks) == 0:
            raise RuntimeError("Catcher did NOT capture any input; check get_blocks layer[0] and model forward path.")
        # keep list of chunks to avoid concatenation
        inps = inps_chunks
        if vision_chunks is not None:
            vision_mask = vision_chunks
        if caption_chunks is not None:
            caption_mask = caption_chunks
        # optional cap on calibration batch to save memory (truncate lists)
        if calib_batch is not None and calib_batch > 0:
            total = sum(chunk.shape[0] for chunk in inps)
            if total > calib_batch:
                kept = []
                kept_vis = [] if isinstance(vision_mask, list) else None
                kept_cap = [] if isinstance(caption_mask, list) else None
                remain = calib_batch
                for idx, chunk in enumerate(inps):
                    if remain <= 0:
                        break
                    take = min(remain, chunk.shape[0])
                    kept.append(chunk[:take])
                    if kept_vis is not None:
                        kept_vis.append(vision_mask[idx][:take])
                    if kept_cap is not None:
                        kept_cap.append(caption_mask[idx][:take])
                    remain -= take
                inps = kept
                if kept_vis is not None:
                    vision_mask = kept_vis
                if kept_cap is not None:
                    caption_mask = kept_cap

    if len(inps) == 0:
        raise RuntimeError("Catcher did NOT capture any input; check get_blocks layer[0] and model forward path.")

    # MBQ-style: use first sample for scaling across all models
    if isinstance(inps, list):
        first_inp = inps[0]
        inps = first_inp
        del first_inp
    if isinstance(vision_mask, list):
        vision_mask = vision_mask[0]
    if isinstance(caption_mask, list):
        caption_mask = caption_mask[0]

    model.to_cpu()
    layers[0] = layers[0].module
    # keep batch dimension for non-OneVision; if dim==2, restore batch
    if not is_onevision and isinstance(inps, torch.Tensor) and inps.dim() == 2:
        inps = inps.unsqueeze(0)

    base_kwargs = dict(layer_kwargs)
    base_kwargs["use_cache"] = False

    layers[0] = layers[0].cpu()
    move_embed(model.model, "cpu")

    gc.collect()
    torch.cuda.empty_cache()

    # entropy-based block partition
    if is_onevision:
        # OneVision: skip entropy search; layerwise blocks with micro-batching already applied
        blocks = [(i, i) for i in range(len(layers))]
        entropies = None
        if verbose:
            print(f"[QVLM] OneVision detected, using layerwise blocks: {blocks}")
    else:
        if blocks is None:
            try:
                analysis_start = time.perf_counter()
                entropies = measure_layer_entropies(model, prompt_inputs, prompt_kwargs, max_samples=16, bins=256, micro_batch=micro_batch)
                blocks = partition_blocks(entropies, max_block_len=max_block_len, percentile=entropy_percentile)
                timing_log["analysis_seconds"] = float(timing_log.get("analysis_seconds", 0.0)) + (time.perf_counter() - analysis_start)
                if verbose:
                    print(f"[QVLM] entropy (first 5): {[round(e,4) for e in entropies[:5]]} ...")
                    print(f"[QVLM] blocks: {blocks}")
            except Exception as e:
                blocks = [(i, i) for i in range(len(layers))]
                entropies = None
                if verbose:
                    print(f"[QVLM] entropy block search failed ({e}), fallback to layerwise blocks.")
        else:
            entropies = None

    # blockwise scale search: per-block loss on block output (MAE/MSE)
    mbq_results = {"scale": []}
    loss_fn = torch.nn.L1Loss() if loss_mode == "mae" else torch.nn.MSELoss()

    # Non-OneVision: follow original MBQ design (single-sample scaling)
    if not is_onevision:
        if isinstance(inps, list):
            inps = inps[0]
        if inps.dim() == 3:
            inps = inps[0]
        if isinstance(vision_mask, list):
            vision_mask = vision_mask[0]
        if isinstance(caption_mask, list):
            caption_mask = caption_mask[0]

    # OneVision: also use first sample to avoid concat/OOM
    if is_onevision and isinstance(inps, list):
        inps = inps[0]
        if isinstance(vision_mask, list):
            vision_mask = vision_mask[0]
        if isinstance(caption_mask, list):
            caption_mask = caption_mask[0]

    pbar = tqdm.tqdm(total=len(blocks), desc=desc)
    for b_idx, (s, e) in enumerate(blocks):
        block_scales = {}
        cur_inp = inps.cuda()
        if cur_inp.dim() == 2:
            cur_inp = cur_inp.unsqueeze(0)
        target = cur_inp.detach()
        for i in range(s, e + 1):
            layer = layers[i].cuda()
            cur_kwargs = _get_layer_kwargs(base_kwargs, next(layer.parameters()).device)
            is_ov_loop = "onevision" in str(model.model.__class__).lower()
            if not is_ov_loop:
                cur_kwargs.pop("position_ids", None)
                if cur_kwargs.get("position_embeddings", None) is None:
                    cur_kwargs.pop("position_embeddings", None)
            else:
                cur_kwargs.pop("position_ids", None)
                cur_kwargs.pop("attention_mask", None)
                cur_kwargs.pop("position_embeddings", None)
                cur_kwargs.pop("past_key_value", None)
                cur_kwargs.pop("cache_position", None)
                cur_kwargs["use_cache"] = False
                bsz, seqlen = cur_inp.shape[0], cur_inp.shape[1]
                pos_ids = torch.arange(seqlen, device=cur_inp.device).unsqueeze(0).expand(bsz, -1)
                if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "rotary_emb"):
                    cur_kwargs["position_embeddings"] = layer.self_attn.rotary_emb(cur_inp, pos_ids)

            block_out, scales_list, input_feat, solver_seconds = _block_forward_collect(
                layer,
                cur_inp,
                cur_kwargs,
                loss_mode=loss_mode,
                w_bit=w_bit,
                a_bit=a_bit,
                q_config=q_config,
                vis_mask=vision_mask,
                cap_mask=caption_mask,
                rotary_emb_fn=rotary_emb_fn,
            )
            timing_log["search_seconds"] = float(timing_log.get("search_seconds", 0.0)) + float(solver_seconds)
            apply_scale(layer, scales_list, input_feat_dict=input_feat)
            block_scales[append_str_prefix(get_op_name(backbone, layer), prefix=f"layers.{i}")] = scales_list
            cur_inp = block_out if block_out.dim() == 3 else block_out.unsqueeze(0)
            layers[i] = layer.cpu()
            torch.cuda.empty_cache()

        _ = loss_fn(block_out, target.to(block_out.device))
        mbq_results["scale"].append(block_scales)
        inps = cur_inp.cpu()
        pbar.update(1)
    pbar.close()

    gc.collect()
    torch.cuda.empty_cache()
    return mbq_results
