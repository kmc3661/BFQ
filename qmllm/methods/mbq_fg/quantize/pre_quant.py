import torch
import torch.nn as nn
import tqdm
import copy
import gc
import functools
import time
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from torch.nn import CrossEntropyLoss
from transformers.models.bloom.modeling_bloom import BloomForCausalLM
from transformers.models.opt.modeling_opt import OPTForCausalLM
from transformers.models.llama.modeling_llama import LlamaForCausalLM

from qmllm.utils.search import append_str_prefix, get_op_name

from qmllm.methods.mbq_fg.quantize.auto_scale_wa_distort import auto_scale_block_wa_distort
from qmllm.methods.mbq_fg.quantize.auto_scale_wa import auto_scale_block_wa
from qmllm.methods.mbq_fg.quantize.auto_scale_distort import auto_scale_block_distort
from qmllm.methods.mbq_fg.quantize.auto_scale import auto_scale_block, apply_scale
from qmllm.quantization.qlinear import WALinear
from qmllm.quantization.quant_funcs import pseudo_quantize_tensor, quantize_activation_per_token_absmax
from .quantizer import get_module_by_name_suffix


__all__ = ["run_mbq"]


class GradCacheHook:
    def __init__(self, vis_masks, cap_masks):
        if vis_masks is None or cap_masks is None:
            raise ValueError
        self.hooks = []
        if isinstance(vis_masks, (list, tuple)):
            self.vis_masks = [m.detach().cpu() if isinstance(m, torch.Tensor) else torch.as_tensor(m, dtype=torch.bool) for m in vis_masks]
        else:
            self.vis_masks = vis_masks.detach().cpu()
        if isinstance(cap_masks, (list, tuple)):
            self.cap_masks = [m.detach().cpu() if isinstance(m, torch.Tensor) else torch.as_tensor(m, dtype=torch.bool) for m in cap_masks]
        else:
            self.cap_masks = cap_masks.detach().cpu()
        self.steps = {}
        self.grad_dict = {}

    @staticmethod
    def _align_mask(mask, n: int, device: torch.device):
        mask = torch.as_tensor(mask, dtype=torch.bool, device=device).flatten()
        if mask.numel() == n:
            return mask
        if mask.numel() > n:
            return mask[:n]
        pad = torch.zeros(n - mask.numel(), dtype=torch.bool, device=device)
        return torch.cat([mask, pad], dim=0)


    def cache_grad_hook(self, module, inp, out, name):
        # initialize step counter, we use step counter to find the right mask for the grad
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
            vis_mask = self._align_mask(vis_mask, N, output_grad.device)
            cap_mask = self._align_mask(cap_mask, N, output_grad.device)

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
                # print(f"Registering hook for layer.{n}")
                self.hooks.append(
                    m.register_full_backward_hook(
                        functools.partial(self.cache_grad_hook, name=f"layers.{n}")
                    )
                )


    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


    def get_grad_dict(self):
        return self.grad_dict
    

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
        # layers = [model.model.layers, model.model.vision_tower.vision_tower.vision_model.encoder.layers]
        layers = model.model.layers
    elif cname == "LlavaQwenForCausalLM":
        layers = model.model.layers
    elif cname == "InternLM2ForCausalLM":
        layers = model.model.layers
    elif cname == "InternVLChatModel":
        layers = model.language_model.model.layers
    elif cname == "Qwen2VLForConditionalGeneration" or "qwen2" in clower:
        # Qwen2/VL variants sometimes nest decoder under language_model
        if hasattr(model, "model") and hasattr(model.model, "language_model") and hasattr(model.model.language_model, "layers"):
            layers = model.model.language_model.layers
        elif hasattr(model, "model") and hasattr(model.model, "layers"):
            layers = model.model.layers
        else:
            layers = None
    elif "llavaonevision" in clower:
        # try multiple fallbacks to find decoder layers
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
            # fallback: scan modules for Qwen2 decoder layers
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
    if isinstance(model, LlamaForCausalLM):
        model.model.embed_tokens = model.model.embed_tokens.to(device)
        model.model.rotary_emb = model.model.rotary_emb.to(device)
    elif isinstance(model, OPTForCausalLM):
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(device)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(
            device
        )
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
        # model.model.rotary_emb = model.model.rotary_emb.to(device)
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
        if hasattr(model.model, "embed_tokens"):
            model.model.embed_tokens = model.model.embed_tokens.to(device)
        if hasattr(model.model, "language_model") and hasattr(model.model.language_model, "embed_tokens"):
            model.model.language_model.embed_tokens = model.model.language_model.embed_tokens.to(device)
    elif "qwen2" in str(model.__class__).lower() and hasattr(model, "model"):
        # fallback for Qwen2/Qwen2.5 VL-like wrappers
        if hasattr(model.model, "language_model") and hasattr(model.model.language_model, "embed_tokens"):
            model.model.language_model.embed_tokens = model.model.language_model.embed_tokens.to(device)
    elif model.__class__.__name__ == "LlavaLlamaModel":
        model.llm.model.embed_tokens = model.llm.model.embed_tokens.to(device)
    else:
        raise NotImplementedError(type(model))


def process_input(prompt_inputs, prompt_kwargs):
    inputs = {**prompt_inputs, **prompt_kwargs}
    inputs["use_cache"] = False
    vision_mask = inputs.pop("vision_mask", None)
    caption_mask = inputs.pop("caption_mask", None)
    inputs.pop("sample_ids", None)
    
    return inputs, vision_mask, caption_mask


def _parse_layer_groups(layer_groups: str, num_layers: int) -> List[Tuple[str, List[int]]]:
    if not layer_groups:
        return [("all", list(range(num_layers)))]
    groups: List[Tuple[str, List[int]]] = []
    for raw in str(layer_groups).split(","):
        item = raw.strip()
        if not item:
            continue
        if "-" in item:
            s, e = item.split("-", 1)
            s_i = max(0, int(s))
            e_i = min(num_layers - 1, int(e))
            if e_i < s_i:
                continue
            idx = list(range(s_i, e_i + 1))
            groups.append((f"{s_i}-{e_i}", idx))
        else:
            i = int(item)
            if 0 <= i < num_layers:
                groups.append((str(i), [i]))
    if not groups:
        groups = [("all", list(range(num_layers)))]
    return groups


def _align_mask_2d(mask: torch.Tensor, bsz: int, seqlen: int, device: torch.device) -> torch.Tensor:
    if mask is None:
        return torch.zeros((bsz, seqlen), dtype=torch.bool, device=device)
    m = mask.to(device=device, dtype=torch.bool)
    if m.dim() == 1:
        m = m.unsqueeze(0)
    if m.shape[0] > bsz:
        m = m[:bsz]
    elif m.shape[0] < bsz:
        pad_b = torch.zeros((bsz - m.shape[0], m.shape[1]), dtype=torch.bool, device=device)
        m = torch.cat([m, pad_b], dim=0)
    if m.shape[1] > seqlen:
        m = m[:, :seqlen]
    elif m.shape[1] < seqlen:
        pad_s = torch.zeros((m.shape[0], seqlen - m.shape[1]), dtype=torch.bool, device=device)
        m = torch.cat([m, pad_s], dim=1)
    return m


def _extract_logits(outputs):
    if outputs is None:
        return None
    logits = getattr(outputs, "logits", None)
    if torch.is_tensor(logits):
        return logits
    if isinstance(outputs, (tuple, list)):
        for item in outputs:
            if torch.is_tensor(item) and item.dim() == 3:
                return item
    return None


def _mean_token_kl(ref_logits: torch.Tensor, cur_logits: torch.Tensor, token_mask: Optional[torch.Tensor]) -> float:
    if (not torch.is_tensor(ref_logits)) or (not torch.is_tensor(cur_logits)):
        return 0.0
    bsz = min(ref_logits.shape[0], cur_logits.shape[0])
    seqlen = min(ref_logits.shape[1], cur_logits.shape[1])
    if bsz <= 0 or seqlen <= 0:
        return 0.0
    ref = ref_logits[:bsz, :seqlen].float()
    cur = cur_logits[:bsz, :seqlen].float()
    logp = torch.log_softmax(ref, dim=-1)
    p = logp.exp()
    logq = torch.log_softmax(cur, dim=-1)
    kl = (p * (logp - logq)).sum(dim=-1)
    if torch.is_tensor(token_mask):
        m = token_mask.to(device=kl.device, dtype=torch.bool)
        if m.dim() == 1:
            m = m.unsqueeze(0)
        m = _align_mask_2d(m, bsz, seqlen, kl.device)
        denom = m.sum().clamp_min(1)
        return float((kl * m.to(kl.dtype)).sum().item() / float(denom.item()))
    return float(kl.mean().item())


def _apply_token_quant_noise(x: torch.Tensor, token_mask: torch.Tensor, n_bits: int, strength: float) -> torch.Tensor:
    if (not torch.is_tensor(token_mask)) or token_mask.numel() == 0:
        return x
    m = token_mask.to(device=x.device, dtype=torch.bool)
    if m.dim() == 1:
        m = m.unsqueeze(0)
    m = _align_mask_2d(m, x.shape[0], x.shape[1], x.device)
    if not m.any():
        return x
    # Match analysis-time noise: per-token min-max fake quantization
    qmin = -(2 ** (int(n_bits) - 1))
    qmax = (2 ** (int(n_bits) - 1)) - 1
    x_min = x.amin(dim=-1, keepdim=True)
    x_max = x.amax(dim=-1, keepdim=True)
    scale = (x_max - x_min) / max(float(qmax - qmin), 1.0)
    scale = torch.clamp(scale, min=1e-8)
    zero = qmin - x_min / scale
    q = torch.round(x / scale + zero).clamp(qmin, qmax)
    qx = (q - zero) * scale
    noised = x + float(strength) * (qx - x)
    return torch.where(m.unsqueeze(-1), noised, x)


def _make_layer_noise_hook(token_mask: torch.Tensor, n_bits: int, strength: float):
    def _hook(_module, _inp, out):
        if isinstance(out, tuple):
            hidden = out[0]
            hidden = _apply_token_quant_noise(hidden, token_mask=token_mask, n_bits=n_bits, strength=strength)
            return (hidden,) + out[1:]
        return _apply_token_quant_noise(out, token_mask=token_mask, n_bits=n_bits, strength=strength)

    return _hook


def _make_linear_input_noise_pre_hook(token_mask: torch.Tensor, n_bits: int, strength: float):
    def _pre_hook(_module, inp):
        if not isinstance(inp, tuple) or len(inp) == 0:
            return inp
        x = inp[0]
        if not torch.is_tensor(x):
            return inp
        # Expected shape is [B, S, C] for decoder linear inputs.
        if x.dim() == 3:
            x_noisy = _apply_token_quant_noise(x, token_mask=token_mask, n_bits=n_bits, strength=strength)
            return (x_noisy,) + inp[1:]
        # Best-effort fallback for [S, C] (single-sample).
        if x.dim() == 2 and torch.is_tensor(token_mask):
            tm = token_mask
            if tm.dim() == 2 and tm.shape[0] == 1 and tm.shape[1] == x.shape[0]:
                x_noisy = _apply_token_quant_noise(x.unsqueeze(0), token_mask=tm, n_bits=n_bits, strength=strength).squeeze(0)
                return (x_noisy,) + inp[1:]
        return inp

    return _pre_hook


def _compute_text_received_score(attn_qt: torch.Tensor, q_pos: torch.Tensor, k_pos: torch.Tensor, mode: str) -> torch.Tensor:
    causal_visible = (q_pos[:, None] >= k_pos[None, :]).to(attn_qt.dtype)
    received = (attn_qt * causal_visible).sum(dim=0)
    exposure = causal_visible.sum(dim=0).clamp_min(1.0)
    received_exp = received / exposure
    if mode == "exposure":
        return received_exp
    visible_targets_per_query = causal_visible.sum(dim=1, keepdim=True).clamp_min(1.0)
    baseline = (causal_visible / visible_targets_per_query).sum(dim=0)
    baseline_exp = baseline / exposure
    eps = 1e-8
    if mode in ("baseline_ratio", "exposure_baseline_ratio"):
        return received / (baseline + eps)
    if mode == "baseline_diff":
        return received - baseline
    if mode == "exposure_baseline_diff":
        return received_exp - baseline_exp
    return attn_qt.mean(dim=0)


def _compute_attention_token_weight(
    layer_attn: torch.Tensor,
    vision_mask: torch.Tensor,
    text_mask: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    text_correction: str = "exposure_baseline_ratio",
    vision_query_source: str = "text",
    query_mix_alpha: float = 0.5,
    eps: float = 1e-6,
) -> torch.Tensor:
    if not torch.is_tensor(layer_attn):
        return None
    bsz, _h, seqlen, _ = layer_attn.shape
    device = layer_attn.device
    vm = _align_mask_2d(vision_mask, bsz, seqlen, device)
    tm = _align_mask_2d(text_mask, bsz, seqlen, device)
    if torch.is_tensor(valid_mask):
        val = _align_mask_2d(valid_mask, bsz, seqlen, device)
        vm = vm & val
        tm = tm & val

    score = torch.zeros((bsz, seqlen), dtype=torch.float32, device=device)
    for b in range(bsz):
        v_idx = vm[b].nonzero(as_tuple=False).squeeze(-1)
        t_idx = tm[b].nonzero(as_tuple=False).squeeze(-1)
        if v_idx.numel() == 0 and t_idx.numel() == 0:
            continue
        if v_idx.numel() > 0 and t_idx.numel() > 0:
            attn_tv = layer_attn[b, :, t_idx][:, :, v_idx].mean(dim=0).float()  # [q_t, v]
            v_score = attn_tv.mean(dim=0)
            if vision_query_source == "both":
                attn_vv = layer_attn[b, :, v_idx][:, :, v_idx].mean(dim=0).float()
                vv_score = attn_vv.mean(dim=0)
                alpha = float(query_mix_alpha)
                v_score = alpha * v_score + (1.0 - alpha) * vv_score
            score[b, v_idx] = v_score
        if t_idx.numel() > 0:
            attn_tt = layer_attn[b, :, t_idx][:, :, t_idx].mean(dim=0).float()  # [q_t, t]
            q_pos = t_idx.to(attn_tt.device)
            k_pos = t_idx.to(attn_tt.device)
            t_score = _compute_text_received_score(attn_tt, q_pos=q_pos, k_pos=k_pos, mode=text_correction)
            score[b, t_idx] = score[b, t_idx] + t_score

    score = torch.relu(score)
    denom = score.sum(dim=1, keepdim=True)
    zero_row = denom <= eps
    if zero_row.any():
        fallback = (vm | tm).to(score.dtype)
        fb_denom = fallback.sum(dim=1, keepdim=True).clamp_min(1.0)
        fallback = fallback / fb_denom
        score = torch.where(zero_row.expand_as(score), fallback, score)
        denom = score.sum(dim=1, keepdim=True)
    score = score / (denom + eps)
    return score.detach()


def _build_logit_sensitivity_weights(
    model,
    layers,
    sample_inputs: List[Dict[str, torch.Tensor]],
    sample_vis_masks: List[torch.Tensor],
    sample_text_masks: List[torch.Tensor],
    layer_groups: List[Tuple[str, List[int]]],
    noise_bits: int = 8,
    noise_strength: float = 1.0,
    stage1_metric: str = "full_logit",
) -> Dict[str, Dict[str, float]]:
    if len(sample_inputs) == 0:
        return {}
    with torch.no_grad():
        baseline: List[Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]] = []
        answer_mask_samples = 0
        pbar_base = tqdm.tqdm(total=len(sample_inputs), desc="[LAGQ] Stage-1 baseline logits")
        for s_inp in sample_inputs:
            ref_out = model(**s_inp, use_cache=False, return_dict=True)
            ref_logits = _extract_logits(ref_out)
            if ref_logits is None:
                raise RuntimeError("Failed to extract logits for sensitivity baseline.")
            attn = s_inp.get("attention_mask", None)
            labels = s_inp.get("labels", None)
            ans_mask = None
            if torch.is_tensor(labels):
                ans_mask = (labels != -100)
                if torch.is_tensor(attn):
                    ans_mask = ans_mask & attn.to(dtype=torch.bool, device=ans_mask.device)
                if bool(ans_mask.any().item()):
                    answer_mask_samples += 1
            baseline.append(
                (
                    ref_logits.detach(),
                    attn.detach() if torch.is_tensor(attn) else None,
                    ans_mask.detach() if torch.is_tensor(ans_mask) else None,
                )
            )
            del ref_out
            pbar_base.update(1)
        pbar_base.close()
        if stage1_metric == "answer_only" and answer_mask_samples == 0:
            print("[LAGQ][warn] stage1_metric=answer_only but no supervised answer tokens found; fallback to full_logit behavior.")

        raw: Dict[str, Dict[str, float]] = {g: {"vision": 0.0, "text": 0.0} for g, _ in layer_groups}
        target_linear_names = {
            # attention
            "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
            "attention.wq", "attention.wk", "attention.wv", "attention.wo",
            # mlp
            "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
            "feed_forward.w1", "feed_forward.w2", "feed_forward.w3",
            "mlp.w1", "mlp.w2", "mlp.w3",
        }
        tasks: List[Tuple[str, List[int], str]] = []
        for g_name, g_lids in layer_groups:
            for mod in ("vision", "text"):
                tasks.append((g_name, g_lids, mod))

        pbar = tqdm.tqdm(total=len(tasks), desc="[LAGQ] Stage-1 layer/modality sensitivity")
        for g_name, g_lids, mod in tasks:
            valid_lids = [lid for lid in g_lids if 0 <= lid < len(layers)]
            if not valid_lids:
                pbar.update(1)
                continue
            kls: List[float] = []
            for s_idx, s_inp in enumerate(sample_inputs):
                m = sample_vis_masks[s_idx] if mod == "vision" else sample_text_masks[s_idx]
                if (not torch.is_tensor(m)) or (not m.any()):
                    continue
                handles = []
                try:
                    for lid in valid_lids:
                        layer = layers[lid]
                        named_linears = get_named_linears(layer)
                        for lname, lmod in named_linears.items():
                            # Inject at actual linear input points used by activation quantization.
                            if lname in target_linear_names:
                                h = lmod.register_forward_pre_hook(
                                    _make_linear_input_noise_pre_hook(
                                        token_mask=m,
                                        n_bits=int(noise_bits),
                                        strength=float(noise_strength),
                                    )
                                )
                                handles.append(h)
                    if len(handles) == 0:
                        # Fallback: if model naming differs, use layer-output hook as best effort.
                        for lid in valid_lids:
                            h = layers[lid].register_forward_hook(
                                _make_layer_noise_hook(
                                    token_mask=m,
                                    n_bits=int(noise_bits),
                                    strength=float(noise_strength),
                                )
                            )
                            handles.append(h)
                    cur_out = model(**s_inp, use_cache=False, return_dict=True)
                    cur_logits = _extract_logits(cur_out)
                    ref_logits, ref_attn, ref_ans = baseline[s_idx]
                    if stage1_metric == "answer_only" and torch.is_tensor(ref_ans) and bool(ref_ans.any().item()):
                        eval_mask = ref_ans
                    else:
                        eval_mask = ref_attn
                    kl_mean = _mean_token_kl(ref_logits, cur_logits, eval_mask)

                    # Token-count normalization:
                    # convert mean KL over valid sequence tokens -> total KL,
                    # then divide by number of perturbed tokens for fair modality comparison.
                    if torch.is_tensor(ref_logits) and torch.is_tensor(cur_logits):
                        bsz = min(ref_logits.shape[0], cur_logits.shape[0])
                        seqlen = min(ref_logits.shape[1], cur_logits.shape[1])
                        if bsz > 0 and seqlen > 0:
                            if torch.is_tensor(eval_mask):
                                valid = eval_mask.to(dtype=torch.bool, device=m.device)
                                if valid.dim() == 1:
                                    valid = valid.unsqueeze(0)
                                valid = _align_mask_2d(valid, bsz, seqlen, m.device)
                                valid_count = int(valid.sum().item())
                            else:
                                valid_count = int(bsz * seqlen)
                            pert_mask = _align_mask_2d(m.to(dtype=torch.bool, device=m.device), bsz, seqlen, m.device)
                            if torch.is_tensor(ref_attn):
                                pert_valid = ref_attn.to(dtype=torch.bool, device=m.device)
                                if pert_valid.dim() == 1:
                                    pert_valid = pert_valid.unsqueeze(0)
                                pert_valid = _align_mask_2d(pert_valid, bsz, seqlen, m.device)
                                pert_count = int((pert_mask & pert_valid).sum().item())
                            else:
                                pert_count = int(pert_mask.sum().item())
                            if pert_count > 0 and valid_count > 0:
                                kl_norm = float(kl_mean) * (float(valid_count) / float(pert_count))
                            else:
                                kl_norm = 0.0
                        else:
                            kl_norm = 0.0
                    else:
                        kl_norm = 0.0
                    kls.append(kl_norm)
                    del cur_out
                finally:
                    for h in handles:
                        try:
                            h.remove()
                        except Exception:
                            pass
            raw[g_name][mod] = float(np.mean(kls)) if len(kls) > 0 else 0.0
            pbar.update(1)
            pbar.set_postfix_str(f"group={g_name} mod={mod}")
        pbar.close()

        all_vals = [max(v, 0.0) for g in raw.values() for v in g.values()]
        pos_vals = [v for v in all_vals if v > 0]
        mean_pos = float(np.mean(pos_vals)) if len(pos_vals) > 0 else 1.0
        out: Dict[str, Dict[str, float]] = {}
        for g_name in raw:
            out[g_name] = {}
            for mod in ("vision", "text"):
                w = max(raw[g_name][mod], 0.0) / max(mean_pos, 1e-8)
                w = max(0.25, min(4.0, float(w)))
                out[g_name][mod] = w
        return out


def _quantize_input_per_token(x: torch.Tensor, a_bit: int, q_config: dict):
    _ = q_config  # keep signature for future extension
    return quantize_activation_per_token_absmax(x, n_bits=a_bit)


def _build_quantized_layer_for_qig(layer, w_bit, a_bit, q_config, wa_quant):
    layer_q = copy.deepcopy(layer).eval()
    named_linears_q = get_named_linears(layer_q)
    if wa_quant:
        for n, m in named_linears_q.items():
            new_linear = WALinear.from_float(
                m,
                weight_quant="per_channel",
                act_quant="per_token",
                w_bit=w_bit,
                a_bit=a_bit,
            )
            father_module = get_module_by_name_suffix(layer_q, ".".join(n.split(".")[:-1]))
            setattr(father_module, n.split(".")[-1], new_linear)
            del new_linear, m
    else:
        for _n, m in named_linears_q.items():
            m.weight.data = pseudo_quantize_tensor(m.weight.data, n_bits=w_bit, **q_config)
    return layer_q


def _compute_qig_token_weight(
    layer,
    layer_input: torch.Tensor,
    layer_kwargs: dict,
    w_bit: int,
    a_bit: int,
    q_config: dict,
    wa_quant: bool,
    qig_steps: int = 8,
    qig_iqr_factor: float = 1.5,
    qig_eps: float = 1e-6,
    qig_disable_iqr: bool = False,
    qig_use_abs: bool = True,
):
    x_fp = layer_input.detach()
    if x_fp.dim() == 2:
        x_fp = x_fp.unsqueeze(0)
    if wa_quant:
        x_ref = _quantize_input_per_token(x_fp, a_bit=a_bit, q_config=q_config)
    else:
        x_ref = torch.zeros_like(x_fp)

    layer_q = _build_quantized_layer_for_qig(layer, w_bit, a_bit, q_config, wa_quant).to(x_fp.device)

    delta = (x_fp - x_ref).detach()
    ig_acc = torch.zeros_like(x_fp, dtype=torch.float32)
    steps = max(1, int(qig_steps))

    for s in range(steps):
        alpha = float(s + 1) / float(steps)
        x_alpha = (x_ref + alpha * delta).detach().requires_grad_(True)
        with torch.enable_grad():
            out_fp = layer(x_alpha, **layer_kwargs)
            out_q = layer_q(x_alpha, **layer_kwargs)
            if isinstance(out_fp, tuple):
                out_fp = out_fp[0]
            if isinstance(out_q, tuple):
                out_q = out_q[0]
            # Token-wise discrepancy objective: sum over token losses.
            token_err = (out_fp - out_q).float().pow(2).mean(dim=-1)
            objective = token_err.sum()
            grad = torch.autograd.grad(objective, x_alpha, retain_graph=False, create_graph=False)[0]
            ig_acc += grad.detach().float()

    del layer_q
    torch.cuda.empty_cache()

    attr = (delta.float() * (ig_acc / float(steps)))
    if qig_use_abs:
        attr = attr.abs()
    token_score = attr.mean(dim=-1)

    if not qig_disable_iqr:
        q1 = torch.quantile(token_score, 0.25, dim=1, keepdim=True)
        q3 = torch.quantile(token_score, 0.75, dim=1, keepdim=True)
        iqr = q3 - q1
        lower = q1 - qig_iqr_factor * iqr
        upper = q3 + qig_iqr_factor * iqr
        token_score = torch.clamp(token_score, min=lower, max=upper)

    token_score = torch.clamp(token_score, min=0.0)
    denom = token_score.sum(dim=1, keepdim=True)
    zero_row = denom <= qig_eps
    if zero_row.any():
        token_score = token_score.clone()
        token_score[zero_row.expand_as(token_score)] = 1.0
        denom = token_score.sum(dim=1, keepdim=True)
    token_weight = token_score / (denom + qig_eps)
    return token_weight.detach()


def _force_attention_backend_for_token_weight(model):
    backbone = getattr(model, "model", model)

    candidates = [
        backbone,
        getattr(backbone, "model", None),
        getattr(backbone, "language_model", None),
        getattr(getattr(backbone, "model", None), "language_model", None),
    ]

    updated = []

    def _set_eager(obj):
        if obj is None or id(obj) in updated:
            return
        cfg = getattr(obj, "config", None)
        if cfg is not None and hasattr(cfg, "_attn_implementation"):
            cfg._attn_implementation = "eager"
            updated.append(id(obj))

    for cand in candidates:
        _set_eager(cand)

    try:
        layers = get_blocks(backbone)
    except Exception:
        layers = []

    for layer in layers:
        self_attn = getattr(layer, "self_attn", None)
        if self_attn is None:
            continue
        _set_eager(self_attn)
        cfg = getattr(self_attn, "config", None)
        if cfg is not None and hasattr(cfg, "_attn_implementation"):
            cfg._attn_implementation = "eager"

    if updated:
        print("[MBQ][attention] forced attention backend to eager for attention-token weighting")


@torch.no_grad()
def run_mbq(
    model,
    prompt_inputs,
    prompt_kwargs,
    w_bit,
    a_bit,
    q_config,
    auto_scale=True,
    loss_mode="mae",
    wa_quant=False,
    reweight=False,
    distort=False,
    finegrained=False,
    qig_steps=8,
    qig_iqr_factor=1.5,
    qig_eps=1e-6,
    qig_disable_iqr=False,
    qig_use_abs=True,
    finegrained_mode="qig",  # ["qig", "attention"]
    lagq_layer_groups="0-10,11-25,26-35",
    lagq_noise_bits=8,
    lagq_noise_strength=1.0,
    lagq_micro_batch_size=0,
    lagq_stage1_samples=0,
    lagq_disable_logit_sensitivity=False,
    lagq_disable_attention_token_weight=False,
    lagq_stage1_metric="full_logit",
    lagq_text_attn_correction="exposure_baseline_ratio",
    lagq_vision_query_source="both",
    lagq_query_mix_alpha=0.5,
    bfq_policy=None,
    desc="Running MBQ...",
    blocks=None,  # optional list of (start,end) layer indices to process; default is all
    timing_log=None,
):
    if timing_log is None:
        timing_log = {}

    if "bigcode" in str(model.model.__class__).lower():
        # otherwise attention_mask will always be on cpu.
        model.transformer.bias = model.transformer.bias.to("cuda")

    if finegrained and finegrained_mode == "attention" and (not lagq_disable_attention_token_weight):
        _force_attention_backend_for_token_weight(model)

    layers = get_blocks(model.model)

    inps = []
    layer_kwargs = {}

    layers[0] = layers[0].cuda()
    move_embed(model.model, "cuda")

    # get input and kwargs to layer 0
    # with_kwargs is only supported in PyTorch 2.0
    # use this Catcher hack for now
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def __getattr__(self, name):
            # Proxy unknown attrs to wrapped module (e.g., attention_type)
            if name == "module":
                return super().__getattr__(name)
            return getattr(self.module, name)

        def forward(self, inp, **kwargs):
            inps.append(inp)
            if len(layer_kwargs) == 0:  # keep kwargs from the first forward (OneVision uses micro-batches)
                layer_kwargs.update(kwargs)
            return self.module(inp, **kwargs)

    # patch layer 0 to catch input and kwargs
    layers[0] = Catcher(layers[0])

    inputs, vision_mask, caption_mask = process_input(prompt_inputs, prompt_kwargs)
    capture_inputs = inputs

    # Move model to CUDA early to keep vision encoders aligned with inputs
    model.to_cuda()
    backbone = getattr(model, "model", model)
    device = next(backbone.parameters()).device
    for k, v in list(inputs.items()):
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(device)

    # Debug: log input tensor shapes (helps catch lost batch dims)
    debug_shapes = {k: tuple(v.shape) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
    print(f"[MBQ][debug] input shapes: {debug_shapes}")

    is_onevision = "onevision" in str(type(backbone)).lower()
    rotary_emb_fn = None
    if is_onevision:
        # locate the shared text rotary embedding module
        candidates = [
            backbone,
            getattr(backbone, "model", None),
            getattr(backbone, "language_model", None),
            getattr(getattr(backbone, "model", None), "language_model", None),
        ]
        for cand in candidates:
            if cand is not None and hasattr(cand, "rotary_emb"):
                rotary_emb_fn = cand.rotary_emb
                break

    # LLaVA-OneVision only: 샘플별로 독립 처리하도록 분리 준비
    onevision_samples = None
    onevision_reweight_samples = None
    onevision_vis_masks = None
    onevision_cap_masks = None
    onevision_vis_mask_batch = None
    onevision_cap_mask_batch = None
    onevision_token_valid_mask = None
    token_vis_mask_batch = None
    token_text_mask_batch = None
    if is_onevision:
        capture_inputs = dict(inputs)
        image_token_id = getattr(getattr(backbone, "config", None), "image_token_id", None)
        # Do not drop labels from `inputs` globally; reweight path needs them.
        capture_inputs.pop("labels", None)
        pv_flat = capture_inputs.get("pixel_values", None)
        grid = capture_inputs.get("image_grid_thw", None)
        if pv_flat is None or grid is None:
            raise ValueError("pixel_values and image_grid_thw are required for OneVision path")
        if not isinstance(grid, torch.Tensor):
            grid = torch.tensor(grid, device=pv_flat.device)
        batch_sz = capture_inputs["input_ids"].shape[0]
        if grid.dim() != 2 or grid.size(0) != batch_sz:
            raise ValueError(f"Unexpected image_grid_thw shape {grid.shape}")
        merge_size = getattr(getattr(backbone, "visual", None), "spatial_merge_size", 2)
        onevision_samples = []
        onevision_reweight_samples = []
        onevision_vis_masks = []
        onevision_cap_masks = []
        labels_all = inputs.get("labels", None)
        flat_idx = 0
        for b in range(batch_sz):
            need = int(grid[b].prod().item())
            # per-sample pixel patches
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

            expected_seq = max(1, need // (merge_size * merge_size))

            # 실제 비전 인코더 출력 길이를 사용해 이미지 토큰 수 결정 (그리드/merge 추정보다 확실한 값)
            def _get_vis_len(out):
                if out is None:
                    return None
                tensor = out.last_hidden_state if hasattr(out, "last_hidden_state") else out
                if not isinstance(tensor, torch.Tensor):
                    return None
                if tensor.dim() == 3:
                    vis_len = tensor.shape[1]
                elif tensor.dim() == 2:
                    vis_len = tensor.shape[0]
                else:
                    return None
                print(f"[MBQ][onevision][debug] vis_out shape={tuple(tensor.shape)} vis_len={vis_len} (expected ~{expected_seq})")
                return vis_len

            with torch.no_grad():
                vis_out = backbone.visual(pv, grid_thw=grid[b:b+1]) if hasattr(backbone, "visual") else None
            vis_len = _get_vis_len(vis_out)
            if vis_len is None:
                vis_len = max(1, need // (merge_size * merge_size))
            num_img_tokens = vis_len
            ids0 = capture_inputs["input_ids"][b:b+1]
            attn0 = capture_inputs.get("attention_mask", torch.ones_like(ids0))[b:b+1]
            base_mask = ids0[0] != image_token_id
            base_ids = ids0[0][base_mask]
            base_attn = attn0[0][base_mask]
            max_full_len = num_img_tokens + int(base_mask.sum().item())
            pad_id = getattr(getattr(model, "processor", None), "tokenizer", None)
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
                "pixel_values": pv,              # keep per-sample tensor, no concat across batch
                "image_grid_thw": grid[b:b+1],   # grid aligned to this sample
            }
            onevision_samples.append(sample)
            # Reweight path needs labels and modality masks aligned with transformed sequence.
            if isinstance(labels_all, torch.Tensor):
                lab0 = labels_all[b]
                base_lab = lab0[base_mask]
                img_lab = torch.full((num_img_tokens,), -100, dtype=lab0.dtype, device=lab0.device)
                merged_lab = torch.cat([img_lab, base_lab], dim=0)
            else:
                merged_lab = torch.full((merged_ids.shape[0],), -100, dtype=torch.long, device=merged_ids.device)
            re_sample = {
                "input_ids": merged_ids.view(1, -1),
                "attention_mask": merged_attn.view(1, -1),
                "labels": merged_lab.view(1, -1),
                "pixel_values": pv,
                "image_grid_thw": grid[b:b+1],
            }
            onevision_reweight_samples.append(re_sample)
            vis_m = torch.zeros_like(merged_ids, dtype=torch.bool)
            vis_m[:num_img_tokens] = True
            cap_m = merged_lab != -100
            onevision_vis_masks.append(vis_m)
            onevision_cap_masks.append(cap_m)
            print(f"[MBQ][onevision][debug] sample {b} tokens={num_img_tokens} grid={grid[b].tolist()} pv_shape={tuple(pv.shape)}")
        if len(onevision_vis_masks) > 0:
            max_tok = max(int(m.numel()) for m in onevision_vis_masks)
            onevision_vis_mask_batch = torch.zeros((len(onevision_vis_masks), max_tok), dtype=torch.bool, device=onevision_vis_masks[0].device)
            onevision_cap_mask_batch = torch.zeros((len(onevision_cap_masks), max_tok), dtype=torch.bool, device=onevision_cap_masks[0].device)
            for idx, (vm, cm) in enumerate(zip(onevision_vis_masks, onevision_cap_masks)):
                n = int(vm.numel())
                onevision_vis_mask_batch[idx, :n] = vm
                onevision_cap_mask_batch[idx, :n] = cm

    if is_onevision:
        # micro-batch forward
        total_samples = len(onevision_samples)
        micro_batch = 1  # reduce peak memory
        inps_chunks = []
        # one-time debug hook on visual encoder to log shapes
        ov_handle = None
        if hasattr(backbone, "visual"):
            def _ov_hook(mod, inp, out):
                pv_shape = tuple(inp[0].shape) if isinstance(inp, (list, tuple)) and len(inp) > 0 else None
                out_tensor = out
                if not isinstance(out_tensor, torch.Tensor) and hasattr(out_tensor, "last_hidden_state"):
                    out_tensor = out_tensor.last_hidden_state
                out_shape = tuple(out_tensor.shape) if isinstance(out_tensor, torch.Tensor) else None
                print(f"[MBQ][onevision][visual_hook] pixel_values={pv_shape} out={out_shape}")
            ov_handle = backbone.visual.register_forward_hook(_ov_hook)
        for start in range(0, total_samples, micro_batch):
            end = min(start + micro_batch, total_samples)
            # 각 샘플 딕셔너리를 그대로 사용
            chunk_inputs = onevision_samples[start]
            # OneVision forward 직전 토큰/비전 피처 정합 상태 추가 로그
            if image_token_id is not None and "input_ids" in chunk_inputs:
                chk_tokens = (chunk_inputs["input_ids"] == image_token_id).sum(dim=1).tolist()
                pv_shape = tuple(chunk_inputs["pixel_values"].shape) if "pixel_values" in chunk_inputs else None
                print(f"[MBQ][onevision][debug] chunk tokens_per_sample={chk_tokens} pv_shape={pv_shape}")
            model(**chunk_inputs)
            if len(inps) > 0 and len(inps) > len(inps_chunks):
                inps_chunks.append(inps[-1])
        if ov_handle is not None:
            ov_handle.remove()
        if len(inps_chunks) > 0:
            # pad to max seq len across chunks (attention_mask is dropped later for OneVision)
            max_len = max(t.shape[1] for t in inps_chunks)
            padded = []
            valid = []
            for t in inps_chunks:
                cur_len = t.shape[1]
                vmask = torch.zeros(1, max_len, dtype=torch.bool, device=t.device)
                vmask[:, :cur_len] = True
                valid.append(vmask)
                if t.shape[1] == max_len:
                    padded.append(t)
                else:
                    pad = torch.zeros(t.shape[0], max_len - t.shape[1], t.shape[2], device=t.device, dtype=t.dtype)
                    padded.append(torch.cat([t, pad], dim=1))
            inps = torch.cat(padded, dim=0)
            onevision_token_valid_mask = torch.cat(valid, dim=0)
            print(f"[MBQ][onevision][debug] captured={len(inps_chunks)} padded_inps_shape={tuple(inps.shape)} max_len={max_len}")
            token_vis_mask_batch = onevision_vis_mask_batch
            if token_vis_mask_batch is not None:
                token_vis_mask_batch = _align_mask_2d(token_vis_mask_batch, inps.shape[0], inps.shape[1], inps.device)
                token_text_mask_batch = (~token_vis_mask_batch)
                if torch.is_tensor(onevision_token_valid_mask):
                    token_text_mask_batch = token_text_mask_batch & _align_mask_2d(onevision_token_valid_mask, inps.shape[0], inps.shape[1], inps.device)
        else:
            raise RuntimeError("OneVision: no inputs captured from catcher.")
    else:
        # non-OneVision: optionally micro-batch the initial catcher forward to avoid
        # attention/logit OOM while keeping the full calibration sample count.
        mb = int(lagq_micro_batch_size) if lagq_micro_batch_size is not None else 0
        if mb > 0:
            bsz_full = None
            for v in inputs.values():
                if isinstance(v, torch.Tensor) and v.dim() >= 1:
                    bsz_full = int(v.shape[0])
                    break
            if bsz_full is None:
                model(**inputs)
            else:
                for start in range(0, bsz_full, mb):
                    end = min(start + mb, bsz_full)
                    chunk_inputs = {}
                    for k, v in inputs.items():
                        if isinstance(v, torch.Tensor) and v.dim() >= 1 and int(v.shape[0]) == bsz_full:
                            chunk_inputs[k] = v[start:end]
                        else:
                            chunk_inputs[k] = v
                    model(**chunk_inputs)
        else:
            model(**inputs)

    if len(inps) == 0:
        raise RuntimeError("Catcher did NOT capture any input; check get_blocks layer[0] and model forward path.")

    model.to_cpu()
    layers[0] = layers[0].module  # restore
    if not is_onevision:
        inps = inps[0]  # legacy single-sample path for non-OneVision models
        bsz, seqlen = inps.shape[0], inps.shape[1]
        if torch.is_tensor(vision_mask):
            token_vis_mask_batch = _align_mask_2d(vision_mask, bsz, seqlen, inps.device)
        else:
            token_vis_mask_batch = torch.zeros((bsz, seqlen), dtype=torch.bool, device=inps.device)
        attn_1d = inputs.get("attention_mask", None)
        if torch.is_tensor(attn_1d):
            valid_mask = _align_mask_2d(attn_1d, bsz, seqlen, inps.device)
        else:
            valid_mask = torch.ones((bsz, seqlen), dtype=torch.bool, device=inps.device)
        token_text_mask_batch = (~token_vis_mask_batch) & valid_mask
    # Reset kwargs
    base_kwargs = dict(layer_kwargs)
    base_kwargs["use_cache"] = False

    layers[0] = layers[0].cpu()
    move_embed(model.model, "cpu")

    gc.collect()
    torch.cuda.empty_cache()

    mbq_results = {
        "scale": [],
    }
    lagq_sensitivity_weights = None
    layer_to_group: Dict[int, str] = {}


    if finegrained and reweight:
        print("[MBQ][warn] finegrained_qig and reweight are both enabled; disabling reweight for Xiang-style token weighting.")
        reweight = False
    elif bfq_policy is not None:
        print("[BFQ] skip MBQ gradient-based reweight stage; using quantization-effect policy instead.")

    if finegrained and finegrained_mode not in ("qig", "attention"):
        print(f"[MBQ][warn] Unknown finegrained_mode={finegrained_mode}; fallback to qig")
        finegrained_mode = "qig"

    # Guard: reweight needs both masks; OneVision path usually lacks them
    if reweight and (vision_mask is None or caption_mask is None):
        print("[MBQ][warn] reweight disabled: vision_mask or caption_mask is None")
        reweight = False

    if finegrained and finegrained_mode == "attention":
        groups = _parse_layer_groups(lagq_layer_groups, num_layers=len(layers))
        for g_name, lids in groups:
            for lid in lids:
                layer_to_group[lid] = g_name
        mbq_results["lagq_layer_groups"] = [g for g, _ in groups]
        mbq_results["lagq_stage1_metric"] = str(lagq_stage1_metric)
        if lagq_disable_attention_token_weight:
            print("[LAGQ] attention token weighting disabled: using only stage-1 layer/modality sensitivity.")
        # Optional stage-1 logit sensitivity by (group, modality)
        if not lagq_disable_logit_sensitivity:
            try:
                sample_inputs: List[Dict[str, torch.Tensor]] = []
                if is_onevision and onevision_samples is not None:
                    for s in onevision_samples:
                        sample_inputs.append({k: v for k, v in s.items()})
                else:
                    bsz = inps.shape[0]
                    for bi in range(bsz):
                        s = {}
                        for k, v in inputs.items():
                            if not isinstance(v, torch.Tensor):
                                continue
                            s[k] = v[bi : bi + 1]
                        sample_inputs.append(s)

                # Optional stage-1 sample budget (0 => use all).
                max_stage1 = int(lagq_stage1_samples) if lagq_stage1_samples is not None else 0
                if max_stage1 > 0 and len(sample_inputs) > max_stage1:
                    sample_inputs = sample_inputs[:max_stage1]

                sv_list: List[torch.Tensor] = []
                st_list: List[torch.Tensor] = []
                for bi in range(len(sample_inputs)):
                    seq_len = None
                    am = sample_inputs[bi].get("attention_mask", None)
                    if torch.is_tensor(am):
                        seq_len = am.shape[1]
                    elif "input_ids" in sample_inputs[bi]:
                        seq_len = sample_inputs[bi]["input_ids"].shape[1]
                    elif "inputs_embeds" in sample_inputs[bi]:
                        seq_len = sample_inputs[bi]["inputs_embeds"].shape[1]
                    if seq_len is None:
                        continue
                    vm = token_vis_mask_batch[bi : bi + 1, :seq_len]
                    tm = token_text_mask_batch[bi : bi + 1, :seq_len]
                    sv_list.append(vm)
                    st_list.append(tm)

                # Stage-1 must run with model/input on the same device.
                # (OneVision often hits CUDA/CPU mismatch here after catcher pass.)
                try:
                    model.to_cuda()
                except Exception:
                    pass
                stage1_device = next(getattr(model, "model", model).parameters()).device
                for s in sample_inputs:
                    for k, v in list(s.items()):
                        if torch.is_tensor(v):
                            s[k] = v.to(stage1_device)
                sv_list = [v.to(stage1_device) if torch.is_tensor(v) else v for v in sv_list]
                st_list = [v.to(stage1_device) if torch.is_tensor(v) else v for v in st_list]

                mbq_results["lagq_stage1_samples_used"] = len(sample_inputs)
                print(f"[LAGQ] stage-1 samples used: {len(sample_inputs)}")
                analysis_start = time.perf_counter()
                lagq_sensitivity_weights = _build_logit_sensitivity_weights(
                    model=model,
                    layers=layers,
                    sample_inputs=sample_inputs,
                    sample_vis_masks=sv_list,
                    sample_text_masks=st_list,
                    layer_groups=groups,
                    noise_bits=int(lagq_noise_bits),
                    noise_strength=float(lagq_noise_strength),
                    stage1_metric=str(lagq_stage1_metric),
                )
                timing_log["analysis_seconds"] = float(timing_log.get("analysis_seconds", 0.0)) + (time.perf_counter() - analysis_start)
                mbq_results["lagq_sensitivity"] = lagq_sensitivity_weights
                print(f"[LAGQ] logit sensitivity weights: {lagq_sensitivity_weights}")
            except Exception as e:
                print(f"[LAGQ][warn] failed to compute logit sensitivity, fallback to uniform: {e}")
                lagq_sensitivity_weights = None
            finally:
                # Keep original MBQ flow (layer-wise processing) on CPU baseline.
                try:
                    model.to_cpu()
                except Exception:
                    pass
                gc.collect()
                torch.cuda.empty_cache()

    if reweight:
        analysis_start = time.perf_counter()
        model.to_cuda()
        print("Save gradient...")
        if is_onevision and onevision_reweight_samples is not None:
            vision_mask = onevision_vis_mask_batch
            caption_mask = onevision_cap_mask_batch
        # save gradient
        grad_cache = GradCacheHook(vis_masks=vision_mask, cap_masks=caption_mask)        
        grad_cache.register_hooks(layers=layers)
        
        with torch.enable_grad():
            mini_batch = 1
            if is_onevision and onevision_reweight_samples is not None:
                total_samples = len(onevision_reweight_samples)
            else:
                total_samples = next(iter(prompt_inputs.values())).shape[0]
            accum_steps = int(total_samples/mini_batch)
            onevision_pixel_ranges = None
            if is_onevision and onevision_reweight_samples is None:
                ov_grid = inputs.get("image_grid_thw", None)
                ov_pv = inputs.get("pixel_values", None)
                if isinstance(ov_grid, torch.Tensor) and isinstance(ov_pv, torch.Tensor):
                    # Some OneVision processors flatten pixel_values across batch.
                    # Build per-sample ranges so pixel_values and image_grid_thw stay aligned.
                    if ov_pv.shape[0] != total_samples:
                        counts = [int(ov_grid[idx].prod().item()) for idx in range(total_samples)]
                        ranges = []
                        st = 0
                        for c in counts:
                            ed = st + c
                            ranges.append((st, ed))
                            st = ed
                        if st == ov_pv.shape[0]:
                            onevision_pixel_ranges = ranges
                        else:
                            merge_size = getattr(getattr(backbone, "visual", None), "spatial_merge_size", 2)
                            reduced_counts = [max(1, c // (merge_size * merge_size)) for c in counts]
                            ranges = []
                            st = 0
                            for c in reduced_counts:
                                ed = st + c
                                ranges.append((st, ed))
                                st = ed
                            if st == ov_pv.shape[0]:
                                onevision_pixel_ranges = ranges
            
            for i in tqdm.tqdm(range(0, total_samples, mini_batch), desc="Running gradient calculation..."):
                mini_inputs = {}
                if is_onevision and onevision_reweight_samples is not None:
                    for k, v in onevision_reweight_samples[i].items():
                        if isinstance(v, torch.Tensor):
                            mini_inputs[k] = v
                else:
                    for k in inputs:
                        if isinstance(inputs[k], torch.Tensor):
                            if (
                                is_onevision
                                and k == "pixel_values"
                                and onevision_pixel_ranges is not None
                                and mini_batch == 1
                            ):
                                st, ed = onevision_pixel_ranges[i]
                                mini_inputs[k] = inputs[k][st:ed]
                            elif is_onevision and k == "image_grid_thw":
                                mini_inputs[k] = inputs[k][i:i+1]
                            else:
                                mini_inputs[k] = inputs[k][i:i+mini_batch]
                
                outputs = model(**mini_inputs)

                loss = outputs[0]

                loss = loss / accum_steps
                loss.backward()

        model.to_cpu()
        grad_avg_dict = grad_cache.get_avg_grad_dict()
        grad_cache.remove_hooks()
        del grad_cache

        attn_list = []
        mlp_list = []

        for key_name in grad_avg_dict:
            if "down_" in key_name or "w2" in key_name:
                mlp_list.append(grad_avg_dict[key_name]["vis_avg_grad"] / grad_avg_dict[key_name]["cap_avg_grad"])
            if "o_proj" in key_name or "wo" in key_name:
                attn_list.append(grad_avg_dict[key_name]["vis_avg_grad"] / grad_avg_dict[key_name]["cap_avg_grad"])

        attn_median = np.median(attn_list)
        mlp_median = np.median(mlp_list)
        timing_log["analysis_seconds"] = float(timing_log.get("analysis_seconds", 0.0)) + (time.perf_counter() - analysis_start)


    if distort:
        # assert wa_quant, "We only support distort input in weight-activation quantization!!!"
        print("Use distort input...")
        inps_distort = copy.deepcopy(inps)

    gc.collect()
    torch.cuda.empty_cache()

    # solve layer by layer
    if blocks is None:
        layer_indices = list(range(len(layers)))
    else:
        layer_indices = []
        for s, e in blocks:
            layer_indices.extend(list(range(s, e + 1)))
    layer_set = set(layer_indices)

    search_start = time.perf_counter()
    pbar = tqdm.tqdm(total=len(layer_indices), desc=desc)
    for i in range(len(layers)):
        if i not in layer_set:
            continue
        pbar.update(1)
        layer = layers[i]
        layer = layer.cuda()
        named_linears = get_named_linears(layer)

        # firstly, get input features of all linear layers
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
        inps = inps.to(next(layer.parameters()).device)  # in case multi-gpu
        if inps.dim() == 2:
            inps = inps.unsqueeze(0)
        # Keep current layer input for scale search (needed by OneVision non-distort WA path).
        layer_inps = inps
        # get output as next layer's input
        cur_kwargs = {}
        for k, v in base_kwargs.items():
            if isinstance(v, torch.Tensor):
                cur_kwargs[k] = v.to(next(layer.parameters()).device)
            else:
                cur_kwargs[k] = v
        is_ov_loop = "onevision" in str(model.model.__class__).lower()
        if not is_ov_loop:
            if cur_kwargs.get("position_embeddings", None) is None:
                cur_kwargs.pop("position_embeddings", None)
            cur_kwargs.pop("position_ids", None)
        else:
            # OneVision: drop masks, always regenerate rotary embeddings to match padded seq len
            cur_kwargs.pop("position_ids", None)
            cur_kwargs.pop("attention_mask", None)
            cur_kwargs.pop("position_embeddings", None)
            cur_kwargs.pop("past_key_value", None)
            cur_kwargs.pop("cache_position", None)
            cur_kwargs["use_cache"] = False
            bsz, seqlen = inps.shape[0], inps.shape[1]
            # Use 2D position ids (bsz, seqlen) so rotary_emb returns (bsz, seqlen, head_dim)
            pos_ids = torch.arange(seqlen, device=inps.device).unsqueeze(0).expand(bsz, -1)
            pos_emb = None
            if rotary_emb_fn is not None:
                pos_emb = rotary_emb_fn(inps, pos_ids)
            elif hasattr(layer, "self_attn") and hasattr(layer.self_attn, "rotary_emb"):
                pos_emb = layer.self_attn.rotary_emb(inps, pos_ids)
            if not (isinstance(pos_emb, tuple) and len(pos_emb) == 2):
                raise RuntimeError("OneVision: failed to rebuild position_embeddings.")
            else:
                # log once for OneVision to verify cos/sin dims vs inps/layer head_dim
                if i == 0:
                    cos, sin = pos_emb
                    hdim = getattr(layer.self_attn, "head_dim", None)
                    print(f"[MBQ][onevision][debug] rope cos={tuple(cos.shape)} sin={tuple(sin.shape)} inps={tuple(inps.shape)} head_dim={hdim}")
            cur_kwargs["position_embeddings"] = pos_emb

        need_layer_attn = bool(
            finegrained
            and finegrained_mode == "attention"
            and (not lagq_disable_attention_token_weight)
        )
        fw_kwargs = dict(cur_kwargs)
        if need_layer_attn:
            fw_kwargs["output_attentions"] = True
        layer_out = layer(inps, **fw_kwargs)
        layer_attn = None
        if isinstance(layer_out, (tuple, list)):
            inps = layer_out[0]
            if need_layer_attn and len(layer_out) > 1:
                # Robust attention extraction across HF variants:
                # some return (hidden, attn, ...), others (hidden, cache, attn, ...)
                for item in layer_out[1:]:
                    if torch.is_tensor(item) and item.dim() >= 4:
                        layer_attn = item
                        break
                if layer_attn is None:
                    for item in layer_out[1:]:
                        if torch.is_tensor(item):
                            layer_attn = item
                            break
        elif torch.is_tensor(layer_out):
            inps = layer_out
        else:
            # ModelOutput-like object
            if hasattr(layer_out, "last_hidden_state") and torch.is_tensor(layer_out.last_hidden_state):
                inps = layer_out.last_hidden_state
            else:
                inps = layer_out[0]
            if need_layer_attn:
                atts = getattr(layer_out, "attentions", None)
                if isinstance(atts, (tuple, list)) and len(atts) > 0 and torch.is_tensor(atts[0]):
                    layer_attn = atts[0]
        # If batch dim is lost (some OneVision/Qwen variants), restore it
        if inps.dim() == 2:
            inps = inps.unsqueeze(0)
        for h in handles:
            h.remove()
        # now solve for scaling
        input_feat = {k: torch.cat(v, dim=0) for k, v in input_feat.items()}

        # Clear GPU memory
        torch.cuda.empty_cache()

        layer_token_weight = None
        if finegrained:
            if finegrained_mode == "qig":
                try:
                    analysis_start = time.perf_counter()
                    with torch.enable_grad():
                        layer_token_weight = _compute_qig_token_weight(
                            layer=layer,
                            layer_input=layer_inps,
                            layer_kwargs=cur_kwargs,
                            w_bit=w_bit,
                            a_bit=a_bit,
                            q_config=q_config,
                            wa_quant=wa_quant,
                            qig_steps=qig_steps,
                            qig_iqr_factor=qig_iqr_factor,
                            qig_eps=qig_eps,
                            qig_disable_iqr=qig_disable_iqr,
                            qig_use_abs=qig_use_abs,
                        )
                    timing_log["analysis_seconds"] = float(timing_log.get("analysis_seconds", 0.0)) + (time.perf_counter() - analysis_start)
                except Exception as e:
                    print(f"[MBQ][warn] QIG token weighting failed at layer {i}: {e}. Falling back to unweighted loss.")
                    layer_token_weight = None
            else:
                try:
                    analysis_start = time.perf_counter()
                    if lagq_disable_attention_token_weight:
                        bsz, seqlen = inps.shape[0], inps.shape[1]
                        if is_ov_loop:
                            valmask = onevision_token_valid_mask
                        else:
                            valmask = inputs.get("attention_mask", None)
                        if torch.is_tensor(valmask):
                            valid = _align_mask_2d(valmask, bsz, seqlen, inps.device).to(inps.dtype)
                        else:
                            valid = torch.ones((bsz, seqlen), dtype=inps.dtype, device=inps.device)
                        denom = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
                        layer_token_weight = (valid / denom).detach()
                    else:
                        if layer_attn is None:
                            layer_token_weight = None
                            raise RuntimeError(
                                "layer_attn is None while attention token weighting is enabled. "
                                "This run is invalid for attention-based LAGQ; use an attention backend that returns attentions."
                            )
                        vmask = token_vis_mask_batch
                        tmask = token_text_mask_batch
                        if torch.is_tensor(vmask):
                            vmask = _align_mask_2d(vmask, layer_attn.shape[0], layer_attn.shape[2], layer_attn.device)
                        if torch.is_tensor(tmask):
                            tmask = _align_mask_2d(tmask, layer_attn.shape[0], layer_attn.shape[2], layer_attn.device)
                        valmask = onevision_token_valid_mask if is_ov_loop else inputs.get("attention_mask", None)
                        if torch.is_tensor(valmask):
                            valmask = _align_mask_2d(valmask, layer_attn.shape[0], layer_attn.shape[2], layer_attn.device)
                        layer_token_weight = _compute_attention_token_weight(
                            layer_attn=layer_attn,
                            vision_mask=vmask,
                            text_mask=tmask,
                            valid_mask=valmask,
                            text_correction=lagq_text_attn_correction,
                            vision_query_source=lagq_vision_query_source,
                            query_mix_alpha=lagq_query_mix_alpha,
                            eps=qig_eps,
                        )
                    timing_log["analysis_seconds"] = float(timing_log.get("analysis_seconds", 0.0)) + (time.perf_counter() - analysis_start)
                except Exception as e:
                    if "layer_attn is None while attention token weighting is enabled" in str(e):
                        raise
                    print(f"[LAGQ][warn] token weighting build failed at layer {i}: {e}. Falling back to unweighted loss.")
                    layer_token_weight = None
            # For OneVision padded calibration batches, exclude padded tokens from
            # token-wise weighting so normalization is not diluted by padding.
            if (
                layer_token_weight is not None
                and is_ov_loop
                and isinstance(onevision_token_valid_mask, torch.Tensor)
            ):
                vmask = onevision_token_valid_mask.to(layer_token_weight.device)
                if vmask.shape[0] > layer_token_weight.shape[0]:
                    vmask = vmask[: layer_token_weight.shape[0]]
                elif vmask.shape[0] < layer_token_weight.shape[0]:
                    pad_b = torch.zeros(
                        layer_token_weight.shape[0] - vmask.shape[0],
                        vmask.shape[1],
                        dtype=torch.bool,
                        device=vmask.device,
                    )
                    vmask = torch.cat([vmask, pad_b], dim=0)
                if vmask.shape[1] > layer_token_weight.shape[1]:
                    vmask = vmask[:, : layer_token_weight.shape[1]]
                elif vmask.shape[1] < layer_token_weight.shape[1]:
                    pad_s = torch.zeros(
                        vmask.shape[0],
                        layer_token_weight.shape[1] - vmask.shape[1],
                        dtype=torch.bool,
                        device=vmask.device,
                    )
                    vmask = torch.cat([vmask, pad_s], dim=1)
                layer_token_weight = layer_token_weight * vmask.to(layer_token_weight.dtype)
                denom = layer_token_weight.sum(dim=1, keepdim=True)
                zero_row = denom <= qig_eps
                if zero_row.any():
                    # If a row is fully masked out (should be rare), fall back to uniform valid-token weights.
                    fallback = vmask.to(layer_token_weight.dtype)
                    fb_denom = fallback.sum(dim=1, keepdim=True).clamp_min(1.0)
                    fallback = fallback / fb_denom
                    layer_token_weight = torch.where(zero_row.expand_as(layer_token_weight), fallback, layer_token_weight)
                    denom = layer_token_weight.sum(dim=1, keepdim=True)
                layer_token_weight = layer_token_weight / (denom + qig_eps)

            # Apply stage-1 sensitivity multipliers (layer-group x modality), then renormalize.
            if (
                layer_token_weight is not None
                and finegrained_mode == "attention"
                and isinstance(token_vis_mask_batch, torch.Tensor)
                and isinstance(token_text_mask_batch, torch.Tensor)
            ):
                g_name = layer_to_group.get(i, None)
                if g_name is not None and isinstance(lagq_sensitivity_weights, dict):
                    wm = lagq_sensitivity_weights.get(g_name, {})
                    wv = float(wm.get("vision", 1.0))
                    wt = float(wm.get("text", 1.0))
                    vmask = _align_mask_2d(token_vis_mask_batch, layer_token_weight.shape[0], layer_token_weight.shape[1], layer_token_weight.device)
                    tmask = _align_mask_2d(token_text_mask_batch, layer_token_weight.shape[0], layer_token_weight.shape[1], layer_token_weight.device)
                    scale = torch.ones_like(layer_token_weight)
                    scale = torch.where(vmask, torch.full_like(scale, wv), scale)
                    scale = torch.where(tmask, torch.full_like(scale, wt), scale)
                    layer_token_weight = layer_token_weight * scale
                    denom = layer_token_weight.sum(dim=1, keepdim=True)
                    zero_row = denom <= qig_eps
                    if zero_row.any():
                        fb = (vmask | tmask).to(layer_token_weight.dtype)
                        fb_denom = fb.sum(dim=1, keepdim=True).clamp_min(1.0)
                        fb = fb / fb_denom
                        layer_token_weight = torch.where(zero_row.expand_as(layer_token_weight), fb, layer_token_weight)
                        denom = layer_token_weight.sum(dim=1, keepdim=True)
                    layer_token_weight = layer_token_weight / (denom + qig_eps)

        layer_bfq_policy = None
        if isinstance(bfq_policy, dict):
            layer_bfq_policy = bfq_policy.get("layers", {}).get(str(i))

        search_cfg = None
        if layer_bfq_policy is not None:
            branch_policy = layer_bfq_policy.get("search_policy")
            if not isinstance(branch_policy, dict):
                branch_policy = layer_bfq_policy.get("activation_policy" if wa_quant else "weight_policy", {})
            if not isinstance(branch_policy, dict):
                branch_policy = {}
            scale_reweight_ratio_dict = {
                "attn": float(layer_bfq_policy.get("reweight_ratio", 1.0)),
                "mlp": float(layer_bfq_policy.get("reweight_ratio", 1.0)),
            }
            search_cfg = {
                "n_grid": int(branch_policy.get("n_grid", layer_bfq_policy.get("n_grid", 20))),
                "ratio_max": float(branch_policy.get("ratio_max", layer_bfq_policy.get("ratio_max", 1.0))),
            }
        elif reweight:
            scale_reweight_ratio_dict = {}
            for key, value in grad_avg_dict.items():
                item_list = key.split(".")
                if str(i) in item_list:
                    if "wo" in item_list or "o_proj" in item_list:
                        scale_reweight_ratio_dict["attn"] = max((value["vis_avg_grad"] / value["cap_avg_grad"]), attn_median)
                    elif "w2" in item_list or "down_proj" in item_list:
                        scale_reweight_ratio_dict["mlp"] = max((value["vis_avg_grad"] / value["cap_avg_grad"]), mlp_median)
        else:
            scale_reweight_ratio_dict = {
                "attn": None,
                "mlp": None
            }

        if (
            auto_scale
        ):  # if it applies, we should also modify the input_feat with scales
            if not reweight and layer_bfq_policy is None:
                ans_mask = None
                vis_mask = None
            else:
                ans_mask = caption_mask
                vis_mask = vision_mask
                if isinstance(ans_mask, torch.Tensor):
                    ans_mask = ans_mask.to(next(layer.parameters()).device)
                if isinstance(vis_mask, torch.Tensor):
                    vis_mask = vis_mask.to(next(layer.parameters()).device)
                if isinstance(ans_mask, torch.Tensor) and ans_mask.dim() == 2:
                    tgt_len = inps.shape[1]
                    if ans_mask.shape[1] > tgt_len:
                        ans_mask = ans_mask[:, :tgt_len]
                    elif ans_mask.shape[1] < tgt_len:
                        pad = torch.zeros(ans_mask.shape[0], tgt_len - ans_mask.shape[1], dtype=torch.bool, device=ans_mask.device)
                        ans_mask = torch.cat([ans_mask, pad], dim=1)
                if isinstance(vis_mask, torch.Tensor) and vis_mask.dim() == 2:
                    tgt_len = inps.shape[1]
                    if vis_mask.shape[1] > tgt_len:
                        vis_mask = vis_mask[:, :tgt_len]
                    elif vis_mask.shape[1] < tgt_len:
                        pad = torch.zeros(vis_mask.shape[0], tgt_len - vis_mask.shape[1], dtype=torch.bool, device=vis_mask.device)
                        vis_mask = torch.cat([vis_mask, pad], dim=1)
            
            layer_search_start = time.perf_counter()
            if wa_quant:
                if distort:
                    scales_list = auto_scale_block_wa_distort(
                        layer,
                        layer_kwargs,
                        w_bit=w_bit,
                        a_bit=a_bit,
                        q_config=q_config,
                        input_feat=input_feat,
                        ans_mask=ans_mask,
                        vis_mask=vis_mask,
                        reweight_ratio_dict=scale_reweight_ratio_dict,
                        q_input=inps_distort,
                        loss_mode=loss_mode,
                        token_weight=layer_token_weight,
                        search_cfg=search_cfg,
                    )
                else:
                    # OneVision decoder has no dedicated auto_scale_wa branch.
                    # Reuse WA-distort backend with q_input=current layer input to keep MBQ scaling active.
                    if "OneVision" in str(type(layer)):
                        scales_list = auto_scale_block_wa_distort(
                            layer,
                            layer_kwargs,
                            w_bit=w_bit,
                            a_bit=a_bit,
                            q_config=q_config,
                            input_feat=input_feat,
                            ans_mask=ans_mask,
                            vis_mask=vis_mask,
                            reweight_ratio_dict=scale_reweight_ratio_dict,
                            q_input=layer_inps,
                            loss_mode=loss_mode,
                            token_weight=layer_token_weight,
                            search_cfg=search_cfg,
                        )
                    else:
                        scales_list = auto_scale_block_wa(
                            layer,
                            layer_kwargs,
                            w_bit=w_bit,
                            a_bit=a_bit,
                            q_config=q_config,
                            input_feat=input_feat,
                            ans_mask=ans_mask,
                            vis_mask=vis_mask,
                            reweight_ratio_dict=scale_reweight_ratio_dict,
                            loss_mode=loss_mode,
                            token_weight=layer_token_weight,
                            search_cfg=search_cfg,
                        )
            else:
                if distort:
                    scales_list = auto_scale_block_distort(
                        layer,
                        layer_kwargs,
                        w_bit=w_bit,
                        q_config=q_config,
                        input_feat=input_feat,
                        ans_mask=ans_mask,
                        vis_mask=vis_mask,
                        reweight_ratio_dict=scale_reweight_ratio_dict,
                        q_input=inps_distort,
                        loss_mode=loss_mode,
                        token_weight=layer_token_weight,
                        search_cfg=search_cfg,
                    )
                else:
                    scales_list = auto_scale_block(
                        layer,
                        layer_kwargs,
                        w_bit=w_bit,
                        q_config=q_config,
                        input_feat=input_feat,
                        ans_mask=ans_mask,
                        vis_mask=vis_mask,
                        reweight_ratio_dict=scale_reweight_ratio_dict,
                        loss_mode=loss_mode,
                        token_weight=layer_token_weight,
                        search_cfg=search_cfg,
                    )
            timing_log["search_seconds"] = float(timing_log.get("search_seconds", 0.0)) + (time.perf_counter() - layer_search_start)

            # apply_scale(layer, scales_list, input_feat_dict=input_feat)
            apply_scale(layers[i], scales_list, input_feat_dict=input_feat)

            if distort:
                # get distort output as next layer's input
                if wa_quant:
                    layer_q = copy.deepcopy(layer)
                    layer_q = layer_q.cuda()
                    named_linears_q = get_named_linears(layer_q)
                    for n, m in named_linears_q.items():
                        new_linear = WALinear.from_float(m, weight_quant="per_channel", act_quant="per_token", w_bit=w_bit, a_bit=a_bit)
                        father_module = get_module_by_name_suffix(layer_q, '.'.join(n.split(".")[:-1]))
                        setattr(father_module, n.split('.')[-1], new_linear)
                        del new_linear, m
                        torch.cuda.empty_cache()
                    # rebuild kwargs for distort forward (OneVision needs fresh rope, no cache)
                    dist_kwargs = {}
                    for k, v in layer_kwargs.items():
                        if isinstance(v, torch.Tensor):
                            dist_kwargs[k] = v.to(next(layer_q.parameters()).device)
                        else:
                            dist_kwargs[k] = v
                    if is_onevision:
                        dist_kwargs.pop("position_ids", None)
                        dist_kwargs.pop("attention_mask", None)
                        dist_kwargs.pop("position_embeddings", None)
                        dist_kwargs.pop("past_key_value", None)
                        dist_kwargs.pop("cache_position", None)
                        dist_kwargs["use_cache"] = False
                        bsz, seqlen = inps_distort.shape[0], inps_distort.shape[1]
                        pos_ids = torch.arange(seqlen, device=inps_distort.device).unsqueeze(0).expand(bsz, -1)
                        pos_emb = None
                        if hasattr(layer_q, "self_attn") and hasattr(layer_q.self_attn, "rotary_emb"):
                            try:
                                pos_emb = layer_q.self_attn.rotary_emb(inps_distort, pos_ids)
                            except TypeError:
                                try:
                                    pos_emb = layer_q.self_attn.rotary_emb(pos_ids)
                                except Exception:
                                    pos_emb = None
                        if isinstance(pos_emb, tuple) and len(pos_emb) == 2:
                            cos, sin = pos_emb
                            if cos.shape[1] != seqlen or sin.shape[1] != seqlen:
                                try:
                                    pos_emb = layer_q.self_attn.rotary_emb(pos_ids)
                                    cos, sin = pos_emb
                                except Exception:
                                    cos = cos[:, :seqlen, :]
                                    sin = sin[:, :seqlen, :]
                                    pos_emb = (cos, sin)
                        if pos_emb is None:
                            head_dim = getattr(layer_q.self_attn, "head_dim", max(1, inps_distort.shape[-1] // getattr(layer_q.self_attn, "num_heads", 1)))
                            cos = torch.zeros(bsz, seqlen, head_dim, device=inps_distort.device, dtype=inps_distort.dtype)
                            sin = torch.zeros_like(cos)
                            pos_emb = (cos, sin)
                        dist_kwargs["position_embeddings"] = pos_emb
                    inps_distort = inps_distort.to(next(layer_q.parameters()).device)  # in case multi-gpu
                    inps_distort = layer_q(inps_distort, **dist_kwargs)[0]
                    del layer_q 
                else:
                    layer_q = copy.deepcopy(layer)
                    layer_q = layer_q.cuda()
                    named_linears_q = get_named_linears(layer_q)
                    for n, m in named_linears_q.items():
                        m.weight.data = pseudo_quantize_tensor(m.weight.data, n_bits=w_bit, **q_config)
                        torch.cuda.empty_cache()
                    dist_kwargs = {}
                    for k, v in layer_kwargs.items():
                        if isinstance(v, torch.Tensor):
                            dist_kwargs[k] = v.to(next(layer_q.parameters()).device)
                        else:
                            dist_kwargs[k] = v
                    if is_onevision:
                        dist_kwargs.pop("position_ids", None)
                        dist_kwargs.pop("attention_mask", None)
                        dist_kwargs.pop("position_embeddings", None)
                        dist_kwargs.pop("past_key_value", None)
                        dist_kwargs.pop("cache_position", None)
                        dist_kwargs["use_cache"] = False
                        bsz, seqlen = inps_distort.shape[0], inps_distort.shape[1]
                        pos_ids = torch.arange(seqlen, device=inps_distort.device).unsqueeze(0).expand(bsz, -1)
                        pos_emb = None
                        if hasattr(layer_q, "self_attn") and hasattr(layer_q.self_attn, "rotary_emb"):
                            try:
                                pos_emb = layer_q.self_attn.rotary_emb(inps_distort, pos_ids)
                            except TypeError:
                                try:
                                    pos_emb = layer_q.self_attn.rotary_emb(pos_ids)
                                except Exception:
                                    pos_emb = None
                        if isinstance(pos_emb, tuple) and len(pos_emb) == 2:
                            cos, sin = pos_emb
                            if cos.shape[1] != seqlen or sin.shape[1] != seqlen:
                                try:
                                    pos_emb = layer_q.self_attn.rotary_emb(pos_ids)
                                    cos, sin = pos_emb
                                except Exception:
                                    cos = cos[:, :seqlen, :]
                                    sin = sin[:, :seqlen, :]
                                    pos_emb = (cos, sin)
                        if pos_emb is None:
                            head_dim = getattr(layer_q.self_attn, "head_dim", max(1, inps_distort.shape[-1] // getattr(layer_q.self_attn, "num_heads", 1)))
                            cos = torch.zeros(bsz, seqlen, head_dim, device=inps_distort.device, dtype=inps_distort.dtype)
                            sin = torch.zeros_like(cos)
                            pos_emb = (cos, sin)
                        dist_kwargs["position_embeddings"] = pos_emb
                    inps_distort = inps_distort.to(next(layer_q.parameters()).device)  # in case multi-gpu
                    inps_distort = layer_q(inps_distort, **dist_kwargs)[0]
                    del layer_q 

            # append prefix to make names global
            mbq_results["scale"] += append_str_prefix(
                scales_list, get_op_name(model.model, layer) + "."
            )

        # Clear GPU memory
        torch.cuda.empty_cache()

        layer = layer.cpu()
        # Haotian: check activation replacement
        del input_feat
        gc.collect()
        torch.cuda.empty_cache()

    return mbq_results


def apply_mbq(model, mbq_results):
    scales = mbq_results["scale"]
    # QVLM path may store block-wise dicts; flatten to the tuple list format expected by apply_scale
    if len(scales) > 0 and isinstance(scales[0], dict):
        flat = []
        for block in scales:
            for prefix, v in block.items():
                pref = prefix if prefix.endswith(".") else prefix + "."
                prefixed = append_str_prefix(v, pref)
                if isinstance(prefixed, tuple) and len(prefixed) == 3:
                    flat.append(prefixed)
                elif isinstance(prefixed, list):
                    for item in prefixed:
                        if isinstance(item, tuple) and len(item) == 3:
                            flat.append(item)
                        elif isinstance(item, (list, tuple)) and len(item) == 3:
                            flat.append(tuple(item))
        scales = flat
    # normalize malformed prefixes only when they carry extra leading "layers.<idx>"
    fixed_scales = []
    def _fix(name):
        if not isinstance(name, str):
            return name
        # only strip malformed prefixes; keep valid names like "layers.0.input_layernorm".
        if re.match(r"^layers\.\d+(?=[A-Za-z_])", name):
            # Some external QVLM caches are serialized with malformed prefixes like
            # "layers.0language_model.model.layers.0..." (missing dot after the index).
            name = re.sub(r"^layers\.\d+(?=[A-Za-z_])", "", name)
        if re.match(r"^layers\.\d+model\.", name):
            name = re.sub(r"^layers\.\d+(model\.)", r"\1", name)
        return name
    for prev_op_name, layer_names, s in scales:
        new_prev = _fix(prev_op_name)
        new_layers = tuple(_fix(n) for n in layer_names)
        fixed_scales.append((new_prev, new_layers, s))
    apply_scale(model, fixed_scales)
