import torch
import torch.nn as nn
import tqdm
import copy
import gc
import functools
import time
from collections import defaultdict
from typing import List

import numpy as np
from torch.nn import CrossEntropyLoss
from transformers.models.bloom.modeling_bloom import BloomForCausalLM
from transformers.models.opt.modeling_opt import OPTForCausalLM
from transformers.models.llama.modeling_llama import LlamaForCausalLM

from qmllm.utils.search import append_str_prefix, get_op_name

from qmllm.methods.mbq.quantize.auto_scale_wa_distort import auto_scale_block_wa_distort
from qmllm.methods.mbq.quantize.auto_scale_wa import auto_scale_block_wa
from qmllm.methods.mbq.quantize.auto_scale_distort import auto_scale_block_distort
from qmllm.methods.mbq.quantize.auto_scale import auto_scale_block, apply_scale
from qmllm.quantization.qlinear import WALinear
from qmllm.quantization.quant_funcs import pseudo_quantize_tensor
from .quantizer import get_module_by_name_suffix


__all__ = ["run_mbq"]


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
    elif model.__class__.__name__ == "InternVLChatModel":
        model.language_model.model.tok_embeddings = model.language_model.model.tok_embeddings.to(device)  
    elif model.__class__.__name__ == "Qwen2VLForConditionalGeneration":
        if hasattr(model.model, "embed_tokens"):
            model.model.embed_tokens = model.model.embed_tokens.to(device)
        if hasattr(model.model, "language_model") and hasattr(model.model.language_model, "embed_tokens"):
            model.model.language_model.embed_tokens = model.model.language_model.embed_tokens.to(device)
    elif "qwen2" in str(model.__class__).lower() and hasattr(model, "model"):
        # fallback for Qwen2.5 VL-like wrappers
        if hasattr(model.model, "language_model") and hasattr(model.model.language_model, "embed_tokens"):
            model.model.language_model.embed_tokens = model.model.language_model.embed_tokens.to(device)
    elif "llavaonevision" in str(model.__class__).lower():
        root = getattr(model, "language_model", None)
        if root is None:
            root = getattr(model, "model", None)
        if root is not None and hasattr(root, "embed_tokens"):
            root.embed_tokens = root.embed_tokens.to(device)
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
    mbq_debug_grad_samples: int = 0,
    mbq_debug_max_layers: int = 0,
    glmi_policy=None,
    timing_log=None,
):
    if "bigcode" in str(model.model.__class__).lower():
        # otherwise attention_mask will always be on cpu.
        model.transformer.bias = model.transformer.bias.to("cuda")

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
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.module, name)

        def forward(self, inp, **kwargs):
            inps.append(inp)
            layer_kwargs.update(kwargs)
            raise ValueError  # early exit to break later inference

    # patch layer 0 to catch input and kwargs
    layers[0] = Catcher(layers[0])

    inputs, vision_mask, caption_mask = process_input(prompt_inputs, prompt_kwargs)

    # Canonicalize modality/answer masks to token mask shape [B, S] once.
    # Older/newer model wrappers can accidentally pass feature-shaped masks (e.g., [S, C]).
    def _infer_bsz_seqlen(_inputs):
        if isinstance(_inputs.get("labels", None), torch.Tensor):
            t = _inputs["labels"]
            if t.dim() >= 2:
                return int(t.shape[0]), int(t.shape[1])
        if isinstance(_inputs.get("attention_mask", None), torch.Tensor):
            t = _inputs["attention_mask"]
            if t.dim() >= 2:
                return int(t.shape[0]), int(t.shape[1])
        if isinstance(_inputs.get("input_ids", None), torch.Tensor):
            t = _inputs["input_ids"]
            if t.dim() >= 2:
                return int(t.shape[0]), int(t.shape[1])
        if isinstance(_inputs.get("inputs_embeds", None), torch.Tensor):
            t = _inputs["inputs_embeds"]
            if t.dim() >= 3:
                return int(t.shape[0]), int(t.shape[1])
        return 1, 1

    def _coerce_mask_2d(mask, target_bsz: int, target_seqlen: int, device: torch.device):
        if not torch.is_tensor(mask):
            return None
        m = mask.to(device=device)
        # Reduce higher-rank mask to token mask.
        while m.dim() > 2:
            if m.shape[-1] == 1:
                m = m.squeeze(-1)
            else:
                m = (m != 0).any(dim=-1)
        if m.dim() == 1:
            m = m.unsqueeze(0)
        if m.dim() == 2 and m.shape[0] == target_seqlen and m.shape[1] != target_seqlen:
            # [S, C]-like accidental feature mask -> [1, S]
            m = (m != 0).any(dim=-1, keepdim=True).transpose(0, 1)
        if m.dim() != 2:
            return None
        cur_b = int(m.shape[0])
        if cur_b != target_bsz:
            if cur_b > target_bsz:
                m = m[:target_bsz]
            elif cur_b == 1:
                m = m.expand((target_bsz, m.shape[1]))
            else:
                rep = int(np.ceil(float(target_bsz) / float(cur_b)))
                m = m.repeat(rep, 1)[:target_bsz]
        if m.shape[1] > target_seqlen:
            m = m[:, :target_seqlen]
        elif m.shape[1] < target_seqlen:
            pad = torch.zeros((m.shape[0], target_seqlen - m.shape[1]), dtype=m.dtype, device=m.device)
            m = torch.cat([m, pad], dim=1)
        return m.to(dtype=torch.bool)

    bsz0, seqlen0 = _infer_bsz_seqlen(inputs)
    mask_device = (
        inputs["labels"].device if isinstance(inputs.get("labels", None), torch.Tensor)
        else (inputs["attention_mask"].device if isinstance(inputs.get("attention_mask", None), torch.Tensor)
              else (inputs["inputs_embeds"].device if isinstance(inputs.get("inputs_embeds", None), torch.Tensor) else "cuda"))
    )

    # answer mask: always derive from labels when available.
    if isinstance(inputs.get("labels", None), torch.Tensor):
        caption_mask = (inputs["labels"] != -100)
    caption_mask = _coerce_mask_2d(caption_mask, bsz0, seqlen0, mask_device)
    if caption_mask is None:
        caption_mask = torch.ones((bsz0, seqlen0), dtype=torch.bool, device=mask_device)
        print("[MBQ][warn] caption_mask unavailable/invalid; fallback to all-ones token mask.")

    vision_mask = _coerce_mask_2d(vision_mask, bsz0, seqlen0, mask_device)
    if vision_mask is None:
        vision_mask = torch.zeros((bsz0, seqlen0), dtype=torch.bool, device=mask_device)
        print("[MBQ][warn] vision_mask unavailable/invalid; fallback to all-zeros token mask.")

    model.to_cuda()
    try:
        model(**inputs)
    except ValueError: # work with early exit
        pass

    model.to_cpu()
    layers[0] = layers[0].module  # restore
    inps = inps[0]
    layer_kwargs["use_cache"] = False

    layers[0] = layers[0].cpu()
    move_embed(model.model, "cpu")

    gc.collect()
    torch.cuda.empty_cache()

    mbq_results = {
        "scale": [],
    }

    # Keep immutable base kwargs captured from the model forward.
    base_kwargs = dict(layer_kwargs)

    def _find_rotary_emb_fn(root_model):
        candidates = [
            getattr(root_model, "model", None),
            root_model,
        ]
        for cand in candidates:
            if cand is None:
                continue
            if hasattr(cand, "rotary_emb"):
                return cand.rotary_emb
            if hasattr(cand, "language_model") and hasattr(cand.language_model, "rotary_emb"):
                return cand.language_model.rotary_emb
            if hasattr(cand, "model") and hasattr(cand.model, "rotary_emb"):
                return cand.model.rotary_emb
            if (
                hasattr(cand, "model")
                and hasattr(cand.model, "language_model")
                and hasattr(cand.model.language_model, "rotary_emb")
            ):
                return cand.model.language_model.rotary_emb
        return None

    rotary_emb_fn = _find_rotary_emb_fn(model.model)

    def _valid_pos_emb(pos_emb, bsz, seqlen, head_dim):
        if not (isinstance(pos_emb, tuple) and len(pos_emb) == 2):
            return False
        cos, sin = pos_emb
        if (not torch.is_tensor(cos)) or (not torch.is_tensor(sin)):
            return False
        if cos.shape[-1] != head_dim or sin.shape[-1] != head_dim:
            return False
        # Qwen2-VL style multimodal rotary embeddings: [3, B, S, D]
        if cos.dim() == 4 or sin.dim() == 4:
            return (
                cos.dim() == 4
                and sin.dim() == 4
                and cos.shape[0] == 3
                and sin.shape[0] == 3
                and cos.shape[1] == bsz
                and sin.shape[1] == bsz
                and cos.shape[2] == seqlen
                and sin.shape[2] == seqlen
            )
        # Accept [B, S, D] or [S, D] (for single sample)
        if cos.dim() >= 2 and cos.shape[-2] != seqlen:
            return False
        if cos.dim() >= 3 and cos.shape[0] != bsz:
            return False
        return True

    def _is_qwen2_vl_decoder(module):
        return module.__class__.__name__ in ["Qwen2VLDecoderLayer", "Qwen2_5_VLDecoderLayer"]

    def _module_device(module):
        if module is None:
            return None
        try:
            return next(module.parameters()).device
        except Exception:
            pass
        try:
            return next(module.buffers()).device
        except Exception:
            pass
        return None

    def _rebuild_pos_emb(inps_tensor, kwargs_dict, layer_mod):
        # Try to reuse provided position_ids when available.
        bsz, seqlen = inps_tensor.shape[0], inps_tensor.shape[1]
        pos_ids = kwargs_dict.get("position_ids", None)
        if torch.is_tensor(pos_ids):
            if pos_ids.dim() == 3:
                if _is_qwen2_vl_decoder(layer_mod):
                    pos_ids = pos_ids.to(device=inps_tensor.device)
                    if pos_ids.shape[0] != 3:
                        if pos_ids.shape[0] == 1:
                            pos_ids = pos_ids.expand(3, -1, -1)
                        else:
                            pos_ids = pos_ids[:3]
                    if pos_ids.shape[1] != bsz:
                        if pos_ids.shape[1] == 1:
                            pos_ids = pos_ids.expand(-1, bsz, -1)
                        else:
                            pos_ids = pos_ids[:, :bsz, :]
                    if pos_ids.shape[2] != seqlen:
                        if pos_ids.shape[2] > seqlen:
                            pos_ids = pos_ids[:, :, :seqlen]
                        else:
                            pad = torch.arange(
                                pos_ids.shape[2], seqlen, device=pos_ids.device, dtype=pos_ids.dtype
                            ).view(1, 1, -1).expand(pos_ids.shape[0], bsz, -1)
                            pos_ids = torch.cat([pos_ids, pad], dim=2)
                else:
                    # Non-multimodal rotary paths use [B, S]
                    pos_ids = pos_ids[0]
            if pos_ids.dim() == 1:
                pos_ids = pos_ids.unsqueeze(0).expand(bsz, -1)
            if pos_ids.dim() == 2:
                pos_ids = pos_ids.to(device=inps_tensor.device)
                if pos_ids.shape[0] != bsz:
                    if pos_ids.shape[0] == 1:
                        pos_ids = pos_ids.expand(bsz, -1)
                    else:
                        pos_ids = pos_ids[:bsz]
                if pos_ids.shape[1] != seqlen:
                    if pos_ids.shape[1] > seqlen:
                        pos_ids = pos_ids[:, :seqlen]
                    else:
                        pad = torch.arange(
                            pos_ids.shape[1], seqlen, device=pos_ids.device, dtype=pos_ids.dtype
                        ).unsqueeze(0).expand(bsz, -1)
                        pos_ids = torch.cat([pos_ids, pad], dim=1)
            else:
                pos_ids = None
        else:
            pos_ids = None

        if pos_ids is None:
            pos_ids = torch.arange(seqlen, device=inps_tensor.device).unsqueeze(0).expand(bsz, -1)
            if _is_qwen2_vl_decoder(layer_mod):
                pos_ids = pos_ids.unsqueeze(0).expand(3, bsz, seqlen)

        rotary_mod = None
        if hasattr(layer_mod, "self_attn") and hasattr(layer_mod.self_attn, "rotary_emb"):
            rotary_mod = layer_mod.self_attn.rotary_emb
        elif rotary_emb_fn is not None:
            rotary_mod = rotary_emb_fn

        if rotary_mod is not None:
            rotary_dev = _module_device(rotary_mod)
            x_for_rot = inps_tensor if rotary_dev is None else inps_tensor.to(rotary_dev)
            pos_ids_for_rot = pos_ids if rotary_dev is None else pos_ids.to(rotary_dev)
            pos_emb = rotary_mod(x_for_rot, pos_ids_for_rot)
            if isinstance(pos_emb, tuple) and len(pos_emb) == 2:
                return tuple(t.to(inps_tensor.device) if torch.is_tensor(t) else t for t in pos_emb)

        return None

    def _sanitize_pos_emb(kwargs_dict, inps_tensor, layer_mod):
        if not (hasattr(layer_mod, "self_attn") and hasattr(layer_mod.self_attn, "head_dim")):
            return kwargs_dict
        head_dim = int(layer_mod.self_attn.head_dim)
        bsz, seqlen = int(inps_tensor.shape[0]), int(inps_tensor.shape[1])
        pe = kwargs_dict.get("position_embeddings", None)
        ok = False
        if isinstance(pe, tuple) and len(pe) == 2 and torch.is_tensor(pe[0]) and torch.is_tensor(pe[1]):
            cos, sin = pe
            # Normalize common malformed shapes.
            if cos.dim() == 4 and cos.shape[1] == 1:
                cos = cos[:, 0]
            if sin.dim() == 4 and sin.shape[1] == 1:
                sin = sin[:, 0]
            if _valid_pos_emb((cos, sin), bsz, seqlen, head_dim):
                kwargs_dict["position_embeddings"] = (cos, sin)
                ok = True
            if cos.dim() == 2:
                cos = cos.unsqueeze(0)
            if sin.dim() == 2:
                sin = sin.unsqueeze(0)
            if (not ok) and cos.dim() == 3 and sin.dim() == 3:
                cos = _align_batch_tensor(cos, bsz)
                sin = _align_batch_tensor(sin, bsz)
                if cos.shape[1] > seqlen:
                    cos = cos[:, :seqlen, :]
                elif cos.shape[1] < seqlen:
                    pad = torch.zeros((cos.shape[0], seqlen - cos.shape[1], cos.shape[2]), dtype=cos.dtype, device=cos.device)
                    cos = torch.cat([cos, pad], dim=1)
                if sin.shape[1] > seqlen:
                    sin = sin[:, :seqlen, :]
                elif sin.shape[1] < seqlen:
                    pad = torch.zeros((sin.shape[0], seqlen - sin.shape[1], sin.shape[2]), dtype=sin.dtype, device=sin.device)
                    sin = torch.cat([sin, pad], dim=1)
                ok = cos.shape[-1] == head_dim and sin.shape[-1] == head_dim and cos.shape[1] == seqlen and sin.shape[1] == seqlen
                if ok:
                    kwargs_dict["position_embeddings"] = (cos, sin)
        if not ok:
            rebuilt = _rebuild_pos_emb(inps_tensor, kwargs_dict, layer_mod)
            if rebuilt is not None and _valid_pos_emb(rebuilt, bsz, seqlen, head_dim):
                kwargs_dict["position_embeddings"] = rebuilt
            else:
                # deterministic last fallback
                cos = torch.zeros((bsz, seqlen, head_dim), dtype=inps_tensor.dtype, device=inps_tensor.device)
                sin = torch.zeros_like(cos)
                kwargs_dict["position_embeddings"] = (cos, sin)
        return kwargs_dict

    def _align_batch_tensor(t: torch.Tensor, target_bsz: int) -> torch.Tensor:
        if (not torch.is_tensor(t)) or t.dim() == 0:
            return t
        cur = int(t.shape[0])
        if cur == target_bsz:
            return t
        if cur > target_bsz:
            return t[:target_bsz]
        # cur < target_bsz
        if cur == 1:
            return t.expand((target_bsz,) + tuple(t.shape[1:]))
        reps = [1] * t.dim()
        reps[0] = int(np.ceil(float(target_bsz) / float(cur)))
        return t.repeat(*reps)[:target_bsz]

    def _align_token_mask(mask: torch.Tensor, target_bsz: int, target_seqlen: int, device: torch.device):
        if not torch.is_tensor(mask):
            return mask
        m = mask.to(device=device)
        # Reduce unexpected high-rank masks to token-level mask.
        while m.dim() > 2:
            if m.shape[-1] == 1:
                m = m.squeeze(-1)
            else:
                m = (m != 0).any(dim=-1)
        if m.dim() == 1:
            m = m.unsqueeze(0)
        # Handle transposed feature-like shape [S, C] -> [1, S]
        if m.dim() == 2 and m.shape[0] == target_seqlen and m.shape[1] != target_seqlen:
            m = (m != 0).any(dim=-1, keepdim=True).transpose(0, 1)
        if m.dim() >= 2:
            m = _align_batch_tensor(m, target_bsz)
            if m.shape[1] > target_seqlen:
                m = m[:, :target_seqlen]
            elif m.shape[1] < target_seqlen:
                pad = torch.zeros(
                    (m.shape[0], target_seqlen - m.shape[1]),
                    dtype=m.dtype,
                    device=m.device,
                )
                m = torch.cat([m, pad], dim=1)
        return m.to(dtype=torch.bool)


    if timing_log is None:
        timing_log = {}

    if reweight and glmi_policy is None:
        analysis_start = time.perf_counter()
        model.to_cuda()
        print("Save gradient...")
        # save gradient
        grad_cache = GradCacheHook(vis_masks=vision_mask, cap_masks=caption_mask)        
        grad_cache.register_hooks(layers=layers)
        
        with torch.enable_grad():
            mini_batch = 1
            total_samples = next(iter(prompt_inputs.values())).shape[0]
            if mbq_debug_grad_samples and int(mbq_debug_grad_samples) > 0:
                total_samples = min(total_samples, int(mbq_debug_grad_samples))
                print(f"[MBQ][debug] limiting reweight gradient samples to {total_samples}")
            accum_steps = int(total_samples/mini_batch)
            
            for i in tqdm.tqdm(range(0, total_samples, mini_batch), desc="Running gradient calculation..."):
                mini_inputs = {}
                for k in inputs:
                    if isinstance(inputs[k], torch.Tensor):
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
    elif glmi_policy is not None:
        print("[GLMI] skip MBQ gradient-based reweight stage; using quantization-effect policy instead.")


    if distort:
        # assert wa_quant, "We only support distort input in weight-activation quantization!!!"
        print("Use distort input...")
        inps_distort = copy.deepcopy(inps)

    gc.collect()
    torch.cuda.empty_cache()

    # solve layer by layer
    max_layers = len(layers)
    if mbq_debug_max_layers and int(mbq_debug_max_layers) > 0:
        max_layers = min(max_layers, int(mbq_debug_max_layers))
        print(f"[MBQ][debug] limiting MBQ layer loop to first {max_layers} layers")

    solver_seconds = 0.0
    for i in tqdm.tqdm(range(max_layers), desc="Running MBQ..."):
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
        # get output as next layer's input

        cur_kwargs = {}
        for k, v in base_kwargs.items():
            if isinstance(v, torch.Tensor):
                vv = v.to(next(layer.parameters()).device)
                # Keep batch-aligned kwargs with current layer input.
                if vv.dim() >= 1:
                    vv = _align_batch_tensor(vv, inps.shape[0])
                cur_kwargs[k] = vv
            elif isinstance(v, tuple) and len(v) == 2 and torch.is_tensor(v[0]) and torch.is_tensor(v[1]):
                v0 = v[0].to(next(layer.parameters()).device)
                v1 = v[1].to(next(layer.parameters()).device)
                if v0.dim() >= 1:
                    v0 = _align_batch_tensor(v0, inps.shape[0])
                if v1.dim() >= 1:
                    v1 = _align_batch_tensor(v1, inps.shape[0])
                cur_kwargs[k] = (v0, v1)
            else:
                cur_kwargs[k] = v

        cur_kwargs = _sanitize_pos_emb(cur_kwargs, inps, layer)

        inps = layer(inps, **cur_kwargs)[0]
        for h in handles:
            h.remove()
        # now solve for scaling
        input_feat = {k: torch.cat(v, dim=0) for k, v in input_feat.items()}

        # Clear GPU memory
        torch.cuda.empty_cache()

        layer_glmi_policy = None
        if isinstance(glmi_policy, dict):
            layer_glmi_policy = glmi_policy.get("layers", {}).get(str(i))

        search_cfg = None
        if layer_glmi_policy is not None:
            branch_policy = layer_glmi_policy.get("search_policy")
            if not isinstance(branch_policy, dict):
                branch_policy = layer_glmi_policy.get("activation_policy" if wa_quant else "weight_policy", {})
            if not isinstance(branch_policy, dict):
                branch_policy = {}
            scale_reweight_ratio_dict = {
                "attn": float(layer_glmi_policy.get("reweight_ratio", 1.0)),
                "mlp": float(layer_glmi_policy.get("reweight_ratio", 1.0)),
            }
            search_cfg = {
                "n_grid": int(branch_policy.get("n_grid", layer_glmi_policy.get("n_grid", 20))),
                "ratio_max": float(branch_policy.get("ratio_max", layer_glmi_policy.get("ratio_max", 1.0))),
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
            if not reweight and layer_glmi_policy is None:
                ans_mask = None
                vis_mask = None
            else:
                ans_mask = _align_token_mask(
                    caption_mask,
                    target_bsz=inps.shape[0],
                    target_seqlen=inps.shape[1],
                    device=inps.device,
                )
                vis_mask = _align_token_mask(
                    vision_mask,
                    target_bsz=inps.shape[0],
                    target_seqlen=inps.shape[1],
                    device=inps.device,
                )
            
            if wa_quant:
                if distort:
                    solver_start = time.perf_counter()
                    scales_list = auto_scale_block_wa_distort(
                        layer,
                        cur_kwargs,
                        w_bit=w_bit,
                        a_bit=a_bit,
                        q_config=q_config,
                        input_feat=input_feat,
                        ans_mask=ans_mask,
                        vis_mask=vis_mask,
                        reweight_ratio_dict=scale_reweight_ratio_dict,
                        q_input=inps_distort,
                        loss_mode=loss_mode,
                        search_cfg=search_cfg,
                    )
                    solver_seconds += time.perf_counter() - solver_start
                else:
                    solver_start = time.perf_counter()
                    scales_list = auto_scale_block_wa(
                        layer,
                        cur_kwargs,
                        w_bit=w_bit,
                        a_bit=a_bit,
                        q_config=q_config,
                        input_feat=input_feat,
                        ans_mask=ans_mask,
                        vis_mask=vis_mask,
                        reweight_ratio_dict=scale_reweight_ratio_dict,
                        loss_mode=loss_mode,
                        search_cfg=search_cfg,
                    )
                    solver_seconds += time.perf_counter() - solver_start
            else:
                if distort:
                    solver_start = time.perf_counter()
                    scales_list = auto_scale_block_distort(
                        layer,
                        cur_kwargs,
                        w_bit=w_bit,
                        q_config=q_config,
                        input_feat=input_feat,
                        ans_mask=ans_mask,
                        vis_mask=vis_mask,
                        reweight_ratio_dict=scale_reweight_ratio_dict,
                        q_input=inps_distort,
                        loss_mode=loss_mode,
                        search_cfg=search_cfg,
                    )
                    solver_seconds += time.perf_counter() - solver_start
                else:
                    solver_start = time.perf_counter()
                    scales_list = auto_scale_block(
                        layer,
                        cur_kwargs,
                        w_bit=w_bit,
                        q_config=q_config,
                        input_feat=input_feat,
                        ans_mask=ans_mask,
                        vis_mask=vis_mask,
                        reweight_ratio_dict=scale_reweight_ratio_dict,
                        loss_mode=loss_mode,
                        search_cfg=search_cfg,
                    )
                    solver_seconds += time.perf_counter() - solver_start

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
                    
                    inps_distort = inps_distort.to(next(layer_q.parameters()).device)  # in case multi-gpu
                    if inps_distort.dim() == 2:
                        inps_distort = inps_distort.unsqueeze(0)
                    dist_kwargs = {}
                    for k, v in cur_kwargs.items():
                        if isinstance(v, torch.Tensor):
                            dist_kwargs[k] = v.to(next(layer_q.parameters()).device)
                        elif isinstance(v, tuple) and len(v) == 2 and torch.is_tensor(v[0]) and torch.is_tensor(v[1]):
                            dist_kwargs[k] = (
                                v[0].to(next(layer_q.parameters()).device),
                                v[1].to(next(layer_q.parameters()).device),
                            )
                        else:
                            dist_kwargs[k] = v
                    dist_kwargs = _sanitize_pos_emb(dist_kwargs, inps_distort, layer_q)
                    inps_distort = layer_q(inps_distort, **dist_kwargs)[0]
                    del layer_q 
                else:
                    layer_q = copy.deepcopy(layer)
                    layer_q = layer_q.cuda()
                    named_linears_q = get_named_linears(layer_q)
                    for n, m in named_linears_q.items():
                        m.weight.data = pseudo_quantize_tensor(m.weight.data, n_bits=w_bit, **q_config)
                        torch.cuda.empty_cache()
                    
                    inps_distort = inps_distort.to(next(layer_q.parameters()).device)  # in case multi-gpu
                    if inps_distort.dim() == 2:
                        inps_distort = inps_distort.unsqueeze(0)
                    dist_kwargs = {}
                    for k, v in cur_kwargs.items():
                        if isinstance(v, torch.Tensor):
                            dist_kwargs[k] = v.to(next(layer_q.parameters()).device)
                        elif isinstance(v, tuple) and len(v) == 2 and torch.is_tensor(v[0]) and torch.is_tensor(v[1]):
                            dist_kwargs[k] = (
                                v[0].to(next(layer_q.parameters()).device),
                                v[1].to(next(layer_q.parameters()).device),
                            )
                        else:
                            dist_kwargs[k] = v
                    dist_kwargs = _sanitize_pos_emb(dist_kwargs, inps_distort, layer_q)
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

    timing_log["search_seconds"] = float(timing_log.get("search_seconds", 0.0)) + float(solver_seconds)
    return mbq_results


def apply_mbq(model, mbq_results):
    apply_scale(model, mbq_results["scale"])
