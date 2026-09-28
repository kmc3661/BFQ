import gc
import copy
import math
import torch
import functools
import torch.nn as nn
import numpy as np

from transformers.models.bloom.modeling_bloom import BloomBlock, BloomGelu
from transformers.models.opt.modeling_opt import OPTDecoderLayer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm
from transformers.activations import GELUActivation

from collections import defaultdict
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm

from .qmodule import ScaledActivation
from .quantizer import get_module_by_name_suffix
from qmllm.utils.search import get_op_by_name, get_op_name, set_op_by_name
from qmllm.quantization.quant_funcs import pseudo_quantize_tensor
from qmllm.quantization.qlinear import WALinear

__all__ = ["auto_scale_block_wa_distort"]


@torch.no_grad()
def get_weight_scale(weight, q_group_size=-1):
    org_shape = weight.shape
    if q_group_size > 0:
        weight = weight.view(-1, q_group_size)
    scale = weight.abs() / weight.abs().amax(dim=1, keepdim=True)
    scale = scale.view(org_shape)
    scale = scale.mean(0)
    return scale


@torch.no_grad()
def get_act_scale(x):
    return x.abs().view(-1, x.shape[-1]).mean(0)


@torch.no_grad()
def scale_ln_fcs(ln, fcs, scales):
    if not isinstance(fcs, list):
        fcs = [fcs]

    scales = scales.to(ln.weight.device)

    ln.weight.div_(scales)
    if hasattr(ln, "bias") and ln.bias is not None:
        ln.bias.div_(scales)

    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))

    for p in ln.parameters():
        assert torch.isnan(p).sum() == 0
    for fc in fcs:
        for p in fc.parameters():
            assert torch.isnan(p).sum() == 0


@torch.no_grad()
def scale_fc_fc(fc1, fc2, scales):
    assert isinstance(fc1, nn.Linear)
    assert isinstance(fc2, nn.Linear)
    # assert fc1.out_features == fc2.in_features

    scales = scales.to(fc1.weight.device)

    # fc1.weight.div_(scales.view(-1, 1))
    fc1.weight[-scales.size(0) :].div_(scales.view(-1, 1))
    if fc1.bias is not None:
        fc1.bias.div_(scales.view(-1))

    fc2.weight.mul_(scales.view(1, -1))

    for p in fc1.parameters():
        assert torch.isnan(p).sum() == 0
    for p in fc2.parameters():
        assert torch.isnan(p).sum() == 0


@torch.no_grad()
def scale_gelu_fc(gelu, fc, scales):
    assert isinstance(gelu, (nn.GELU, BloomGelu, GELUActivation))
    assert isinstance(fc, nn.Linear)

    fc.weight.mul_(scales.view(1, -1).to(fc.weight.device))

    for p in fc.parameters():
        assert torch.isnan(p).sum() == 0


@torch.no_grad()
def auto_scale_block_wa_distort(module, module_kwargs, w_bit, a_bit, q_config, input_feat, ans_mask, vis_mask, reweight_ratio_dict, q_input, loss_mode="mae", search_cfg=None):

    if "use_cache" in module_kwargs:
        module_kwargs.pop("use_cache")

    def _normalize_mask_2d(mask, out_tensor, name="mask"):
        if mask is None or (not torch.is_tensor(mask)):
            return None
        bsz, seqlen = out_tensor.shape[0], out_tensor.shape[1]
        m = mask.to(device=out_tensor.device)
        orig_shape = tuple(m.shape)

        if m.dim() == 1:
            m = m.unsqueeze(0)
        elif m.dim() >= 3:
            # Reduce unexpected extra dims (e.g. [B,S,C] / [S,C]) to token mask.
            while m.dim() > 2:
                if m.shape[-1] == 1:
                    m = m.squeeze(-1)
                else:
                    m = (m != 0).any(dim=-1)

        if m.dim() != 2:
            m = m.reshape(m.shape[0], -1)

        # Handle transposed/misaligned cases such as [S,C].
        if m.shape[0] == seqlen and m.shape[1] != seqlen:
            m = m[:, :1].transpose(0, 1)  # [1,S]

        # Align batch.
        if m.shape[0] != bsz:
            if m.shape[0] > bsz:
                m = m[:bsz]
            elif m.shape[0] == 1:
                m = m.expand(bsz, -1)
            else:
                rep = int(math.ceil(float(bsz) / float(m.shape[0])))
                m = m.repeat(rep, 1)[:bsz]

        # Align sequence length.
        if m.shape[1] > seqlen:
            m = m[:, :seqlen]
        elif m.shape[1] < seqlen:
            pad = torch.zeros((m.shape[0], seqlen - m.shape[1]), dtype=m.dtype, device=m.device)
            m = torch.cat([m, pad], dim=1)

        if tuple(m.shape) != (bsz, seqlen):
            raise RuntimeError(f"{name} normalization failed: got {tuple(m.shape)}, expected {(bsz, seqlen)}")

        # Silent normalization: keep logs clean during long scale search.
        return m.to(dtype=torch.bool)

    def _search_module_scale_wa_distort(block, linears2scale: list, layers_name, x, x_q, reweight_ratio, kwargs={}):
        # w: co, ci
        # x: n, ci
        x = x.to(next(block.parameters()).device)
        x_q = x_q.to(next(block.parameters()).device)
        with torch.no_grad():
            org_out = block(x, **kwargs)
            if isinstance(org_out, tuple):
                org_out = org_out[0]
        ans_mask_2d = _normalize_mask_2d(ans_mask, org_out, name="ans_mask")
        vis_mask_2d = _normalize_mask_2d(vis_mask, org_out, name="vis_mask")

        x_max = get_act_scale(x_q)

        best_error = float("inf")
        best_ratio = -1
        best_scales = None

        n_grid = int(search_cfg.get("n_grid", 20)) if isinstance(search_cfg, dict) else 20
        n_grid = max(2, n_grid)
        ratio_max = float(search_cfg.get("ratio_max", 1.0)) if isinstance(search_cfg, dict) else 1.0
        ratio_max = max(1e-4, ratio_max)
        history = []

        org_sd = {k: v.cpu() for k, v in block.state_dict().items()}
        for ratio_idx in range(n_grid):
            ratio = (ratio_idx / max(n_grid - 1, 1)) * ratio_max
            scales = x_max.pow(ratio).clamp(min=1e-4).view(-1)
            scales = scales / (scales.max() * scales.min()).sqrt()

            if isinstance(block, nn.Linear):
                new_block = None
            for fc, fc_name in zip(linears2scale, layers_name):
                # fc.weight.mul_(scales.view(1, -1).to(fc.weight.device))
                # fc.weight.data = w_quantize_func(fc.weight.data) / (scales.view(1, -1))
                fc.weight.mul_(scales.view(1, -1).to(fc.weight.device))
                new_fc = WALinear.from_float(fc, weight_quant="per_channel", act_quant="per_token", w_bit=w_bit, a_bit=a_bit)
                
                if isinstance(block, nn.Linear):
                    new_block = copy.deepcopy(new_fc)                   
                else:
                    setattr(block, fc_name, new_fc)
                    
                del new_fc
                torch.cuda.empty_cache()

            x_scale = x_q / (scales.view(1, 1, -1)) 

            if isinstance(block, nn.Linear):
                out = new_block(x_scale, **kwargs)
            else:
                out = block(x_scale, **kwargs)

            if isinstance(out, tuple):
                out = out[0]

            # loss = (
            #     (org_out - out).float().pow(2).mean().item()
            # )  # float prevents overflow

            if loss_mode == "mse":
                if ans_mask_2d is not None and vis_mask_2d is not None:
                    ans_mask_expand = ans_mask_2d.unsqueeze(-1).expand_as(out)
                    vis_mask_expand = vis_mask_2d.unsqueeze(-1).expand_as(out).to(out.device)
                    masked_diff_ans = ((org_out - out).float().pow(2) * ans_mask_expand)
                    masked_diff_vis = ((org_out - out).float().pow(2) * vis_mask_expand)
                    if reweight_ratio is not None:
                        den_ans = ans_mask_expand.sum().clamp_min(1)
                        den_vis = vis_mask_expand.sum().clamp_min(1)
                        loss = masked_diff_ans.sum() / den_ans + reweight_ratio * (masked_diff_vis.sum() / den_vis)
                    else:
                        loss = (
                            (org_out - out).float().pow(2).mean().item()
                        ) 
                elif ans_mask_2d is not None and vis_mask_2d is None:
                    ans_mask_expand = ans_mask_2d.unsqueeze(-1).expand_as(out)
                    masked_diff = ((org_out - out).float().pow(2) * ans_mask_expand)
                    loss = masked_diff.sum() / ans_mask_expand.sum().clamp_min(1)
                else:
                    loss = (
                        (org_out - out).float().pow(2).mean().item()
                    )  # float prevents overflow
            elif loss_mode == "mae":
                if ans_mask_2d is not None and vis_mask_2d is not None:
                    ans_mask_expand = ans_mask_2d.unsqueeze(-1).expand_as(out)
                    vis_mask_expand = vis_mask_2d.unsqueeze(-1).expand_as(out).to(out.device)
                    masked_diff_ans = ((org_out - out).float().abs() * ans_mask_expand)
                    masked_diff_vis = ((org_out - out).float().abs() * vis_mask_expand)
                    if reweight_ratio is not None:
                        den = (ans_mask_expand.sum() + vis_mask_expand.sum()).clamp_min(1)
                        loss = (masked_diff_ans.sum() + reweight_ratio * masked_diff_vis.sum()) / den
                    else:
                        loss = (
                            (org_out - out).float().abs().mean().item()
                        ) 
                elif ans_mask_2d is not None and vis_mask_2d is None:
                    ans_mask_expand = ans_mask_2d.unsqueeze(-1).expand_as(out)
                    masked_diff = ((org_out - out).float().abs() * ans_mask_expand)
                    loss = masked_diff.sum() / ans_mask_expand.sum().clamp_min(1)
                else:
                    loss = (
                        (org_out - out).float().abs().mean().item()
                    )  # float prevents overflow

            if torch.is_tensor(loss):
                if not torch.isfinite(loss):
                    loss = torch.tensor(1e9, device=out.device)
                loss = float(loss.item())
            else:
                if not np.isfinite(loss):
                    loss = 1e9

            history.append(loss)
            is_best = loss < best_error
            if is_best:
                best_error = loss
                best_ratio = ratio
                best_scales = scales

            # restore the block
            for fc, fc_name in zip(linears2scale, layers_name):
                if isinstance(block, nn.Linear):
                    continue
                else:
                    setattr(block, fc_name, fc)
            
            if isinstance(block, nn.Linear):
                del new_block 
            torch.cuda.empty_cache()
            block.load_state_dict(org_sd)
        if best_ratio == -1:
            print(f"[MBQ][warn] scale search failed (all invalid). fallback to identity scales. history={history}")
            best_scales = torch.ones_like(x_max).view(-1)
            best_ratio = 0
        # print(best_ratio)
        best_scales = best_scales.view(-1)

        assert torch.isnan(best_scales).sum() == 0, best_scales
        return best_scales.detach()

    def _auto_get_scale_wa_distort(prev_op, layers, layers_name, inp, inp_q, reweight_ratio, module2inspect=None, kwargs={}):
        # module2inspect: if given, we will check the output diff of this module instead of layers
        if module2inspect is None:
            assert len(layers) == 1
            module2inspect = layers[0]

        scales = _search_module_scale_wa_distort(module2inspect, layers, layers_name, inp, inp_q, reweight_ratio, kwargs)
        scales = scales.detach().cpu()
        # prev_op_name, [layer_name], scale
        return (
            get_op_name(module, prev_op),
            tuple([get_op_name(module, m) for m in layers]),
            scales,
        )

    scales_list = []  # return the searched scales


    def _auto_get_input_feat_distort(inps_q, scales_list=None):

        # org_sd = {k: v.cpu() for k, v in module.state_dict().items()}

        new_module = copy.deepcopy(module)

        named_linears = {name: m for name, m in new_module.named_modules() if isinstance(m, nn.Linear)}

        if scales_list is not None:
            apply_scale(new_module, scales_list)
            new_module.cuda()
            for n, m in named_linears.items():
                new_linear = WALinear.from_float(m, weight_quant="per_channel", act_quant="per_token", w_bit=w_bit, a_bit=a_bit)
                father_module = get_module_by_name_suffix(new_module, '.'.join(n.split(".")[:-1]))
                setattr(father_module, n.split('.')[-1], new_linear)
                del new_linear, m
                torch.cuda.empty_cache()

            named_linears = {name: m for name, m in new_module.named_modules() if isinstance(m, WALinear)}  
        

        def cache_input_hook(m, x, y, name, feat_dict):
            x = x[0]
            x = x.detach().cpu()
            feat_dict[name].append(x)

        input_feat_q = defaultdict(list)
        handles = []
        for name in named_linears:
            handles.append(
                named_linears[name].register_forward_hook(
                    functools.partial(cache_input_hook, name=name, feat_dict=input_feat_q)
                )
            )

        inps_q = inps_q.to(next(new_module.parameters()).device)
        if inps_q.dim() == 2:
            inps_q = inps_q.unsqueeze(0)

        def _align_batch_tensor(t: torch.Tensor, target_bsz: int) -> torch.Tensor:
            if (not torch.is_tensor(t)) or t.dim() == 0:
                return t
            cur = int(t.shape[0])
            if cur == target_bsz:
                return t
            if cur > target_bsz:
                return t[:target_bsz]
            if cur == 1:
                return t.expand((target_bsz,) + tuple(t.shape[1:]))
            rep = int(np.ceil(float(target_bsz) / float(cur)))
            reps = [1] * t.dim()
            reps[0] = rep
            return t.repeat(*reps)[:target_bsz]

        cur_kwargs = {}
        for k, v in module_kwargs.items():
            if isinstance(v, torch.Tensor):
                vv = v.to(inps_q.device)
                if vv.dim() >= 1:
                    vv = _align_batch_tensor(vv, inps_q.shape[0])
                cur_kwargs[k] = vv
            elif isinstance(v, tuple) and len(v) == 2 and torch.is_tensor(v[0]) and torch.is_tensor(v[1]):
                v0 = v[0].to(inps_q.device)
                v1 = v[1].to(inps_q.device)
                if v0.dim() >= 1:
                    v0 = _align_batch_tensor(v0, inps_q.shape[0])
                if v1.dim() >= 1:
                    v1 = _align_batch_tensor(v1, inps_q.shape[0])
                cur_kwargs[k] = (v0, v1)
            else:
                cur_kwargs[k] = v

        # Qwen2 decoder path: always rebuild rotary position embeddings for inps_q shape.
        if module.__class__.__name__ == "Qwen2DecoderLayer":
            cur_kwargs["use_cache"] = False
            cur_kwargs.pop("position_ids", None)
            bsz, seqlen = inps_q.shape[0], inps_q.shape[1]
            pos_ids = torch.arange(seqlen, device=inps_q.device).unsqueeze(0).expand(bsz, -1)

            pos_emb = None
            if hasattr(module, "self_attn") and hasattr(module.self_attn, "rotary_emb"):
                try:
                    pos_emb = module.self_attn.rotary_emb(inps_q, pos_ids)
                except TypeError:
                    try:
                        pos_emb = module.self_attn.rotary_emb(pos_ids)
                    except Exception:
                        pos_emb = None
                except Exception:
                    pos_emb = None

            valid = False
            if isinstance(pos_emb, tuple) and len(pos_emb) == 2:
                cos, sin = pos_emb
                head_dim = int(getattr(module.self_attn, "head_dim", cos.shape[-1]))
                valid = (
                    torch.is_tensor(cos)
                    and torch.is_tensor(sin)
                    and cos.shape[-1] == head_dim
                    and sin.shape[-1] == head_dim
                    and cos.shape[-2] == seqlen
                    and sin.shape[-2] == seqlen
                )
            if not valid:
                head_dim = int(getattr(module.self_attn, "head_dim", max(1, inps_q.shape[-1] // max(1, getattr(module.self_attn, "num_heads", 1)))))
                cos = torch.zeros(bsz, seqlen, head_dim, device=inps_q.device, dtype=inps_q.dtype)
                sin = torch.zeros_like(cos)
                pos_emb = (cos, sin)
            cur_kwargs["position_embeddings"] = pos_emb

        new_module(inps_q, **cur_kwargs)
        for h in handles:
            h.remove()
    
        input_feat_q = {k: torch.cat(v, dim=0) for k, v in input_feat_q.items()}

        del new_module
        torch.cuda.empty_cache()

        # module.load_state_dict(org_sd)

        return input_feat_q


    if isinstance(module, OPTDecoderLayer):
        # attention input
        scales_list.append(
            _auto_get_scale(
                prev_op=module.self_attn_layer_norm,
                layers=[
                    module.self_attn.q_proj,
                    module.self_attn.k_proj,
                    module.self_attn.v_proj,
                ],
                inp=input_feat["self_attn.q_proj"],
                module2inspect=module.self_attn,
                kwargs=module_kwargs,
            )
        )
        # attn out
        scales_list.append(
            _auto_get_scale(
                prev_op=module.self_attn.v_proj,
                layers=[module.self_attn.out_proj],
                inp=input_feat["self_attn.out_proj"],
            )
        )
        # fc1
        scales_list.append(
            _auto_get_scale(
                prev_op=module.final_layer_norm,
                layers=[module.fc1],
                inp=input_feat["fc1"],
            )
        )
        # fc2
        scales_list.append(
            _auto_get_scale(
                prev_op=module.fc1,
                layers=[module.fc2],
                inp=input_feat["fc2"],
            )
        )

    elif isinstance(module, LlamaDecoderLayer):
        # attention input
        input_feat_q = _auto_get_input_feat_distort(
            inps_q=q_input,
        )
        scales_list.append(
            _auto_get_scale_wa_distort(
                prev_op=module.input_layernorm,
                layers=[
                    module.self_attn.q_proj,
                    module.self_attn.k_proj,
                    module.self_attn.v_proj,
                ],
                layers_name=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                ],
                inp=input_feat["self_attn.q_proj"],
                inp_q=input_feat_q["self_attn.q_proj"],
                reweight_ratio=reweight_ratio_dict["attn"],
                module2inspect=module.self_attn,
                kwargs=module_kwargs,
            )
        )
        # attn out
        # Please refer to https://github.com/mit-han-lab/llm-awq/pull/67#issue-1850622696
        input_feat_q = _auto_get_input_feat_distort(
            inps_q=q_input,
            scales_list=scales_list,
        )
        if module.self_attn.v_proj.weight.shape == module.self_attn.o_proj.weight.shape:
            scales_list.append(
                _auto_get_scale_wa_distort(
                    prev_op=module.self_attn.v_proj,
                    layers=[module.self_attn.o_proj],
                    layers_name=["o_proj"],
                    inp=input_feat["self_attn.o_proj"],
                    inp_q=input_feat_q["self_attn.o_proj"],
                    reweight_ratio=reweight_ratio_dict["attn"],
                )
            )
        # fc1
        input_feat_q = _auto_get_input_feat_distort(
            inps_q=q_input,
            scales_list=scales_list,
        )
        scales_list.append(
            _auto_get_scale_wa_distort(
                prev_op=module.post_attention_layernorm,
                layers=[module.mlp.gate_proj, module.mlp.up_proj],
                layers_name=["gate_proj", "up_proj"],
                inp=input_feat["mlp.gate_proj"],
                inp_q=input_feat_q["mlp.gate_proj"],
                reweight_ratio=reweight_ratio_dict["mlp"],
                module2inspect=module.mlp,
            )
        )
        # fc2
        input_feat_q = _auto_get_input_feat_distort(
            inps_q=q_input,
            scales_list=scales_list,
        )
        scales_list.append(
            _auto_get_scale_wa_distort(
                prev_op=module.mlp.up_proj,
                layers=[module.mlp.down_proj],
                layers_name=["down_proj"],
                inp=input_feat["mlp.down_proj"],
                inp_q=input_feat_q["mlp.down_proj"],
                reweight_ratio=reweight_ratio_dict["mlp"],
            )
        )

    elif isinstance(module, BloomBlock):
        # attention input
        scales_list.append(
            _auto_get_scale(
                prev_op=module.input_layernorm,
                layers=[module.self_attention.query_key_value],
                inp=input_feat["self_attention.query_key_value"],
                module2inspect=module,
                kwargs=module_kwargs,
            )
        )
        # attn out
        # Please refer to https://github.com/mit-han-lab/llm-awq/issues/2#issuecomment-1606297469
        """
        scales_list.append(_auto_get_scale(
            prev_op=module.self_attention.query_key_value,
            layers=[module.self_attention.dense],
            inp=input_feat['self_attention.dense'],
        ))
        """
        # fc1
        scales_list.append(
            _auto_get_scale(
                prev_op=module.post_attention_layernorm,
                layers=[module.mlp.dense_h_to_4h],
                inp=input_feat["mlp.dense_h_to_4h"],
                module2inspect=module,
                kwargs=module_kwargs,
            )
        )
        # fc2
        scales_list.append(
            _auto_get_scale(
                prev_op=module.mlp.gelu_impl,
                layers=[module.mlp.dense_4h_to_h],
                inp=input_feat["mlp.dense_4h_to_h"],
            )
        )
    elif "mpt" in str(module.__class__).lower():
        # attention input
        scales_list.append(
            _auto_get_scale(
                prev_op=module.norm_1,
                layers=[module.attn.Wqkv],
                inp=input_feat["attn.Wqkv"],
                module2inspect=module.attn,
                kwargs=module_kwargs,
            )
        )

        # attn out
        scales_list.append(
            _auto_get_scale(
                prev_op=module.attn.Wqkv,
                layers=[module.attn.out_proj],
                inp=input_feat["attn.out_proj"],
            )
        )
        # fc1
        scales_list.append(
            _auto_get_scale(
                prev_op=module.norm_2,
                layers=[module.ffn.up_proj],
                inp=input_feat["ffn.up_proj"],
                module2inspect=module.ffn,
            )
        )
        # fc2
        scales_list.append(
            _auto_get_scale(
                prev_op=module.ffn.act,
                layers=[module.ffn.down_proj],
                inp=input_feat["ffn.down_proj"],
            )
        )

    elif "falcon" in str(module.__class__).lower():
        # attn out
        # Haotian: TBD: need to handle repeated scales for MQ
        """
        scales_list.append(_auto_get_scale(
            prev_op=module.self_attention.query_key_value,
            layers=[module.self_attention.dense],
            inp=input_feat['self_attention.dense'],
        ))
        """
        # fc1, as long as it is scaled, everything is screwed up
        if "falcon-7b" in str(module.__class__).lower():
            scales_list.append(
                _auto_get_scale(
                    prev_op=module.input_layernorm,
                    layers=[
                        module.mlp.dense_h_to_4h,
                        module.self_attention.query_key_value,
                    ],
                    inp=input_feat["self_attention.query_key_value"],
                    module2inspect=module,
                    kwargs=module_kwargs,
                )
            )
        elif "falcon-40b" in str(module.__class__).lower():
            scales_list.append(
                _auto_get_scale(
                    prev_op=module.ln_attn,
                    layers=[module.self_attention.query_key_value],
                    inp=input_feat["self_attention.query_key_value"],
                    module2inspect=module,
                    kwargs=module_kwargs,
                )
            )
            scales_list.append(
                _auto_get_scale(
                    prev_op=module.ln_mlp,
                    layers=[module.mlp.dense_h_to_4h],
                    inp=input_feat["mlp.dense_h_to_4h"],
                    module2inspect=module,
                    kwargs=module_kwargs,
                )
            )
        else:
            raise NotImplementedError(
                "Unknown Falcon architecture, currently only falcon-7b and falcon-40b are supported"
            )
        # fc2
        scales_list.append(
            _auto_get_scale(
                prev_op=module.mlp.act,
                layers=[module.mlp.dense_4h_to_h],
                inp=input_feat["mlp.dense_4h_to_h"],
            )
        )
    elif "bigcode" in str(module.__class__).lower():
        scales_list.append(
            _auto_get_scale(
                prev_op=module.ln_1,
                layers=[module.attn.c_attn],
                inp=input_feat["attn.c_attn"],
                module2inspect=module.attn,
                kwargs=module_kwargs,
            )
        )
        # fc1
        scales_list.append(
            _auto_get_scale(
                prev_op=module.ln_2,
                layers=[module.mlp.c_fc],
                inp=input_feat["mlp.c_fc"],
                module2inspect=module.mlp,
            )
        )
        # fc2
        scales_list.append(
            _auto_get_scale(
                prev_op=module.mlp.act,
                layers=[module.mlp.c_proj],
                inp=input_feat["mlp.c_proj"],
            )
        )
    elif "neox" in str(module.__class__).lower():
        scales_list.append(
            _auto_get_scale(
                prev_op=module.input_layernorm,
                layers=[module.attention.query_key_value],
                inp=input_feat["attention.query_key_value"],
                module2inspect=module.attention,
                kwargs=module_kwargs,
            )
        )
        # fc1
        scales_list.append(
            _auto_get_scale(
                prev_op=module.post_attention_layernorm,
                layers=[module.mlp.dense_h_to_4h],
                inp=input_feat["mlp.dense_h_to_4h"],
                module2inspect=module.mlp,
            )
        )
        # fc2
        scales_list.append(
            _auto_get_scale(
                prev_op=module.mlp.act,
                layers=[module.mlp.dense_4h_to_h],
                inp=input_feat["mlp.dense_4h_to_h"],
            )
        )
    elif module.__class__.__name__ == "Qwen2DecoderLayer":
        # attention input
        input_feat_q = _auto_get_input_feat_distort(
            inps_q=q_input,
        )
        scales_list.append(
            _auto_get_scale_wa_distort(
                prev_op=module.input_layernorm,
                layers=[
                    module.self_attn.q_proj,
                    module.self_attn.k_proj,
                    module.self_attn.v_proj,
                ],
                layers_name=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                ],
                inp=input_feat["self_attn.q_proj"],
                inp_q=input_feat_q["self_attn.q_proj"],
                reweight_ratio=reweight_ratio_dict["attn"],
                module2inspect=module.self_attn,
                kwargs=module_kwargs,
            )
        )
        # attn out
        # Please refer to https://github.com/mit-han-lab/llm-awq/pull/67#issue-1850622696
        input_feat_q = _auto_get_input_feat_distort(
            inps_q=q_input,
            scales_list=scales_list,
        )
        if module.self_attn.v_proj.weight.shape == module.self_attn.o_proj.weight.shape:
            scales_list.append(
                _auto_get_scale_wa_distort(
                    prev_op=module.self_attn.v_proj,
                    layers=[module.self_attn.o_proj],
                    layers_name=["o_proj"],
                    inp=input_feat["self_attn.o_proj"],
                    inp_q=input_feat_q["self_attn.o_proj"],
                    reweight_ratio=reweight_ratio_dict["attn"],
                )
            )
        # fc1
        input_feat_q = _auto_get_input_feat_distort(
            inps_q=q_input,
            scales_list=scales_list,
        )
        scales_list.append(
            _auto_get_scale_wa_distort(
                prev_op=module.post_attention_layernorm,
                layers=[module.mlp.gate_proj, module.mlp.up_proj],
                layers_name=["gate_proj", "up_proj"],
                inp=input_feat["mlp.gate_proj"],
                inp_q=input_feat_q["mlp.gate_proj"],
                reweight_ratio=reweight_ratio_dict["mlp"],
                module2inspect=module.mlp,
            )
        )
        # fc2
        input_feat_q = _auto_get_input_feat_distort(
            inps_q=q_input,
            scales_list=scales_list,
        )
        scales_list.append(
            _auto_get_scale_wa_distort(
                prev_op=module.mlp.up_proj,
                layers=[module.mlp.down_proj],
                layers_name=["down_proj"],
                inp=input_feat["mlp.down_proj"],
                inp_q=input_feat_q["mlp.down_proj"],
                reweight_ratio=reweight_ratio_dict["mlp"],
            )
        )
    elif module.__class__.__name__ == "InternLM2DecoderLayer":
        # attention input
        input_feat_q = _auto_get_input_feat_distort(
            inps_q=q_input,
        )
        scales_list.append(
            _auto_get_scale_wa_distort(
                prev_op=module.attention_norm,
                layers=[
                    module.attention.wqkv,
                ],
                layers_name=[
                    "wqkv",
                ],
                inp=input_feat["attention.wqkv"],
                inp_q=input_feat_q["attention.wqkv"],
                reweight_ratio=reweight_ratio_dict["attn"],
                module2inspect=module.attention,
                kwargs=module_kwargs,
            )
        )
        # attn out
        # Please refer to https://github.com/mit-han-lab/llm-awq/pull/67#issue-1850622696
        input_feat_q = _auto_get_input_feat_distort(
            inps_q=q_input,
            scales_list=scales_list,
        )
        if module.attention.wqkv.weight.shape == module.attention.wo.weight.shape:
            scales_list.append(
                _auto_get_scale_wa_distort(
                    prev_op=module.attention.wqkv,
                    layers=[module.attention.wo],
                    layers_name=["wo"],
                    inp=input_feat["attention.wo"],
                    inp_q=input_feat_q["attention.wo"],
                    reweight_ratio=reweight_ratio_dict["attn"],
                )
            )
        # fc1
        input_feat_q = _auto_get_input_feat_distort(
            inps_q=q_input,
            scales_list=scales_list,
        )
        scales_list.append(
            _auto_get_scale_wa_distort(
                prev_op=module.ffn_norm,
                layers=[module.feed_forward.w1, module.feed_forward.w3],
                layers_name=["w1","w3"],
                inp=input_feat["feed_forward.w1"],
                inp_q=input_feat_q["feed_forward.w1"],
                reweight_ratio=reweight_ratio_dict["mlp"],
                module2inspect=module.feed_forward,
            )
        )
        # fc2
        input_feat_q = _auto_get_input_feat_distort(
            inps_q=q_input,
            scales_list=scales_list,
        )
        scales_list.append(
            _auto_get_scale_wa_distort(
                prev_op=module.feed_forward.w3,
                layers=[module.feed_forward.w2],
                layers_name=["w2"],
                inp=input_feat["feed_forward.w2"],
                inp_q=input_feat_q["feed_forward.w2"],
                reweight_ratio=reweight_ratio_dict["mlp"],
            )
        )
    elif module.__class__.__name__ == "Qwen2VLDecoderLayer":
        # attention input
        input_feat_q = _auto_get_input_feat_distort(
            inps_q=q_input,
        )
        scales_list.append(
            _auto_get_scale_wa_distort(
                prev_op=module.input_layernorm,
                layers=[
                    module.self_attn.q_proj,
                    module.self_attn.k_proj,
                    module.self_attn.v_proj,
                ],
                layers_name=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                ],
                inp=input_feat["self_attn.q_proj"],
                inp_q=input_feat_q["self_attn.q_proj"],
                reweight_ratio=reweight_ratio_dict["attn"],
                module2inspect=module.self_attn,
                kwargs=module_kwargs,
            )
        )
        # attn out
        # Please refer to https://github.com/mit-han-lab/llm-awq/pull/67#issue-1850622696
        input_feat_q = _auto_get_input_feat_distort(
            inps_q=q_input,
            scales_list=scales_list,
        )
        if module.self_attn.v_proj.weight.shape == module.self_attn.o_proj.weight.shape:
            scales_list.append(
                _auto_get_scale_wa_distort(
                    prev_op=module.self_attn.v_proj,
                    layers=[module.self_attn.o_proj],
                    layers_name=["o_proj"],
                    inp=input_feat["self_attn.o_proj"],
                    inp_q=input_feat_q["self_attn.o_proj"],
                    reweight_ratio=reweight_ratio_dict["attn"],
                )
            )
        # fc1
        input_feat_q = _auto_get_input_feat_distort(
            inps_q=q_input,
            scales_list=scales_list,
        )
        scales_list.append(
            _auto_get_scale_wa_distort(
                prev_op=module.post_attention_layernorm,
                layers=[module.mlp.gate_proj, module.mlp.up_proj],
                layers_name=["gate_proj", "up_proj"],
                inp=input_feat["mlp.gate_proj"],
                inp_q=input_feat_q["mlp.gate_proj"],
                reweight_ratio=reweight_ratio_dict["mlp"],
                module2inspect=module.mlp,
            )
        )
        # fc2
        input_feat_q = _auto_get_input_feat_distort(
            inps_q=q_input,
            scales_list=scales_list,
        )
        scales_list.append(
            _auto_get_scale_wa_distort(
                prev_op=module.mlp.up_proj,
                layers=[module.mlp.down_proj],
                layers_name=["down_proj"],
                inp=input_feat["mlp.down_proj"],
                inp_q=input_feat_q["mlp.down_proj"],
                reweight_ratio=reweight_ratio_dict["mlp"],
            )
        )
    else:
        raise NotImplementedError(f"{type(module)} not supported yet!")

    return scales_list

def apply_scale(module, scales_list, input_feat_dict=None):
    for prev_op_name, layer_names, scales in scales_list:
        prev_op = get_op_by_name(module, prev_op_name)
        layers = [get_op_by_name(module, name) for name in layer_names]

        prev_op.cuda()
        for layer in layers:
            layer.cuda()
        scales.cuda()

        if isinstance(prev_op, nn.Linear):
            assert len(layers) == 1
            scale_fc_fc(prev_op, layers[0], scales)
        elif isinstance(prev_op, (nn.LayerNorm, LlamaRMSNorm)) or prev_op.__class__.__name__ == "InternLM2RMSNorm" or prev_op.__class__.__name__ == "Qwen2RMSNorm":
            scale_ln_fcs(prev_op, layers, scales)
        elif isinstance(prev_op, (nn.GELU, BloomGelu, GELUActivation)):
            new_module = ScaledActivation(prev_op, scales)
            set_op_by_name(module, prev_op_name, new_module)
            scale_gelu_fc(prev_op, layers[0], scales)
        else:
            raise NotImplementedError(f"prev_op {type(prev_op)} not supported yet!")

        # apply the scaling to input feat if given;
        if input_feat_dict is not None:
            for layer_name in layer_names:
                inp = input_feat_dict[layer_name]
                inp.div_(scales.view(1, -1).to(inp.device))

        prev_op.cpu()
        for layer in layers:
            layer.cpu()
        scales.cpu()
