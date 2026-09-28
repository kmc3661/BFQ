import torch
import torch.nn as nn
from qmllm.quantization.quant_funcs import pseudo_quantize_tensor
from qmllm.quantization.qlinear import WALinear


__all__ = [
    "quantize_projector_weight",
    "quantize_projector_weight_act",
    "quantize_vision_encoder_weight",
]


# ============================================================
# 1) Vision Encoder Linear Layers Quantization
# ============================================================

def quantize_vision_encoder_weight(
    vision_module: nn.Module,
    w_bit: int,
    q_config: dict,
):
    """
    Siglip vision encoder 내부의 Linear weight만 양자화.
    """
    q_cfg = dict(q_config)
    gsize = q_cfg.get("q_group_size", -1)
    warned = False
    for name, m in vision_module.named_modules():
        if isinstance(m, nn.Linear):
            gs = gsize
            if gs is not None and gs > 0 and (m.weight.shape[-1] % gs != 0):
                gs = -1
                if not warned:
                    print(f"[MBQ][warn] Vision linear {name} in_features {m.weight.shape[-1]} not divisible by group size; fallback to per-tensor quant")
                    warned = True
            q_cfg["q_group_size"] = gs
            m.weight.data = pseudo_quantize_tensor(
                m.weight.data,
                n_bits=w_bit,
                **q_cfg
            )


# ============================================================
# 2) Projector Weight-Only Quantization
# ============================================================

def quantize_projector_weight(
    projector: nn.Module,
    w_bit: int,
    q_config: dict,
):
    """
    Gemma3MultiModalProjector 내부의 모든 Linear에 weight-only quant 적용.
    """
    q_cfg = dict(q_config)
    gsize = q_cfg.get("q_group_size", -1)
    warned = False
    for name, m in projector.named_modules():
        if isinstance(m, nn.Linear):
            gs = gsize
            if gs is not None and gs > 0 and (m.weight.shape[-1] % gs != 0):
                gs = -1
                if not warned:
                    print(f"[MBQ][warn] Projector linear {name} in_features {m.weight.shape[-1]} not divisible by group size; fallback to per-tensor quant")
                    warned = True
            q_cfg["q_group_size"] = gs
            m.weight.data = pseudo_quantize_tensor(
                m.weight.data,
                n_bits=w_bit,
                **q_cfg
            )


# ============================================================
# 3) Projector WA Quantization (WALinear 변환)
# ============================================================

def quantize_projector_weight_act(
    projector: nn.Module,
    w_bit: int,
    a_bit: int,
):
    """
    projector Linear을 WALinear로 변환.
    act_quant(per_token), weight_quant(per_channel)
    """
    for name, m in list(projector.named_modules()):
        if isinstance(m, nn.Linear):
            parent = _get_parent_module(projector, name)
            if parent is None:
                continue

            new_m = WALinear.from_float(
                m,
                weight_quant="per_channel",
                act_quant="per_token",
                w_bit=w_bit,
                a_bit=a_bit
            )
            setattr(parent, name.split('.')[-1], new_m)


# ============================================================
# Parent finder
# ============================================================

def _get_parent_module(root, full_name: str):
    """
    root.submodule1.submodule2... 형태에서 parent 찾기
    """
    parts = full_name.split(".")[:-1]
    parent = root
    for p in parts:
        if not hasattr(parent, p):
            return None
        parent = getattr(parent, p)
    return parent
