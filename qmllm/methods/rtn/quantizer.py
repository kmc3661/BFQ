import torch
import torch.nn as nn
from tqdm import tqdm
from qmllm.quantization.quant_funcs import pseudo_quantize_tensor
from qmllm.quantization.qlinear import WALinear
from transformers.models.bloom.modeling_bloom import BloomForCausalLM
from transformers.models.opt.modeling_opt import OPTForCausalLM

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

    if layers is None:
        raise NotImplementedError(f"Cannot find decoder layers for {type(model)}")
    return layers

@torch.no_grad()
def pseudo_quantize_model_weight(
    model,
    w_bit,
    q_config,
):

    layers = get_blocks(model)
    for i in tqdm(range(len(layers)), desc="pseudo weight quantization..."):
        named_linears = get_named_linears(layers[i])
        for n, m in named_linears.items():
            # m.cuda()
            m.weight.data = pseudo_quantize_tensor(
                m.weight.data, n_bits=w_bit, **q_config
            )
            # m.cpu()


def get_module_by_name_suffix(model, module_name: str):
    for name, module in model.named_modules():
        if name.endswith(module_name):
            return module


@torch.no_grad()
def pseudo_quantize_model_weight_act(
    model,
    w_bit,
    a_bit,
):
    
    layers = get_blocks(model)
    for i in tqdm(range(len(layers)), desc="pseudo weight activation quantization..."):
        named_linears = get_named_linears(layers[i])
        for n, m in named_linears.items():
            new_linear = WALinear.from_float(m, weight_quant="per_channel", act_quant="per_token", w_bit=w_bit, a_bit=a_bit)
            father_module = get_module_by_name_suffix(layers[i], '.'.join(n.split(".")[:-1]))
            setattr(father_module, n.split('.')[-1], new_linear)
            del new_linear, m
            torch.cuda.empty_cache()