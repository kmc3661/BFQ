import torch
import numpy as np

from qmllm.methods.mbq_fg.quantize.pre_quant import get_blocks, process_input


def _tensor_entropy(x: torch.Tensor, bins: int = 256, max_samples: int = 100000) -> float:
    """Compute entropy of a tensor with histogram approximation."""
    if x.numel() == 0:
        return 0.0
    with torch.no_grad():
        flat = x.detach().float().view(-1)
        if flat.numel() > max_samples:
            # uniform random subsample to limit cost
            idx = torch.randperm(flat.numel(), device=flat.device)[:max_samples]
            flat = flat[idx]
        flat = flat.cpu()
        min_v = flat.min().item()
        max_v = flat.max().item()
        # avoid zero width
        if min_v == max_v:
            return 0.0
        hist = torch.histc(flat, bins=bins, min=min_v, max=max_v)
        prob = hist / hist.sum()
        prob = prob[prob > 0]
        ent = -(prob * torch.log(prob)).sum().item()
    return ent


def measure_layer_entropies(model, prompt_inputs, prompt_kwargs, max_samples: int = 16, bins: int = 256) -> list:
    """
    Run one forward pass and record entropy of each decoder layer input.
    Returns a list of entropy values (len = num_layers).
    """
    backbone = getattr(model, "model", model)
    layers = get_blocks(backbone)
    entropies = [None] * len(layers)

    def make_hook(idx):
        def hook(module, inputs):
            x = inputs[0]
            entropies[idx] = _tensor_entropy(x, bins=bins, max_samples=max_samples * 1000)
        return hook

    handles = [layer.register_forward_pre_hook(make_hook(i)) for i, layer in enumerate(layers)]

    try:
        inputs, _, _ = process_input(prompt_inputs, prompt_kwargs)
        # move to device of model
        model.to_cuda()
        device = next(backbone.parameters()).device
        for k, v in list(inputs.items()):
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(device)
        with torch.no_grad():
            model(**inputs)
    finally:
        for h in handles:
            h.remove()

    # fill missing entries with median
    valid = [e for e in entropies if e is not None]
    if len(valid) == 0:
        return [0.0] * len(entropies)
    med = float(np.median(valid))
    entropies = [med if e is None else e for e in entropies]
    return entropies


def partition_blocks(entropies, max_block_len: int = 6, percentile: float = 0.6):
    """
    Simple entropy-based block partition.
    - Start a new block when entropy exceeds threshold or block length hits max_block_len.
    """
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
        # new block if entropy high or max length reached
        if ent > thresh or cur_len >= max_block_len:
            blocks.append((start, idx - 1))
            start = idx
            cur_len = 1
        else:
            cur_len += 1
    blocks.append((start, len(entropies) - 1))
    return blocks


def format_blocks(blocks):
    return ", ".join([f"[{s}-{e}]" for s, e in blocks])
