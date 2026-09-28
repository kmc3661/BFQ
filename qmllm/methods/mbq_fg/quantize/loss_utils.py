import torch


def _align_mask(mask: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    if mask is None:
        return None
    if not isinstance(mask, torch.Tensor):
        mask = torch.as_tensor(mask)
    mask = mask.to(device=out.device, dtype=torch.bool)
    if mask.dim() == 1:
        mask = mask.unsqueeze(0)
    tgt_b, tgt_s = out.shape[0], out.shape[1]
    if mask.shape[0] > tgt_b:
        mask = mask[:tgt_b]
    elif mask.shape[0] < tgt_b:
        pad_b = torch.zeros((tgt_b - mask.shape[0], mask.shape[1]), dtype=torch.bool, device=mask.device)
        mask = torch.cat([mask, pad_b], dim=0)
    if mask.shape[1] > tgt_s:
        mask = mask[:, :tgt_s]
    elif mask.shape[1] < tgt_s:
        pad_s = torch.zeros((mask.shape[0], tgt_s - mask.shape[1]), dtype=torch.bool, device=mask.device)
        mask = torch.cat([mask, pad_s], dim=1)
    return mask


def _align_token_weight(token_weight: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    if token_weight is None:
        return None
    if not isinstance(token_weight, torch.Tensor):
        token_weight = torch.as_tensor(token_weight)
    tw = token_weight.to(device=out.device, dtype=torch.float32)
    if tw.dim() == 1:
        tw = tw.unsqueeze(0)
    tgt_b, tgt_s = out.shape[0], out.shape[1]
    if tw.shape[0] > tgt_b:
        tw = tw[:tgt_b]
    elif tw.shape[0] < tgt_b:
        pad_b = torch.zeros((tgt_b - tw.shape[0], tw.shape[1]), dtype=tw.dtype, device=tw.device)
        tw = torch.cat([tw, pad_b], dim=0)
    if tw.shape[1] > tgt_s:
        tw = tw[:, :tgt_s]
    elif tw.shape[1] < tgt_s:
        pad_s = torch.zeros((tw.shape[0], tgt_s - tw.shape[1]), dtype=tw.dtype, device=tw.device)
        tw = torch.cat([tw, pad_s], dim=1)
    tw = torch.clamp(tw, min=0.0)
    denom = tw.sum(dim=1, keepdim=True)
    zero_row = denom <= 1e-12
    if zero_row.any():
        tw = tw.clone()
        tw[zero_row.expand_as(tw)] = 1.0 / max(1, tw.shape[1])
        denom = tw.sum(dim=1, keepdim=True)
    return tw / (denom + 1e-12)


def compute_recon_loss(
    org_out: torch.Tensor,
    out: torch.Tensor,
    loss_mode: str = "mae",
    ans_mask: torch.Tensor = None,
    vis_mask: torch.Tensor = None,
    reweight_ratio: float = None,
    token_weight: torch.Tensor = None,
):
    if loss_mode == "mse":
        diff = (org_out - out).float().pow(2)
    else:
        diff = (org_out - out).float().abs()

    # Xiang2026-style token-wise weighting has top priority.
    tw = _align_token_weight(token_weight, out)
    if tw is not None:
        token_err = diff.mean(dim=-1)
        return (token_err * tw).sum() / (tw.sum() + 1e-12)

    ans_mask = _align_mask(ans_mask, out)
    vis_mask = _align_mask(vis_mask, out)
    if ans_mask is not None and vis_mask is not None:
        ans_expand = ans_mask.unsqueeze(-1).expand_as(out)
        vis_expand = vis_mask.unsqueeze(-1).expand_as(out)
        diff_ans = diff * ans_expand
        diff_vis = diff * vis_expand
        if reweight_ratio is not None:
            if loss_mode == "mse":
                return diff_ans.sum() / (ans_expand.sum() + 1e-12) + reweight_ratio * (
                    diff_vis.sum() / (vis_expand.sum() + 1e-12)
                )
            return (diff_ans.sum() + reweight_ratio * diff_vis.sum()) / (
                ans_expand.sum() + vis_expand.sum() + 1e-12
            )
        return diff.mean()
    if ans_mask is not None:
        ans_expand = ans_mask.unsqueeze(-1).expand_as(out)
        diff_ans = diff * ans_expand
        return diff_ans.sum() / (ans_expand.sum() + 1e-12)
    return diff.mean()
