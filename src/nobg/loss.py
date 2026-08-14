import torch
import torch.nn.functional as F


def iou_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    inter = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - inter
    return (1 - inter / (union + 1e-8)).mean()


def ssim_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    C1 = 0.01**2
    C2 = 0.03**2
    mu_x = F.avg_pool2d(pred, 3, 1, 1)
    mu_y = F.avg_pool2d(target, 3, 1, 1)
    sigma_x = F.avg_pool2d(pred * pred, 3, 1, 1) - mu_x * mu_x
    sigma_y = F.avg_pool2d(target * target, 3, 1, 1) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(pred * target, 3, 1, 1) - mu_x * mu_y
    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / (
        (mu_x * mu_x + mu_y * mu_y + C1) * (sigma_x + sigma_y + C2)
    )
    return torch.clamp((1 - ssim_map) / 2, 0, 1).mean()


def birefnet_loss(scaled_preds: list[torch.Tensor], gt: torch.Tensor) -> torch.Tensor:
    """Multi-scale pixel loss matching BiRefNet training: weighted BCE + IoU + SSIM."""
    loss = torch.tensor(0.0, device=gt.device)
    for pred in scaled_preds:
        if pred.shape[2:] != gt.shape[2:]:
            pred = F.interpolate(
                pred, size=gt.shape[2:], mode="bilinear", align_corners=True
            )
        pred_sig = pred.sigmoid()
        loss = loss + 30 * F.binary_cross_entropy_with_logits(pred, gt)
        loss = loss + 0.5 * iou_loss(pred_sig, gt)
        loss = loss + 10 * ssim_loss(pred_sig, gt)
    return loss


def sigmoid_focal_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.6,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Focal loss (`arXiv:1708.02002`) on raw logits, mean-reduced.

    ``alpha`` weights the positive class; note the default is **0.6**, not
    RetinaNet's 0.25 — SAM 3's semantic-segmentation head uses a
    foreground-favouring alpha because a matte's positive class covers a large
    fraction of the image rather than a handful of anchors.
    """
    ce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
    prob = pred.sigmoid()
    p_t = prob * target + (1 - prob) * (1 - target)
    loss = ce * (1 - p_t).pow(gamma)
    if alpha >= 0:
        loss = loss * (alpha * target + (1 - alpha) * (1 - target))
    return loss.mean()


def dice_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Soft dice loss on raw logits: ``1 - 2|X∩Y| / (|X|+|Y|)``, smoothed by 1.

    Per-sample over flattened spatial dims, then averaged over the batch. This is
    the mask-overlap term of DETR-family set losses, which SAM 3 inherits.
    """
    prob = pred.sigmoid().flatten(1)
    tgt = target.flatten(1)
    numerator = 2 * (prob * tgt).sum(-1)
    denominator = prob.sum(-1) + tgt.sum(-1)
    return (1 - (numerator + 1) / (denominator + 1)).mean()


def sam3_loss(
    scaled_preds: list[torch.Tensor],
    gt: torch.Tensor,
    *,
    focal_alpha: float = 0.6,
    focal_gamma: float = 2.0,
    focal_weight: float = 20.0,
    dice_weight: float = 30.0,
) -> torch.Tensor:
    """SAM 3's semantic-segmentation objective: weighted focal + dice.

    Mirrors the ``SemanticSegCriterion`` of Meta's SAM 3 training code — focal
    loss on the mask logits plus a dice term, at their published relative
    weights (20 : 30, with ``focal_alpha=0.6``). It is written here from those
    formulations, not copied: Meta's implementation is under the SAM License
    while nobg is Apache-2.0.

    Deliberately **not** part of the original criterion, because nobg's wrapper
    has no matching output:

    - The Hungarian-matched set losses (box L1/GIoU, ``IABCEMdetr``
      classification, per-instance mask/dice) need instance-level targets;
      nobg trains on a single merged matte, so there is nothing to match.
    - The presence-head BCE needs a per-image "is the concept present" label.
      Every training pair here has a foreground, so that target is constantly
      1 and the term carries no gradient signal worth having.

    Signature matches ``birefnet_loss`` — a list of predictions and one ground
    truth — so it drops into ``Sam3.criterion`` unchanged. ``Sam3`` always
    passes a single-element list.

    Args:
        scaled_preds: Raw mask logits, each ``(B, 1, H, W)``. Any that do not
            match ``gt`` spatially are bilinearly resized to it.
        gt: ``(B, 1, H, W)`` target in ``[0, 1]``.
    """
    loss = torch.tensor(0.0, device=gt.device)
    for pred in scaled_preds:
        if pred.shape[2:] != gt.shape[2:]:
            pred = F.interpolate(
                pred, size=gt.shape[2:], mode="bilinear", align_corners=False
            )
        loss = loss + focal_weight * sigmoid_focal_loss(
            pred, gt, alpha=focal_alpha, gamma=focal_gamma
        )
        loss = loss + dice_weight * dice_loss(pred, gt)
    return loss
