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
