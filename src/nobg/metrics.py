"""Evaluation metrics for binary dichotomous image segmentation.

These mirror the metric suite used to benchmark BiRefNet (MAE, S-measure,
max/mean/weighted F-measure, max/mean E-measure) plus common overlap metrics
(IoU, Dice, accuracy, BER) and the alpha-matting metrics used by SAMA / ZIM
(SAD, MSE, gradient error, connectivity error).

Convention (matching ``loss.py``): every function takes a **probability** map
``pred`` in ``[0, 1]`` and a ground truth ``gt`` in ``[0, 1]``, both shaped
``(B, 1, H, W)``. The caller is responsible for applying ``sigmoid`` to model
logits before calling these. Each function returns a scalar tensor averaged
over the batch.

The matting metrics (``sad``, ``mse_metric``, ``gradient_error``,
``connectivity_error``) treat both maps as **continuous alpha mattes** in
``[0, 1]`` and do NOT binarize — the segmentation metrics above binarize
``pred`` at a threshold, the matting metrics use the soft values directly.

Cost tiers:
    cheap  O(N)   : mae, iou_metric, dice, accuracy, ber, sad, mse_metric,
                    boundary_iou (a few pooling ops)
    medium O(N*T) : f_measure_max/mean, e_measure_max/mean (histogram based),
                    gradient_error (two separable convolutions)
    expensive     : s_measure, weighted_f_measure, connectivity_error
                    (per-image python loops; ``weighted_f_measure``
                    approximates the official Euclidean distance transform with
                    a fixed Gaussian kernel, and ``connectivity_error`` derives
                    its connected components via pure-torch label propagation
                    rather than the reference scipy labelling, so absolute
                    values differ slightly from the official numpy tools).

Boundary quality:
    ``boundary_iou`` (Cheng et al., "Boundary IoU", CVPR 2021) is the standard
    boundary-quality metric (HQSeg-44K reports it as mBIoU). It is region-metric
    blind to interior errors, so it separates near-saturated models (HRSOD /
    UHRSD / DAVIS-S) that plain IoU / S-measure cannot. Pure-torch: the boundary
    band is ``mask XOR erode(mask)``, with erosion implemented as min-pooling.
"""

import torch
import torch.nn.functional as F

EPS = 1e-8


# --------------------------------------------------------------------------- #
# Cheap metrics - O(N)
# --------------------------------------------------------------------------- #
def mae(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Mean absolute error between probability map and ground truth."""
    return (pred - gt).abs().mean(dim=(1, 2, 3)).mean()


def iou_metric(
    pred: torch.Tensor, gt: torch.Tensor, threshold: float = 0.5
) -> torch.Tensor:
    """Hard IoU after binarizing ``pred`` at ``threshold``.

    Distinct from ``loss.iou_loss`` which is a soft, differentiable ``1 - IoU``.
    """
    p = (pred >= threshold).float()
    g = (gt >= 0.5).float()
    inter = (p * g).sum(dim=(1, 2, 3))
    union = p.sum(dim=(1, 2, 3)) + g.sum(dim=(1, 2, 3)) - inter
    return (inter / (union + EPS)).mean()


def dice(pred: torch.Tensor, gt: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Dice coefficient after binarizing ``pred`` at ``threshold``."""
    p = (pred >= threshold).float()
    g = (gt >= 0.5).float()
    inter = (p * g).sum(dim=(1, 2, 3))
    denom = p.sum(dim=(1, 2, 3)) + g.sum(dim=(1, 2, 3))
    return (2 * inter / (denom + EPS)).mean()


def accuracy(
    pred: torch.Tensor, gt: torch.Tensor, threshold: float = 0.5
) -> torch.Tensor:
    """Proportion of correctly classified pixels."""
    p = (pred >= threshold).float()
    g = (gt >= 0.5).float()
    return (p == g).float().mean(dim=(1, 2, 3)).mean()


def ber(pred: torch.Tensor, gt: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Balanced error rate: mean of false-positive and false-negative rates."""
    p = (pred >= threshold).float()
    g = (gt >= 0.5).float()
    tp = (p * g).sum(dim=(1, 2, 3))
    tn = ((1 - p) * (1 - g)).sum(dim=(1, 2, 3))
    fp = (p * (1 - g)).sum(dim=(1, 2, 3))
    fn = ((1 - p) * g).sum(dim=(1, 2, 3))
    fpr = fp / (fp + tn + EPS)
    fnr = fn / (fn + tp + EPS)
    return (0.5 * (fpr + fnr)).mean()


def _erode(mask: torch.Tensor, iterations: int) -> torch.Tensor:
    """Binary erosion of a ``(B, 1, H, W)`` {0,1} mask via 3x3 min-pooling.

    ``iterations`` steps of a 3x3 structuring element = erosion by a
    Chebyshev-radius-``iterations`` square (min-pool is erosion for binary maps).
    """
    if iterations <= 0:
        return mask
    x = mask
    for _ in range(iterations):
        x = -F.max_pool2d(-x, kernel_size=3, stride=1, padding=1)
    return x


def boundary_iou(
    pred: torch.Tensor,
    gt: torch.Tensor,
    threshold: float = 0.5,
    dilation_ratio: float = 0.02,
) -> torch.Tensor:
    """Boundary IoU (Cheng et al., CVPR 2021); reported as mBIoU on HQSeg-44K.

    IoU computed only over a thin band around each mask's contour, so interior
    agreement does not mask boundary errors. The band width per image is
    ``round(dilation_ratio * image_diagonal)`` (min 1), matching the reference
    implementation's ``dilation_ratio`` (default 0.02). The boundary region is
    ``mask XOR erode(mask, width)``. Binarizes ``pred`` at ``threshold``.
    Averaged over the batch.
    """
    h, w = pred.shape[-2:]
    d = int(round(dilation_ratio * ((h**2 + w**2) ** 0.5)))
    d = max(d, 1)
    p = (pred >= threshold).float()
    g = (gt >= 0.5).float()
    p_bnd = (p - _erode(p, d)).clamp(0, 1)
    g_bnd = (g - _erode(g, d)).clamp(0, 1)
    inter = (p_bnd * g_bnd).sum(dim=(1, 2, 3))
    union = p_bnd.sum(dim=(1, 2, 3)) + g_bnd.sum(dim=(1, 2, 3)) - inter
    return (inter / (union + EPS)).mean()


# --------------------------------------------------------------------------- #
# Medium metrics - O(N * num_thresholds), histogram based
# --------------------------------------------------------------------------- #
def _threshold_counts(
    pred: torch.Tensor, gt: torch.Tensor, num_thresholds: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-image, per-threshold ``(tp, predicted_positive, n_fg)`` via histograms.

    Thresholds are the ``num_thresholds`` right bin edges of ``[0, 1]``. A pixel
    with value ``v`` is predicted positive at threshold ``t`` when ``v >= t``,
    which is the reverse-cumulative histogram count.

    Returns tensors shaped ``(B, num_thresholds)`` for ``tp`` and
    ``predicted_positive`` and ``(B,)`` for ``n_fg``.
    """
    b = pred.shape[0]
    g = (gt >= 0.5).float().view(b, -1)
    p = pred.view(b, -1).clamp(0, 1)

    # Bin index in [0, num_thresholds - 1] for each pixel.
    idx = (p * num_thresholds).long().clamp(max=num_thresholds - 1)
    device = pred.device
    tp = torch.zeros(b, num_thresholds, device=device)
    pp = torch.zeros(b, num_thresholds, device=device)
    for i in range(b):
        pp[i] = torch.bincount(idx[i], minlength=num_thresholds).float()
        tp[i] = torch.bincount(idx[i], weights=g[i], minlength=num_thresholds).float()

    # Reverse cumulative sum: count of pixels with value >= threshold bin.
    pp = pp.flip(-1).cumsum(-1).flip(-1)
    tp = tp.flip(-1).cumsum(-1).flip(-1)
    n_fg = g.sum(-1)
    return tp, pp, n_fg


def _f_measure_curve(
    pred: torch.Tensor, gt: torch.Tensor, num_thresholds: int = 255
) -> torch.Tensor:
    """Per-threshold F-measure (``beta^2 = 0.3``), averaged over the batch.

    Returns a ``(num_thresholds,)`` tensor.
    """
    beta2 = 0.3
    tp, pp, n_fg = _threshold_counts(pred, gt, num_thresholds)
    precision = tp / (pp + EPS)
    recall = tp / (n_fg.unsqueeze(1) + EPS)
    f = (1 + beta2) * precision * recall / (beta2 * precision + recall + EPS)
    return f.mean(dim=0)


def f_measure_max(
    pred: torch.Tensor, gt: torch.Tensor, num_thresholds: int = 255
) -> torch.Tensor:
    """Maximum F-measure across thresholds."""
    return _f_measure_curve(pred, gt, num_thresholds).max()


def f_measure_mean(
    pred: torch.Tensor, gt: torch.Tensor, num_thresholds: int = 255
) -> torch.Tensor:
    """Mean F-measure across thresholds."""
    return _f_measure_curve(pred, gt, num_thresholds).mean()


def _e_measure_at(pred_bin: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Enhanced-alignment measure for a batch of binarized predictions.

    ``pred_bin`` and ``gt`` are ``(B, 1, H, W)`` in ``{0, 1}``. Returns ``(B,)``.
    """
    b = pred_bin.shape[0]
    n = pred_bin[0].numel()
    out = torch.zeros(b, device=pred_bin.device)
    for i in range(b):
        fm = pred_bin[i]
        gm = gt[i]
        gt_mean = gm.mean()
        # Degenerate GT: score by pixel agreement directly.
        if gt_mean <= EPS:
            out[i] = 1.0 - fm.mean()
            continue
        if gt_mean >= 1 - EPS:
            out[i] = fm.mean()
            continue
        align_fm = fm - fm.mean()
        align_gt = gm - gt_mean
        align = 2 * align_gt * align_fm / (align_gt**2 + align_fm**2 + EPS)
        enhanced = (align + 1) ** 2 / 4
        out[i] = enhanced.sum() / (n - 1 + EPS)
    return out


def _e_measure_curve(
    pred: torch.Tensor, gt: torch.Tensor, num_thresholds: int = 255
) -> torch.Tensor:
    """Per-threshold E-measure averaged over the batch. Returns ``(num_thresholds,)``."""
    g = (gt >= 0.5).float()
    scores = torch.zeros(num_thresholds, device=pred.device)
    for t in range(num_thresholds):
        thr = (t + 1) / num_thresholds
        pred_bin = (pred >= thr).float()
        scores[t] = _e_measure_at(pred_bin, g).mean()
    return scores


def e_measure_mean(
    pred: torch.Tensor, gt: torch.Tensor, num_thresholds: int = 255
) -> torch.Tensor:
    """Mean E-measure across thresholds."""
    return _e_measure_curve(pred, gt, num_thresholds).mean()


def e_measure_max(
    pred: torch.Tensor, gt: torch.Tensor, num_thresholds: int = 255
) -> torch.Tensor:
    """Maximum E-measure across thresholds."""
    return _e_measure_curve(pred, gt, num_thresholds).max()


# --------------------------------------------------------------------------- #
# Expensive metrics
# --------------------------------------------------------------------------- #
def _ssim_single(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Global single-window SSIM between two 2-D maps (scalar tensor)."""
    mu_x = x.mean()
    mu_y = y.mean()
    n = x.numel()
    sigma_x2 = ((x - mu_x) ** 2).sum() / (n - 1 + EPS)
    sigma_y2 = ((y - mu_y) ** 2).sum() / (n - 1 + EPS)
    sigma_xy = ((x - mu_x) * (y - mu_y)).sum() / (n - 1 + EPS)
    c1 = 0.01**2
    c2 = 0.03**2
    num = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    den = (mu_x**2 + mu_y**2 + c1) * (sigma_x2 + sigma_y2 + c2)
    return num / (den + EPS)


def _s_object(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Object-aware structural similarity term for a single ``(H, W)`` map."""

    def _score(p: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        area = g.sum()
        x = (p * g).sum() / (area + EPS)
        sigma_x = torch.sqrt(((p - x) ** 2 * g).sum() / (area + EPS) + EPS)
        return 2 * x / (x**2 + 1 + sigma_x + EPS)

    fg = _score(pred, gt)
    bg = _score(1 - pred, 1 - gt)
    u = gt.mean()
    return u * fg + (1 - u) * bg


def _s_region(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Region-aware structural similarity term for a single ``(H, W)`` map.

    Splits at the GT weighted centroid into four quadrants, computes single
    window SSIM per quadrant, and area-weights them.
    """
    h, w = gt.shape
    total = gt.sum()
    if total <= EPS:
        return torch.tensor(0.0, device=gt.device)

    rows = torch.arange(h, device=gt.device).float()
    cols = torch.arange(w, device=gt.device).float()
    cy = int(torch.round((gt.sum(1) * rows).sum() / (total + EPS)).item())
    cx = int(torch.round((gt.sum(0) * cols).sum() / (total + EPS)).item())
    cy = max(1, min(h - 1, cy))
    cx = max(1, min(w - 1, cx))

    quadrants = [
        (slice(0, cy), slice(0, cx)),
        (slice(0, cy), slice(cx, w)),
        (slice(cy, h), slice(0, cx)),
        (slice(cy, h), slice(cx, w)),
    ]
    score = torch.tensor(0.0, device=gt.device)
    for ys, xs in quadrants:
        g_q = gt[ys, xs]
        p_q = pred[ys, xs]
        weight = g_q.numel() / (h * w)
        score = score + weight * _ssim_single(p_q, g_q)
    return score


def s_measure(pred: torch.Tensor, gt: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
    """Structure measure ``S_alpha = alpha * S_object + (1 - alpha) * S_region``.

    Per-image python loop (eval batches are small). Handles degenerate GT.
    """
    b = pred.shape[0]
    out = torch.zeros(b, device=pred.device)
    for i in range(b):
        p = pred[i, 0].clamp(0, 1)
        g = (gt[i, 0] >= 0.5).float()
        y = g.mean()
        if y <= EPS:
            out[i] = 1.0 - p.mean()
        elif y >= 1 - EPS:
            out[i] = p.mean()
        else:
            out[i] = alpha * _s_object(p, g) + (1 - alpha) * _s_region(p, g)
    return out.mean()


def weighted_f_measure(
    pred: torch.Tensor, gt: torch.Tensor, beta2: float = 1.0
) -> torch.Tensor:
    """Weighted F-measure (Margolin et al.), approximate.

    The official metric weights errors by a Euclidean distance transform on the
    background. Pure-torch has no EDT, so the spatial weighting is approximated
    with a fixed Gaussian blur (7x7, ``sigma = 5``) of the error map. Values
    therefore diverge slightly from the reference numpy implementation; adequate
    for training-time monitoring.
    """
    b = pred.shape[0]
    # Gaussian kernel.
    ksize, sigma = 7, 5.0
    ax = torch.arange(ksize, device=pred.device).float() - (ksize - 1) / 2
    g1 = torch.exp(-(ax**2) / (2 * sigma**2))
    kernel = g1[:, None] * g1[None, :]
    kernel = (kernel / kernel.sum()).view(1, 1, ksize, ksize)

    out = torch.zeros(b, device=pred.device)
    for i in range(b):
        p = pred[i : i + 1].clamp(0, 1)
        g = (gt[i : i + 1] >= 0.5).float()
        e = (p - g).abs()
        ew = F.conv2d(e, kernel, padding=ksize // 2)
        # Weighted true/false counts.
        tp = ((1 - ew) * g).sum()
        fp = (ew * (1 - g)).sum()
        fn = (ew * g).sum()
        precision = tp / (tp + fp + EPS)
        recall = tp / (tp + fn + EPS)
        out[i] = (1 + beta2) * precision * recall / (beta2 * precision + recall + EPS)
    return out.mean()


# --------------------------------------------------------------------------- #
# Matting metrics - continuous alpha, no binarization (SAMA / ZIM style)
# --------------------------------------------------------------------------- #
def sad(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Sum of absolute differences over the alpha matte.

    Summed over all pixels of each image, then averaged over the batch. The
    matting literature usually reports this in units of 1e3 (i.e. the value
    below divided by 1000); this returns the raw pixel sum so the caller can
    scale as needed.
    """
    return (pred - gt).abs().sum(dim=(1, 2, 3)).mean()


def mse_metric(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Mean squared error over the alpha matte.

    Distinct from ``mae``: penalizes larger errors more heavily. Averaged over
    pixels and the batch.
    """
    return ((pred - gt) ** 2).mean(dim=(1, 2, 3)).mean()


def gradient_error(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Gradient error: L1 difference of spatial gradients (Sobel).

    Captures edge / boundary quality of the alpha matte. Summed over pixels of
    the gradient-magnitude difference per image, then averaged over the batch.
    """
    kx = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=pred.device,
    ).view(1, 1, 3, 3)
    ky = kx.transpose(-1, -2)

    def _grad_mag(x: torch.Tensor) -> torch.Tensor:
        gx = F.conv2d(x, kx, padding=1)
        gy = F.conv2d(x, ky, padding=1)
        return torch.sqrt(gx**2 + gy**2 + EPS)

    diff = (_grad_mag(pred) - _grad_mag(gt)).abs()
    return diff.sum(dim=(1, 2, 3)).mean()


def _connected_components(mask: torch.Tensor, n_iter: int = 256) -> torch.Tensor:
    """Label 4-connected components of a 2-D ``{0, 1}`` mask via propagation.

    Pure-torch (no scipy): each foreground pixel starts with a unique id equal
    to its flat index, then repeatedly takes the max id over its 4-neighbourhood
    until labels stop changing. Returns an ``(H, W)`` integer label map (0 for
    background). ``n_iter`` bounds the propagation; the eval maps are small.
    """
    h, w = mask.shape
    ids = torch.arange(1, h * w + 1, device=mask.device).view(h, w).float()
    labels = ids * mask
    for _ in range(n_iter):
        padded = F.pad(labels.view(1, 1, h, w), (1, 1, 1, 1), value=0)
        up = padded[:, :, :-2, 1:-1]
        down = padded[:, :, 2:, 1:-1]
        left = padded[:, :, 1:-1, :-2]
        right = padded[:, :, 1:-1, 2:]
        neigh = torch.maximum(torch.maximum(up, down), torch.maximum(left, right))
        new = torch.maximum(labels.view(1, 1, h, w), neigh).view(h, w) * mask
        if torch.equal(new, labels):
            break
        labels = new
    return labels.long()


def connectivity_error(
    pred: torch.Tensor, gt: torch.Tensor, threshold: float = 0.5
) -> torch.Tensor:
    """Connectivity error: penalizes fragmented / disconnected alpha regions.

    Approximates the reference metric (Rhemann et al.): both maps are binarized
    at ``threshold``, the largest connected component of each is kept, and the
    error is the summed absolute difference between the alpha values inside the
    original masks and their largest-connected-component versions. Uses a
    pure-torch label propagation for connected components (see
    ``_connected_components``), so absolute values differ slightly from the
    official scipy-based tool. Per-image python loop; averaged over the batch.
    """
    b = pred.shape[0]
    out = torch.zeros(b, device=pred.device)
    for i in range(b):
        p = pred[i, 0].clamp(0, 1)
        g = gt[i, 0].clamp(0, 1)

        def _largest_cc_alpha(alpha: torch.Tensor) -> torch.Tensor:
            binm = (alpha >= threshold).float()
            labels = _connected_components(binm)
            if labels.max() == 0:
                return torch.zeros_like(alpha)
            counts = torch.bincount(labels.view(-1))
            counts[0] = 0  # ignore background label
            keep = counts.argmax()
            return alpha * (labels == keep).float()

        p_cc = _largest_cc_alpha(p)
        g_cc = _largest_cc_alpha(g)
        # Connectivity term: alpha lost by dropping non-largest components.
        conn = ((p - p_cc).abs() - (g - g_cc).abs()).abs()
        out[i] = conn.sum()
    return out.mean()
