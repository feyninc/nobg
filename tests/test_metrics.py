import torch

from nobg.metrics import (
    accuracy,
    ber,
    boundary_iou,
    connectivity_error,
    dice,
    e_measure_max,
    e_measure_mean,
    f_measure_max,
    f_measure_mean,
    gradient_error,
    iou_metric,
    mae,
    mse_metric,
    s_measure,
    sad,
    weighted_f_measure,
)


def _mask(pattern: str, size: int = 16) -> torch.Tensor:
    """Build a (1, 1, size, size) mask in {0, 1}."""
    m = torch.zeros(1, 1, size, size)
    if pattern == "left":
        m[..., : size // 2] = 1.0
    elif pattern == "right":
        m[..., size // 2 :] = 1.0
    elif pattern == "full":
        m[...] = 1.0
    return m


class TestMetrics:
    def test_perfect_match(self):
        g = _mask("left")
        p = g.clone()
        assert mae(p, g).item() < 1e-6
        assert iou_metric(p, g).item() > 0.999
        assert boundary_iou(p, g).item() > 0.999
        assert dice(p, g).item() > 0.999
        assert accuracy(p, g).item() > 0.999
        assert ber(p, g).item() < 1e-6
        assert f_measure_max(p, g).item() > 0.99
        assert e_measure_mean(p, g).item() > 0.99
        assert s_measure(p, g).item() > 0.99
        assert weighted_f_measure(p, g).item() > 0.9
        assert sad(p, g).item() < 1e-4
        assert mse_metric(p, g).item() < 1e-8
        assert gradient_error(p, g).item() < 1e-3
        assert connectivity_error(p, g).item() < 1e-4

    def test_disjoint(self):
        g = _mask("left")
        p = _mask("right")
        assert mae(p, g).item() > 0.9
        assert iou_metric(p, g).item() < 1e-6
        assert boundary_iou(p, g).item() < 1e-6
        assert dice(p, g).item() < 1e-6
        assert s_measure(p, g).item() < 0.5
        assert weighted_f_measure(p, g).item() < 0.2
        # Every foreground pixel is wrong: SAD == number of GT-foreground pixels.
        assert sad(p, g).item() > 0.9 * g.sum().item()
        assert mse_metric(p, g).item() > 0.4

    def test_matting_soft_alpha(self):
        # SAD/MSE operate on soft alpha without binarization.
        g = _mask("left") * 0.8
        p = _mask("left") * 0.5
        # Per-pixel abs diff 0.3 over the 128 foreground pixels of a 16x16 half.
        assert abs(sad(p, g).item() - 0.3 * 128) < 1e-3
        assert abs(mse_metric(p, g).item() - (0.3**2) * 0.5) < 1e-4

    def test_connectivity_penalizes_fragmentation(self):
        # A prediction with a spurious disconnected blob should incur more
        # connectivity error than a clean single-component prediction.
        g = _mask("left")
        p_clean = g.clone()
        p_frag = g.clone()
        p_frag[..., 0:2, -2:] = 1.0  # detached blob in the far corner
        assert (
            connectivity_error(p_frag, g).item() > connectivity_error(p_clean, g).item()
        )

    def test_gt_all_zero(self):
        g = torch.zeros(1, 1, 16, 16)
        p_good = torch.zeros(1, 1, 16, 16)
        p_bad = torch.ones(1, 1, 16, 16)
        assert s_measure(p_good, g).item() > 0.99
        assert s_measure(p_bad, g).item() < 0.01
        assert e_measure_mean(p_good, g).item() > 0.99

    def test_gt_all_one(self):
        g = _mask("full")
        p_good = _mask("full")
        assert s_measure(p_good, g).item() > 0.99
        assert mae(p_good, g).item() < 1e-6

    def test_batched(self):
        g = torch.cat([_mask("left"), _mask("right")], dim=0)
        p = g.clone()
        assert g.shape[0] == 2
        assert mae(p, g).item() < 1e-6
        assert iou_metric(p, g).item() > 0.999
        val = s_measure(p, g)
        assert val.ndim == 0
        assert val.item() > 0.99

    def test_noise_monotonicity(self):
        torch.manual_seed(0)
        g = _mask("left")
        p_clean = g.float()
        p_noisy = (g * 0.6 + 0.2).clamp(0, 1)  # softer, less confident
        assert mae(p_clean, g).item() < mae(p_noisy, g).item()
        assert s_measure(p_clean, g).item() >= s_measure(p_noisy, g).item()

    def test_returns_finite_scalars(self):
        torch.manual_seed(1)
        p = torch.rand(2, 1, 16, 16)
        g = (torch.rand(2, 1, 16, 16) > 0.5).float()
        for fn in (
            mae,
            iou_metric,
            boundary_iou,
            dice,
            accuracy,
            ber,
            f_measure_max,
            f_measure_mean,
            e_measure_mean,
            e_measure_max,
            s_measure,
            weighted_f_measure,
            sad,
            mse_metric,
            gradient_error,
            connectivity_error,
        ):
            v = fn(p, g)
            assert v.ndim == 0
            assert torch.isfinite(v).item()

    def test_boundary_iou_sensitive_to_shift(self):
        # A shifted blob keeps high region IoU but boundary IoU should drop
        # sharply — the reason mBIoU is added (region metrics miss edge errors).
        size = 128
        g = torch.zeros(1, 1, size, size)
        g[..., 32:96, 32:96] = 1.0
        p = torch.zeros(1, 1, size, size)
        p[..., 36:100, 36:100] = 1.0  # shifted by 4 px
        assert iou_metric(p, g).item() > 0.7
        assert boundary_iou(p, g).item() < iou_metric(p, g).item()
