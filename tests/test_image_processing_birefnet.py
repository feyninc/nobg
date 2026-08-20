import json
import os

import numpy as np
import pytest
import torch
from PIL import Image

from nobg import AutoProcessor, BiRefNet, BiRefNetImageProcessor
from nobg.birefnet.modeling_birefnet import BiRefNetConfig
from nobg.utils import _box_blur


def _reference_box_blur(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """Box filter written out offset by offset, with replicate padding.

    Deliberately naive and independent of the shipped separable implementation:
    it clamps indices instead of padding and sums every offset in the window,
    using OpenCV's anchor (``kernel_size // 2`` before the pixel).
    """
    lo = kernel_size // 2
    hi = kernel_size - 1 - lo
    height, width = x.shape[-2:]
    acc = torch.zeros_like(x)
    for dy in range(-lo, hi + 1):
        rows = (torch.arange(height) + dy).clamp(0, height - 1)
        for dx in range(-lo, hi + 1):
            cols = (torch.arange(width) + dx).clamp(0, width - 1)
            acc += x[..., rows, :][..., cols]
    return acc / (kernel_size * kernel_size)


def _fringed_image(size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """A red disc on green, observed through its own soft-edged matte.

    Returns the mixed image and the matte, so refinement has a known right
    answer: pure red in the soft band, with the green fully unmixed.
    """
    rows, cols = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    dist = ((rows - size / 2) ** 2 + (cols - size / 2) ** 2).sqrt()
    alpha = (1 - (dist - size / 4) / (size / 10)).clamp(0, 1)
    foreground = torch.zeros(3, size, size)
    foreground[0] = 1.0
    background = torch.zeros(3, size, size)
    background[1] = 1.0
    return foreground * alpha + background * (1 - alpha), alpha


def _to_pil(image: torch.Tensor) -> Image.Image:
    arr = (image.clamp(0, 1) * 255).to(torch.uint8).permute(1, 2, 0).numpy()
    return Image.fromarray(arr, mode="RGB")


class TestBiRefNetImageProcessor:
    @pytest.fixture
    def processor(self):
        return BiRefNetImageProcessor()

    @pytest.fixture
    def rgb_image(self):
        return Image.new("RGB", (64, 48), (255, 255, 255))

    def test_default_attributes(self, processor):
        assert processor.size["height"] == 1024
        assert processor.size["width"] == 1024
        assert processor.resample == 2  # PILImageResampling.BILINEAR
        assert tuple(processor.image_mean) == (0.485, 0.456, 0.406)
        assert tuple(processor.image_std) == (0.229, 0.224, 0.225)
        assert processor.do_resize
        assert processor.do_rescale
        assert processor.do_normalize
        assert processor.do_convert_rgb
        assert processor.rescale_factor == 1 / 255

    def test_preprocess_pil(self, processor, rgb_image):
        out = processor(
            rgb_image, size={"height": 32, "width": 32}, return_tensors="pt"
        )
        assert out["pixel_values"].shape == (1, 3, 32, 32)
        assert out["pixel_values"].dtype == torch.float32

    def test_preprocess_normalization_values(self, processor):
        white = Image.new("RGB", (40, 40), (255, 255, 255))
        out = processor(white, size={"height": 16, "width": 16}, return_tensors="pt")
        expected = [
            (1.0 - m) / s for m, s in zip((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]
        channel_means = out["pixel_values"][0].mean(dim=(1, 2))
        for got, exp in zip(channel_means.tolist(), expected):
            assert got == pytest.approx(exp, abs=1e-4)

    def test_preprocess_batch_mixed_sizes(self, processor):
        imgs = [Image.new("RGB", (64, 48)), Image.new("RGB", (30, 90))]
        out = processor(imgs, size={"height": 32, "width": 32}, return_tensors="pt")
        assert out["pixel_values"].shape == (2, 3, 32, 32)

    def test_segmentation_maps_binarized(self, processor, rgb_image):
        mask = Image.new("L", (64, 48))
        mask.paste(200, (0, 0, 32, 48))  # left half -> foreground
        mask.paste(60, (32, 0, 64, 48))  # right half -> background
        out = processor(
            rgb_image,
            segmentation_maps=mask,
            size={"height": 32, "width": 32},
            return_tensors="pt",
        )
        labels = out["labels"]
        assert labels.shape == (1, 1, 32, 32)
        assert labels.dtype == torch.float32
        assert set(labels.unique().tolist()) <= {0.0, 1.0}
        # left half (200) -> 1.0, right half (60) -> 0.0
        assert labels[0, 0, :, :16].mean() == pytest.approx(1.0)
        assert labels[0, 0, :, 16:].mean() == pytest.approx(0.0)

    def test_save_pretrained(self, processor, tmp_path):
        processor.save_pretrained(tmp_path)
        assert "preprocessor_config.json" in os.listdir(tmp_path)
        with open(tmp_path / "preprocessor_config.json") as f:
            saved = json.load(f)
        assert saved["image_processor_type"] == "BiRefNetImageProcessor"
        assert saved["size"] == {"height": 1024, "width": 1024}
        assert list(saved["image_mean"]) == [0.485, 0.456, 0.406]

    def test_from_pretrained_roundtrip_custom_size(self, tmp_path):
        proc = BiRefNetImageProcessor(size={"height": 512, "width": 512})
        proc.save_pretrained(tmp_path)
        loaded = BiRefNetImageProcessor.from_pretrained(tmp_path)
        assert loaded.size["height"] == 512
        assert loaded.size["width"] == 512
        assert tuple(loaded.image_mean) == (0.485, 0.456, 0.406)
        assert tuple(loaded.image_std) == (0.229, 0.224, 0.225)
        out = loaded(Image.new("RGB", (100, 80)), return_tensors="pt")
        assert out["pixel_values"].shape == (1, 3, 512, 512)

    def test_auto_processor_local_dir_roundtrip(self, tmp_path):
        # AutoProcessor dispatches on preprocessor_config.json's image_processor_type,
        # so it can reload a saved directory without any Hub access.
        BiRefNetImageProcessor(size={"height": 256, "width": 256}).save_pretrained(
            tmp_path
        )
        loaded = AutoProcessor.from_pretrained(tmp_path)
        assert isinstance(loaded, BiRefNetImageProcessor)
        assert loaded.size["height"] == 256
        assert loaded.size["width"] == 256

    def test_auto_processor_unknown_local_dir_raises(self, tmp_path):
        (tmp_path / "preprocessor_config.json").write_text(
            json.dumps({"image_processor_type": "SomethingElse"})
        )
        with pytest.raises(ValueError):
            AutoProcessor.from_pretrained(tmp_path)

    def test_post_process_alpha_matting(self, processor):
        outputs = {"logits": torch.randn(2, 1, 16, 16)}
        mattes = processor.post_process_alpha_matting(
            outputs, target_sizes=[(37, 53), (20, 20)]
        )
        assert len(mattes) == 2
        assert mattes[0].shape == (37, 53)
        assert mattes[1].shape == (20, 20)
        for m in mattes:
            assert m.min() >= 0.0 and m.max() <= 1.0

    def test_post_process_no_target_sizes(self, processor):
        outputs = {"logits": torch.randn(1, 1, 16, 16)}
        mattes = processor.post_process_alpha_matting(outputs)
        assert mattes[0].shape == (16, 16)

    def test_post_process_target_sizes_length_mismatch_raises(self, processor):
        outputs = {"logits": torch.randn(2, 1, 16, 16)}
        with pytest.raises(ValueError):
            processor.post_process_alpha_matting(outputs, target_sizes=[(10, 10)])

    def test_cutout(self, processor):
        image = Image.new("RGB", (40, 30), (10, 20, 30))
        alpha = torch.rand(30, 40)
        cut = processor.cutout(image, alpha)
        assert cut.mode == "RGBA"
        assert cut.size == (40, 30)
        got = torch.from_numpy(np.array(cut.getchannel("A")))
        expected = (alpha.clamp(0, 1) * 255).to(torch.uint8)
        assert torch.equal(got.to(torch.uint8), expected)

    def test_cutout_refine_keeps_alpha_and_changes_colors(self, processor):
        # A soft edge over a green background: refining must leave the matte
        # untouched while pulling the leaked green out of the RGB channels.
        image, alpha = _fringed_image(48)
        plain = processor.cutout(_to_pil(image), alpha)
        refined = processor.cutout(_to_pil(image), alpha, refine=True)
        assert refined.mode == "RGBA"
        assert refined.size == plain.size
        assert np.array_equal(
            np.array(refined.getchannel("A")), np.array(plain.getchannel("A"))
        )
        band = (alpha > 0.05) & (alpha < 0.95)
        green_before = np.array(plain.getchannel("G"))[band.numpy()].mean()
        green_after = np.array(refined.getchannel("G"))[band.numpy()].mean()
        assert green_after < green_before / 2

    def test_refine_foreground_removes_color_fringe(self, processor):
        image, alpha = _fringed_image(96)
        refined = processor.refine_foreground(image, alpha)
        band = (alpha > 0.05) & (alpha < 0.95)
        # Green is the background that bled in; red is the true subject color.
        assert refined[1][band].mean() < image[1][band].mean() / 4
        assert refined[0][band].mean() > 0.8

    def test_refine_foreground_opaque_matte_is_identity(self, processor):
        # alpha == 1 everywhere leaves nothing to unmix, so the estimator must
        # hand back the original colors.
        image = torch.rand(3, 32, 24)
        refined = processor.refine_foreground(image, torch.ones(32, 24), r=9)
        assert torch.allclose(refined, image, atol=1e-4)

    @pytest.mark.parametrize(
        ("image_shape", "alpha_shape"),
        [
            ((3, 20, 16), (20, 16)),
            ((3, 20, 16), (1, 20, 16)),
            ((2, 3, 20, 16), (2, 1, 20, 16)),
            ((2, 3, 20, 16), (1, 20, 16)),
        ],
    )
    def test_refine_foreground_shapes(self, processor, image_shape, alpha_shape):
        out = processor.refine_foreground(
            torch.rand(image_shape), torch.rand(alpha_shape), r=5
        )
        assert out.shape == image_shape

    def test_refine_foreground_resizes_mismatched_alpha(self, processor):
        out = processor.refine_foreground(
            torch.rand(3, 20, 16), torch.rand(40, 32), r=5
        )
        assert out.shape == (3, 20, 16)

    def test_refine_foreground_pil_roundtrip(self, processor):
        image = Image.new("RGB", (24, 18), (200, 30, 40))
        out = processor.refine_foreground(image, Image.new("L", (24, 18), 128), r=5)
        assert isinstance(out, Image.Image)
        assert out.mode == "RGB"
        assert out.size == (24, 18)

    @pytest.mark.parametrize(
        "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
    )
    def test_refine_foreground_preserves_dtype_without_overflow(self, processor, dtype):
        # Half precision overflows the estimator's divisions unless it computes
        # in float32, which shows up as non-finite (black) patches.
        out = processor.refine_foreground(
            torch.rand(3, 32, 32).to(dtype), torch.rand(32, 32).to(dtype), r=9
        )
        assert out.dtype == dtype
        assert torch.isfinite(out).all()

    def test_refine_foreground_radius_wider_than_image(self, processor):
        out = processor.refine_foreground(
            torch.rand(3, 16, 16), torch.rand(16, 16), r=90
        )
        assert out.shape == (3, 16, 16)
        assert torch.isfinite(out).all()

    def test_refine_foreground_rejects_bad_rank(self, processor):
        with pytest.raises(ValueError):
            processor.refine_foreground(torch.rand(20, 16), torch.rand(20, 16))

    @pytest.mark.parametrize("kernel_size", [2, 3, 5, 6, 7, 8])
    def test_box_blur_matches_reference_box_filter(self, kernel_size):
        # The blur runs as two 1-D pools for speed; this pins it to an explicit
        # replicate-padded box filter using OpenCV's anchor, which is the
        # `cv2.blur` this replaces. Compared away from the border, where the
        # reference's index clamping and the pooling padding must agree exactly.
        torch.manual_seed(0)
        x = torch.rand(1, 2, 24, 24)
        got = _box_blur(x, kernel_size)
        expected = _reference_box_blur(x, kernel_size)
        margin = kernel_size + 1
        assert got.shape == x.shape
        assert torch.allclose(
            got[..., margin:-margin, margin:-margin],
            expected[..., margin:-margin, margin:-margin],
            atol=1e-6,
        )

    def test_box_blur_unit_kernel_is_identity(self):
        x = torch.rand(1, 1, 8, 8)
        assert torch.equal(_box_blur(x, 1), x)

    def test_box_blur_rejects_zero_kernel(self):
        with pytest.raises(ValueError):
            _box_blur(torch.rand(1, 1, 8, 8), 0)

    def test_end_to_end_with_small_model(self):
        config = BiRefNetConfig(
            image_size=128,
            patch_size=4,
            embed_dim=16,
            num_layers=4,
            depths=[1, 1, 1, 1],
            num_heads=[1, 2, 4, 8],
            window_size=2,
            dec_channels_inter=8,
        )
        model = BiRefNet(config=config).eval()
        processor = BiRefNetImageProcessor(size={"height": 128, "width": 128})

        image = Image.new("RGB", (200, 150), (120, 130, 140))
        mask = Image.new("L", (200, 150), 255)
        inputs = processor(image, segmentation_maps=mask, return_tensors="pt")
        out = model(pixel_values=inputs["pixel_values"], labels=inputs["labels"])
        assert out["logits"].shape == (1, 1, 128, 128)
        assert out["loss"].ndim == 0
        assert torch.isfinite(out["loss"])


class TestBiRefNetPredict:
    @pytest.fixture
    def model(self):
        return BiRefNet(
            config=BiRefNetConfig(
                image_size=128,
                patch_size=4,
                embed_dim=16,
                num_layers=4,
                depths=[1, 1, 1, 1],
                num_heads=[1, 2, 4, 8],
                window_size=2,
                dec_channels_inter=8,
            )
        )

    @pytest.fixture
    def processor(self):
        return BiRefNetImageProcessor(size={"height": 128, "width": 128})

    def test_single_image_returns_cutout(self, model, processor):
        image = Image.new("RGB", (60, 40), (10, 20, 30))
        cut = model.predict(processor, image)
        assert isinstance(cut, Image.Image)
        assert cut.mode == "RGBA"
        # Composited back at the input's own resolution, not the model's.
        assert cut.size == (60, 40)

    def test_list_returns_list_at_original_sizes(self, model, processor):
        images = [Image.new("RGB", (60, 40)), Image.new("RGB", (33, 77))]
        cuts = model.predict(processor, images)
        assert isinstance(cuts, list)
        assert [c.size for c in cuts] == [(60, 40), (33, 77)]

    def test_return_type_alpha(self, model, processor):
        image = Image.new("RGB", (60, 40))
        alpha = model.predict(processor, image, return_type="alpha")
        assert isinstance(alpha, torch.Tensor)
        assert alpha.shape == (40, 60)
        assert alpha.min() >= 0 and alpha.max() <= 1

    def test_batch_size_does_not_change_results(self, model, processor):
        images = [Image.new("RGB", (60, 40)), Image.new("RGB", (33, 77))]
        one = model.predict(processor, images, return_type="alpha", batch_size=1)
        two = model.predict(processor, images, return_type="alpha", batch_size=2)
        for a, b in zip(one, two):
            assert torch.allclose(a, b, atol=1e-5)

    def test_accepts_a_path(self, model, processor, tmp_path):
        path = tmp_path / "in.png"
        Image.new("RGB", (60, 40), (1, 2, 3)).save(path)
        assert model.predict(processor, str(path)).size == (60, 40)

    def test_accepts_a_numpy_array(self, model, processor):
        arr = np.zeros((40, 60, 3), dtype=np.uint8)
        assert model.predict(processor, arr).size == (60, 40)

    def test_empty_list_returns_empty_list(self, model, processor):
        assert model.predict(processor, []) == []

    def test_matches_the_manual_pipeline(self, model, processor):
        model.eval()
        image = Image.new("RGB", (60, 40), (90, 100, 110))
        inputs = processor(images=[image], return_tensors="pt")
        with torch.no_grad():
            out = model(pixel_values=inputs["pixel_values"])
        expected = processor.post_process_alpha_matting(
            out, target_sizes=[(image.height, image.width)]
        )[0]
        got = model.predict(processor, image, return_type="alpha")
        assert torch.equal(expected, got)

    def test_restores_training_mode(self, model, processor):
        model.train()
        model.predict(processor, Image.new("RGB", (60, 40)))
        assert model.training
        model.eval()
        model.predict(processor, Image.new("RGB", (60, 40)))
        assert not model.training

    def test_rejects_a_preprocessed_batch(self, model, processor):
        inputs = processor(images=Image.new("RGB", (60, 40)), return_tensors="pt")
        with pytest.raises(TypeError, match="raw images"):
            model.predict(processor, inputs)

    def test_rejects_bad_arguments(self, model, processor):
        image = Image.new("RGB", (60, 40))
        with pytest.raises(ValueError, match="return_type"):
            model.predict(processor, image, return_type="rgb")
        with pytest.raises(ValueError, match="batch_size"):
            model.predict(processor, image, batch_size=0)

    def test_signature_stops_at_image(self, model, processor):
        # BiRefNet's processor takes neither text nor boxes, so a third
        # positional argument is a caller error, not something to swallow.
        with pytest.raises(TypeError):
            model.predict(processor, Image.new("RGB", (60, 40)), "the dog")


class TestBiRefNetProcess:
    """`process` is `predict` with the processor built from the model's own config."""

    @pytest.fixture
    def model(self):
        return BiRefNet(
            config=BiRefNetConfig(
                image_size=128,
                patch_size=4,
                embed_dim=16,
                num_layers=4,
                depths=[1, 1, 1, 1],
                num_heads=[1, 2, 4, 8],
                window_size=2,
                dec_channels_inter=8,
            )
        )

    def test_default_processor_follows_the_config(self, model):
        proc = model.default_processor()
        assert isinstance(proc, BiRefNetImageProcessor)
        assert proc.size["height"] == 128 and proc.size["width"] == 128

    def test_default_processor_tracks_a_changed_image_size(self):
        model = BiRefNet(config=BiRefNetConfig(image_size=64, window_size=2))
        assert model.default_processor().size["height"] == 64

    def test_default_processor_is_not_shared(self, model):
        # Fresh each call, so mutating one never leaks into a later `process`.
        assert model.default_processor() is not model.default_processor()

    def test_single_image_returns_cutout(self, model):
        cut = model.process(Image.new("RGB", (60, 40), (10, 20, 30)))
        assert isinstance(cut, Image.Image)
        assert cut.mode == "RGBA"
        assert cut.size == (60, 40)

    def test_accepts_a_path(self, model, tmp_path):
        path = tmp_path / "in.png"
        Image.new("RGB", (60, 40), (1, 2, 3)).save(path)
        assert model.process(str(path)).size == (60, 40)

    def test_list_returns_list_at_original_sizes(self, model):
        images = [Image.new("RGB", (60, 40)), Image.new("RGB", (33, 77))]
        cuts = model.process(images, batch_size=2)
        assert [c.size for c in cuts] == [(60, 40), (33, 77)]

    def test_matches_predict_with_the_same_processor(self, model):
        image = Image.new("RGB", (60, 40), (90, 100, 110))
        processor = BiRefNetImageProcessor(size={"height": 128, "width": 128})
        expected = model.predict(processor, image, return_type="alpha")
        assert torch.equal(expected, model.process(image, return_type="alpha"))

    def test_processor_kwargs_are_forwarded(self, model):
        # `size` reaches the processor's __call__, overriding the class default.
        alpha = model.process(
            Image.new("RGB", (60, 40)),
            return_type="alpha",
            size={"height": 64, "width": 64},
        )
        assert alpha.shape == (40, 60)

    def test_rejects_bad_arguments(self, model):
        with pytest.raises(ValueError, match="return_type"):
            model.process(Image.new("RGB", (60, 40)), return_type="rgb")

    def test_signature_stops_at_image(self, model):
        with pytest.raises(TypeError):
            model.process(Image.new("RGB", (60, 40)), "the dog")
