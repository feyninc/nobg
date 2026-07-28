import json
import os

import numpy as np
import pytest
import torch
from PIL import Image

from nobg import AutoProcessor, BiRefNet, BiRefNetImageProcessor
from nobg.birefnet.modeling_birefnet import BiRefNetConfig


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
