import json
import os

import pytest
import torch

from nobg import BiRefNet, AutoModel
from nobg.birefnet.modeling_birefnet import BiRefNetConfig


class TestBiRefNet:
    @pytest.fixture
    def small_config(self):
        return BiRefNetConfig(
            image_size=128,
            patch_size=4,
            embed_dim=16,
            depths=[1, 1, 1, 1],
            num_heads=[1, 2, 4, 8],
            window_size=2,
            dec_channels_inter=8,
            use_multi_scale_input=True,
            use_gradient_attention=True,
            use_image_patch_injection=True,
        )

    @pytest.fixture
    def small_model(self, small_config):
        return BiRefNet(config=small_config)

    def test_init_default_config(self):
        model = BiRefNet()
        assert model.config.embed_dim == 192
        assert model.config.image_size == 1024
        assert model.config.depths == [2, 2, 18, 2]
        assert model.config.num_heads == [6, 12, 24, 48]

    def test_init_custom_config(self, small_model):
        assert small_model.config.embed_dim == 16
        assert small_model.config.image_size == 128
        assert small_model.config.depths == [1, 1, 1, 1]
        assert small_model.config.window_size == 2

    def test_forward(self, small_model):
        small_model.eval()
        x = torch.randn(1, 3, 128, 128)
        with torch.no_grad():
            out = small_model(x)
        assert out["logits"].shape == (1, 1, 128, 128)
        assert len(out["intermediate_logits"]) == 3
        for m in out["intermediate_logits"]:
            assert m.shape == (1, 1, 128, 128)

    def test_forward_batch(self, small_model):
        small_model.eval()
        x = torch.randn(2, 3, 128, 128)
        with torch.no_grad():
            out = small_model(x)
        assert out["logits"].shape == (2, 1, 128, 128)

    def test_save_pretrained(self, small_model, tmp_path):
        small_model.save_pretrained(tmp_path)
        files = os.listdir(tmp_path)
        assert "model.safetensors" in files
        assert "config.json" in files

    def test_save_pretrained_config_content(self, small_model, tmp_path):
        small_model.save_pretrained(tmp_path)
        with open(tmp_path / "config.json") as f:
            saved_config = json.load(f)
        assert saved_config["embed_dim"] == 16
        assert saved_config["image_size"] == 128
        assert saved_config["depths"] == [1, 1, 1, 1]
        assert saved_config["num_heads"] == [1, 2, 4, 8]
        assert saved_config["window_size"] == 2
        assert saved_config["dec_channels_inter"] == 8
        assert saved_config["use_multi_scale_input"] is True
        assert saved_config["use_gradient_attention"] is True
        assert saved_config["use_image_patch_injection"] is True

    def test_from_pretrained_roundtrip(self, small_model, tmp_path):
        small_model.save_pretrained(tmp_path)
        loaded = BiRefNet.from_pretrained(tmp_path)
        assert isinstance(loaded.config, BiRefNetConfig)
        assert loaded.config.embed_dim == 16
        assert loaded.config.image_size == 128
        assert loaded.config.depths == [1, 1, 1, 1]
        assert loaded.config.num_heads == [1, 2, 4, 8]

    def test_weights_preserved_after_roundtrip(self, small_model, tmp_path):
        small_model.eval()
        x = torch.randn(1, 3, 128, 128)
        with torch.no_grad():
            expected = small_model(x)
        small_model.save_pretrained(tmp_path)
        loaded = BiRefNet.from_pretrained(tmp_path)
        loaded.eval()
        with torch.no_grad():
            actual = loaded(x)
        assert torch.allclose(expected["logits"], actual["logits"])


class TestAutoModel:
    def test_automodel_raises_on_unknown(self):
        with pytest.raises(Exception):
            AutoModel.from_pretrained("nonexistent/repo-that-does-not-exist")
