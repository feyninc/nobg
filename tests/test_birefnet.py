import json
import os
from types import SimpleNamespace

import pytest
import torch

from nobg import AutoModel, BiRefNet, auto
from nobg.birefnet.modeling_birefnet import BiRefNetConfig


class TestBiRefNet:
    @pytest.fixture
    def small_config(self):
        return BiRefNetConfig(
            image_size=128,
            patch_size=4,
            embed_dim=16,
            num_layers=4,
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
        assert model.config.num_layers == 4
        assert model.config.nobg_version

    def test_init_custom_config(self, small_model):
        assert small_model.config.embed_dim == 16
        assert small_model.config.image_size == 128
        assert small_model.config.depths == [1, 1, 1, 1]
        assert small_model.config.window_size == 2

    def test_config_num_layers_mismatch_raises(self):
        with pytest.raises(ValueError):
            BiRefNetConfig(num_layers=3, depths=[1, 1, 1, 1], num_heads=[1, 2, 4, 8])

    def test_forward(self, small_model):
        small_model.eval()
        x = torch.randn(1, 3, 128, 128)
        with torch.no_grad():
            out = small_model(x)
        assert out["logits"].shape == (1, 1, 128, 128)
        # multi-scale supervision heads: one per non-final decoder stage
        assert len(out["intermediate_logits"]) == 3
        for m in out["intermediate_logits"]:
            assert m.shape[:2] == (1, 1)

    def test_forward_batch(self, small_model):
        small_model.eval()
        x = torch.randn(2, 3, 128, 128)
        with torch.no_grad():
            out = small_model(x)
        assert out["logits"].shape == (2, 1, 128, 128)

    def test_forward_with_labels_returns_loss(self, small_model):
        small_model.eval()
        x = torch.randn(1, 3, 128, 128)
        labels = (torch.rand(1, 1, 128, 128) > 0.5).float()
        out = small_model(x, labels=labels)
        assert out["loss"].ndim == 0
        assert torch.isfinite(out["loss"])

    def test_criterion_hot_swap(self, small_model):
        calls = []

        def fake_loss(scaled_preds, gt):
            calls.append(len(scaled_preds))
            return torch.tensor(1.234)

        small_model.criterion = fake_loss
        small_model.eval()
        x = torch.randn(1, 3, 128, 128)
        labels = torch.zeros(1, 1, 128, 128)
        out = small_model(x, labels=labels)
        assert out["loss"].item() == pytest.approx(1.234)
        assert calls == [4]  # 3 intermediate + final
        # criterion is a plain attribute, never serialized
        assert "criterion" not in small_model.state_dict()

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
        assert saved_config["num_layers"] == 4
        assert "nobg_version" in saved_config
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

    def test_from_origin_same_shape(self, small_model, tmp_path):
        """from_origin of a same-config model reproduces its outputs."""
        small_model.eval()
        small_model.save_pretrained(tmp_path)
        grown = BiRefNet.from_origin(tmp_path)
        grown.eval()
        x = torch.randn(1, 3, 128, 128)
        with torch.no_grad():
            a = small_model(x)["logits"]
            b = grown(x)["logits"]
        assert torch.allclose(a, b, atol=1e-5)

    def test_from_origin_growth(self, small_model, tmp_path):
        """Growing depths keeps old blocks' weights; extra blocks are fresh."""
        small_model.save_pretrained(tmp_path)
        grown = BiRefNet.from_origin(tmp_path, depths=[1, 1, 2, 1])
        assert grown.config.depths == [1, 1, 2, 1]
        assert grown.config.num_layers == 4
        old_sd = small_model.state_dict()
        new_sd = grown.state_dict()
        # block 0 of stage 3 was injected from the origin
        key = "bb.swin.encoder.layers.2.blocks.0.mlp.fc1.weight"
        assert torch.equal(old_sd[key], new_sd[key])
        # the grown block exists and runs
        assert "bb.swin.encoder.layers.2.blocks.1.mlp.fc1.weight" in new_sd
        grown.eval()
        with torch.no_grad():
            out = grown(torch.randn(1, 3, 128, 128))
        assert torch.isfinite(out["logits"]).all()

    def test_from_origin_from_module_instance(self, small_model):
        grown = BiRefNet.from_origin(small_model)
        assert grown.config.depths == small_model.config.depths
        small_model.eval()
        grown.eval()
        x = torch.randn(1, 3, 128, 128)
        with torch.no_grad():
            a = small_model(x)["logits"]
            b = grown(x)["logits"]
        assert torch.allclose(a, b, atol=1e-5)


class TestAutoModel:
    def test_automodel_raises_on_unknown(self, monkeypatch):
        monkeypatch.setattr(
            auto, "model_info", lambda *a, **k: SimpleNamespace(tags=["not-a-nobg-tag"])
        )
        with pytest.raises(ValueError, match="not part of nobg"):
            AutoModel.from_pretrained("nonexistent/repo-that-does-not-exist")
