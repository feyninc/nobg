import json
import os

import torch

from nobg import ANN
from nobg.ann.modeling_ann import ANNConfig


class TestANN:
    def test_init_default_config(self):
        model = ANN()
        assert model.config.a == 2
        assert model.config.b == 1
        assert isinstance(model.layer, torch.nn.Linear)
        assert model.layer.in_features == 2
        assert model.layer.out_features == 1

    def test_init_custom_config(self):
        cfg = ANNConfig(a=4, b=3)
        model = ANN(config=cfg)
        assert model.config.a == 4
        assert model.config.b == 3
        assert model.layer.in_features == 4
        assert model.layer.out_features == 3

    def test_forward(self):
        model = ANN(config=ANNConfig(a=4, b=2))
        x = torch.randn(1, 4)
        out = model(x)
        assert out.shape == (1, 2)

    def test_forward_batch(self):
        model = ANN(config=ANNConfig(a=3, b=5))
        x = torch.randn(8, 3)
        out = model(x)
        assert out.shape == (8, 5)

    def test_save_pretrained(self, tmp_path):
        model = ANN(config=ANNConfig(a=6, b=2))
        model.save_pretrained(tmp_path)
        files = os.listdir(tmp_path)
        assert "model.safetensors" in files
        assert "config.json" in files

    def test_save_pretrained_config_content(self, tmp_path):
        model = ANN(config=ANNConfig(a=6, b=2))
        model.save_pretrained(tmp_path)
        with open(tmp_path / "config.json") as f:
            saved_config = json.load(f)
        assert saved_config == {"a": 6, "b": 2}

    def test_from_pretrained_roundtrip(self, tmp_path):
        model = ANN(config=ANNConfig(a=5, b=3))
        model.save_pretrained(tmp_path)
        loaded = ANN.from_pretrained(tmp_path)
        assert isinstance(loaded.config, ANNConfig)
        assert loaded.config.a == 5
        assert loaded.config.b == 3
        assert loaded.layer.in_features == 5
        assert loaded.layer.out_features == 3

    def test_weights_preserved_after_roundtrip(self, tmp_path):
        model = ANN(config=ANNConfig(a=4, b=2))
        x = torch.randn(1, 4)
        expected = model(x)
        model.save_pretrained(tmp_path)
        loaded = ANN.from_pretrained(tmp_path)
        actual = loaded(x)
        assert torch.allclose(expected, actual)
