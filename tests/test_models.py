import json
import os

import pytest
import torch

from nobg import ANN, GPT2, AutoModel
from nobg.modeling_ann import ANNConfig
from nobg.modeling_gpt2 import GPT2Config


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


class TestGPT2:
    @pytest.fixture
    def small_config(self):
        return GPT2Config(
            n_positions=32,
            vocab_size=100,
            n_layer=2,
            n_head=2,
            n_embd=64,
        )

    @pytest.fixture
    def small_model(self, small_config):
        return GPT2(config=small_config)

    def test_init_default(self):
        model = GPT2()
        assert model.config.vocab_size == 50257
        assert model.config.n_embd == 768
        assert model.config.n_positions == 1024
        assert model.config.n_layer == 12
        assert model.config.n_head == 12

    def test_init_custom(self, small_model):
        assert small_model.config.vocab_size == 100
        assert small_model.config.n_embd == 64
        assert small_model.config.n_positions == 32
        assert small_model.config.n_layer == 2
        assert small_model.config.n_head == 2

    def test_forward_shape(self, small_model):
        input_ids = torch.randint(0, 100, (1, 10))
        out = small_model(input_ids=input_ids)
        assert out.shape == (1, 10, 100)

    def test_forward_batch(self, small_model):
        input_ids = torch.randint(0, 100, (4, 16))
        out = small_model(input_ids=input_ids)
        assert out.shape == (4, 16, 100)

    def test_forward_with_labels(self, small_model):
        input_ids = torch.randint(0, 100, (2, 8))
        labels = torch.randint(0, 100, (2, 8))
        out = small_model(input_ids=input_ids, labels=labels)
        assert isinstance(out, dict)
        assert "loss" in out
        assert "logits" in out
        assert out["logits"].shape == (2, 8, 100)
        assert out["loss"].ndim == 0

    def test_forward_exceeds_block_size(self, small_model):
        input_ids = torch.randint(0, 100, (1, 64))
        with pytest.raises(AssertionError, match="cannot forward sequence"):
            small_model(input_ids=input_ids)

    def test_save_pretrained(self, small_model, tmp_path):
        small_model.save_pretrained(tmp_path)
        files = os.listdir(tmp_path)
        assert "model.safetensors" in files
        assert "config.json" in files

    def test_save_pretrained_config_content(self, small_model, tmp_path):
        small_model.save_pretrained(tmp_path)
        with open(tmp_path / "config.json") as f:
            saved_config = json.load(f)
        assert saved_config == {
            "n_positions": 32,
            "vocab_size": 100,
            "n_layer": 2,
            "n_head": 2,
            "n_embd": 64,
        }

    def test_from_pretrained_roundtrip(self, small_model, tmp_path):
        small_model.save_pretrained(tmp_path)
        loaded = GPT2.from_pretrained(tmp_path)
        assert isinstance(loaded.config, GPT2Config)
        assert loaded.config.vocab_size == 100
        assert loaded.config.n_embd == 64
        assert loaded.config.n_positions == 32
        assert loaded.config.n_layer == 2
        assert loaded.config.n_head == 2

    def test_weights_preserved_after_roundtrip(self, small_model, tmp_path):
        small_model.eval()
        input_ids = torch.randint(0, 100, (1, 10))
        with torch.no_grad():
            expected = small_model(input_ids=input_ids)
        small_model.save_pretrained(tmp_path)
        loaded = GPT2.from_pretrained(tmp_path)
        loaded.eval()
        with torch.no_grad():
            actual = loaded(input_ids=input_ids)
        assert torch.allclose(expected, actual)

    def test_generate_returns_tensor(self, small_model):
        input_ids = torch.randint(0, 100, (1, 5))
        attention_mask = torch.ones_like(input_ids)
        out = small_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=3,
        )
        assert isinstance(out, torch.Tensor)
        assert out.shape[1] == 8

    def test_generate_return_generated_only(self, small_model):
        input_ids = torch.randint(0, 100, (1, 5))
        attention_mask = torch.ones_like(input_ids)
        out = small_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=4,
            return_generated_only=True,
        )
        assert isinstance(out, torch.Tensor)
        assert out.shape == (4,)


class TestAutoModel:
    def test_automodel_raises_on_unknown(self):
        with pytest.raises(Exception):
            AutoModel.from_pretrained("nonexistent/repo-that-does-not-exist")
