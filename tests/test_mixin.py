import json
from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch

from nobg import BiRefNet, OnnxModel, Sam3, mixin
from nobg.birefnet.modeling_birefnet import BiRefNetConfig
from nobg.sam3.modeling_sam3 import Sam3Config

# The export needs onnxscript (torch.export path) and the runtime needs
# onnxruntime; both ship in the `onnx` extra.
pytest.importorskip("onnxruntime")
pytest.importorskip("onnxscript")


@pytest.fixture(scope="module")
def birefnet():
    """A real-but-tiny BiRefNet, matching tests/test_birefnet.py's fixture."""
    return BiRefNet(
        BiRefNetConfig(
            image_size=128,
            patch_size=4,
            embed_dim=16,
            num_layers=4,
            depths=[1, 1, 1, 1],
            num_heads=[1, 2, 4, 8],
            window_size=2,
            dec_channels_inter=8,
        )
    ).eval()


@pytest.fixture(scope="module")
def birefnet_export(tmp_path_factory, birefnet):
    """Export once for the whole module — tracing a Swin backbone is slow."""
    directory = tmp_path_factory.mktemp("birefnet_onnx")
    path = birefnet.onnx_save_pretrained(directory)
    return directory, path


@pytest.fixture(scope="module")
def birefnet_onnx(birefnet_export):
    directory, _ = birefnet_export
    return BiRefNet.onnx_from_pretrained(directory, providers=["CPUExecutionProvider"])


@pytest.fixture(scope="module")
def sam3():
    """A real-but-tiny SAM3, matching tests/test_sam3.py's fixture."""
    return Sam3(
        Sam3Config(
            image_size=112,
            vision_hidden_size=32,
            vision_intermediate_size=64,
            vision_num_hidden_layers=2,
            vision_num_attention_heads=2,
            vision_patch_size=14,
            vision_window_size=4,
            vision_global_attn_indexes=[1],
            vision_pretrain_image_size=112,
            fpn_hidden_size=32,
            hidden_size=32,
            intermediate_size=64,
            num_attention_heads=2,
            geometry_num_layers=1,
            detr_encoder_num_layers=1,
            detr_decoder_num_layers=1,
            num_queries=10,
            num_upsampling_stages=2,
            text_vocab_size=1000,
            text_hidden_size=32,
            text_intermediate_size=64,
            text_projection_dim=32,
            text_num_hidden_layers=2,
            text_num_attention_heads=2,
        )
    ).eval()


class TestOnnxMixin:
    def test_save_pretrained_writes_the_export(self, birefnet_export):
        directory, path = birefnet_export
        assert path == str(directory / "model.onnx")
        assert (directory / "model.onnx").is_file()
        assert (directory / "config.json").is_file()
        assert (directory / "README.md").is_file()

    def test_save_pretrained_config_content(self, birefnet, birefnet_export):
        directory, _ = birefnet_export
        with open(directory / "config.json") as f:
            saved = json.load(f)
        assert saved == asdict(birefnet.config)

    def test_save_pretrained_model_card_is_tagged_onnx(self, birefnet_export):
        directory, _ = birefnet_export
        card = (directory / "README.md").read_text()
        front_matter = card.split("---")[1]
        assert "- onnx" in front_matter
        # the nobg dispatch tags survive, so AutoProcessor still resolves the repo
        assert "- nobg-birefnet" in front_matter
        assert "onnx_from_pretrained" in card

    def test_save_pretrained_custom_file_name(self, birefnet, tmp_path):
        path = birefnet.onnx_save_pretrained(tmp_path, file_name="matte.onnx")
        assert path == str(tmp_path / "matte.onnx")
        loaded = BiRefNet.onnx_from_pretrained(
            tmp_path, file_name="matte.onnx", providers=["CPUExecutionProvider"]
        )
        assert loaded.input_names == ("pixel_values",)

    def test_save_pretrained_restores_training_mode(self, birefnet, tmp_path):
        birefnet.train()
        try:
            birefnet.onnx_save_pretrained(tmp_path)
            assert birefnet.training
        finally:
            birefnet.eval()

    def test_save_pretrained_exports_the_eval_graph(self, birefnet, tmp_path):
        """A model left in train mode must still export its inference graph.

        BiRefNet's decoder has dropout and batch norm, so a traced train-mode
        graph would not match the model users get from `from_pretrained`.
        """
        birefnet.train()
        try:
            birefnet.onnx_save_pretrained(tmp_path)
        finally:
            birefnet.eval()
        loaded = BiRefNet.onnx_from_pretrained(
            tmp_path, providers=["CPUExecutionProvider"]
        )
        pixel_values = torch.randn(1, 3, 128, 128)
        with torch.no_grad():
            expected = birefnet(pixel_values)["logits"]
        actual = loaded(pixel_values=pixel_values)["logits"]
        assert torch.allclose(actual, expected, atol=1e-4)

    def test_from_pretrained_returns_an_onnx_model(self, birefnet, birefnet_onnx):
        assert isinstance(birefnet_onnx, OnnxModel)
        assert birefnet_onnx.input_names == ("pixel_values",)
        assert birefnet_onnx.output_names == ("logits",)
        assert birefnet_onnx.batch_size == 1
        assert birefnet_onnx.providers == ["CPUExecutionProvider"]
        assert "BiRefNet" in repr(birefnet_onnx)

    def test_from_pretrained_rebuilds_the_config_dataclass(
        self, birefnet, birefnet_onnx
    ):
        assert isinstance(birefnet_onnx.config, BiRefNetConfig)
        assert birefnet_onnx.config == birefnet.config

    def test_from_pretrained_without_an_export_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="model.onnx"):
            BiRefNet.onnx_from_pretrained(tmp_path)

    def test_logits_match_the_torch_model(self, birefnet, birefnet_onnx):
        pixel_values = torch.randn(1, 3, 128, 128)
        with torch.no_grad():
            expected = birefnet(pixel_values)["logits"]
        actual = birefnet_onnx(pixel_values=pixel_values)["logits"]
        assert actual.shape == expected.shape
        assert torch.allclose(actual, expected, atol=1e-4)

    def test_forward_casts_and_accepts_numpy(self, birefnet, birefnet_onnx):
        """A float64 tensor is cast to the graph's dtype rather than rejected."""
        pixel_values = torch.randn(1, 3, 128, 128, dtype=torch.float64)
        actual = birefnet_onnx(pixel_values=pixel_values)["logits"]
        from_numpy = birefnet_onnx(pixel_values=pixel_values.to(torch.float32).numpy())[
            "logits"
        ]
        assert torch.equal(actual, from_numpy)

    def test_forward_rejects_unknown_inputs(self, birefnet_onnx):
        with pytest.raises(TypeError, match="unexpected input"):
            birefnet_onnx(pixel_values=torch.randn(1, 3, 128, 128), score_threshold=0.5)

    def test_forward_rejects_missing_inputs(self, birefnet_onnx):
        with pytest.raises(ValueError, match="missing input"):
            birefnet_onnx()

    def test_train_raises(self, birefnet_onnx):
        assert birefnet_onnx.eval() is birefnet_onnx
        assert birefnet_onnx.train(False) is birefnet_onnx
        with pytest.raises(RuntimeError, match="cannot be trained"):
            birefnet_onnx.train()

    def test_predict_returns_a_cutout(self, birefnet, birefnet_onnx):
        from PIL import Image

        image = Image.new("RGB", (60, 40), "white")
        cutout = birefnet_onnx.predict(birefnet.default_processor(), image)
        assert cutout.mode == "RGBA"
        assert cutout.size == (60, 40)

    def test_predict_alpha_matches_the_torch_model(self, birefnet, birefnet_onnx):
        from PIL import Image

        image = Image.new("RGB", (50, 30), "gray")
        processor = birefnet.default_processor()
        expected = birefnet.predict(processor, image, return_type="alpha")
        actual = birefnet_onnx.predict(processor, image, return_type="alpha")
        assert actual.shape == expected.shape
        assert torch.allclose(actual, expected, atol=1e-4)

    def test_process_builds_the_default_processor(self, birefnet_onnx):
        from PIL import Image

        matte = birefnet_onnx.process(Image.new("RGB", (32, 32)), return_type="alpha")
        assert matte.shape == (32, 32)
        assert float(matte.min()) >= 0.0 and float(matte.max()) <= 1.0

    def test_process_without_a_config_raises(self, birefnet_onnx):
        headless = OnnxModel(birefnet_onnx.session, model_class=BiRefNet)
        with pytest.raises(RuntimeError, match="cannot be inferred"):
            headless.process(object())

    def test_decode_config_falls_back_to_a_namespace(self):
        """An unannotated `config` parameter still yields attribute access."""

        class Unannotated(BiRefNet):
            pass

        Unannotated._hub_mixin_init_parameters = {}
        decoded = Unannotated._onnx_decode_config({"image_size": 64})
        assert isinstance(decoded, SimpleNamespace)
        assert decoded.image_size == 64

    def test_push_to_hub_prefixes_the_username_and_uploads_the_folder(
        self, birefnet, monkeypatch
    ):
        import huggingface_hub

        uploaded = {}

        class FakeApi:
            def __init__(self, token=None):
                uploaded["token"] = token

            def create_repo(self, repo_id, private=None, exist_ok=False):
                uploaded["created"] = repo_id
                return SimpleNamespace(repo_id=repo_id)

            def upload_folder(self, *, repo_id, folder_path, **kwargs):
                uploaded["repo_id"] = repo_id
                uploaded["files"] = sorted(p.name for p in folder_path.iterdir())
                uploaded["card"] = (folder_path / "README.md").read_text()
                return f"https://huggingface.co/{repo_id}/commit/deadbeef"

        monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
        monkeypatch.setattr(mixin, "whoami", lambda token=None: {"name": "tester"})

        url = birefnet.onnx_push_to_hub("birefnet-onnx", token="hf_xxx")

        assert uploaded["created"] == "tester/birefnet-onnx"
        assert uploaded["repo_id"] == "tester/birefnet-onnx"
        assert uploaded["token"] == "hf_xxx"
        assert "model.onnx" in uploaded["files"]
        assert "config.json" in uploaded["files"]
        # the card is rendered for the resolved repo, not a placeholder
        assert "tester/birefnet-onnx" in uploaded["card"]
        assert url.endswith("deadbeef")

    def test_push_to_hub_keeps_the_torch_card_untagged(self, birefnet, tmp_path):
        """The `onnx` tag is per-export, not a mutation of the class's card data."""
        birefnet.onnx_save_pretrained(tmp_path / "onnx")
        birefnet.save_pretrained(tmp_path / "torch")
        torch_card = (tmp_path / "torch" / "README.md").read_text()
        assert "- onnx" not in torch_card.split("---")[1]


class TestSam3Onnx:
    """SAM3 exercises the other export path: prompt inputs, TorchScript tracer."""

    def test_dummy_inputs_include_the_prompt(self, sam3):
        inputs = sam3.onnx_dummy_inputs()
        assert list(inputs) == ["pixel_values", "input_ids", "attention_mask"]
        assert inputs["pixel_values"].shape == (1, 3, 112, 112)
        # Sam3Processor pads text to 32 tokens, so the graph is traced at that
        length = sam3.config.text_max_position_embeddings
        assert inputs["input_ids"].shape == (1, length)
        assert inputs["input_ids"].dtype == torch.long
        assert inputs["attention_mask"].tolist() == [[1, 1] + [0] * (length - 2)]

    def test_dummy_inputs_batch_size(self, sam3):
        inputs = sam3.onnx_dummy_inputs(3)
        assert inputs["pixel_values"].shape[0] == 3
        assert inputs["input_ids"].shape[0] == 3

    def test_uses_the_torchscript_exporter(self):
        assert Sam3.onnx_dynamo is False
        assert BiRefNet.onnx_dynamo is True

    def test_roundtrip_matches_the_torch_model(self, sam3, tmp_path):
        sam3.onnx_save_pretrained(tmp_path)
        loaded = Sam3.onnx_from_pretrained(tmp_path, providers=["CPUExecutionProvider"])
        assert isinstance(loaded.config, Sam3Config)
        assert loaded.config == sam3.config
        assert loaded.input_names == ("pixel_values", "input_ids", "attention_mask")

        # A prompt the graph was *not* traced with: only its shape is baked in.
        inputs = sam3.onnx_dummy_inputs()
        inputs["input_ids"][:, 1:4] = torch.tensor([101, 102, 999])
        inputs["attention_mask"][:, :4] = 1
        with torch.no_grad():
            expected = sam3(**inputs)["logits"]
        actual = loaded(**inputs)["logits"]
        assert actual.shape == expected.shape
        assert torch.allclose(actual, expected, atol=1e-4)
