import json
import shutil
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
def birefnet_subfolder_export(tmp_path_factory, birefnet):
    """Export into `onnx/` once — the layout of a shared checkpoint repo."""
    directory = tmp_path_factory.mktemp("birefnet_onnx_subfolder")
    path = birefnet.onnx_save_pretrained(directory, subfolder=mixin.ONNX_SUBFOLDER)
    return directory, path


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

    def test_save_pretrained_subfolder_is_self_contained(
        self, birefnet_subfolder_export
    ):
        """`onnx/` carries its own config, so it can be uploaded on its own."""
        directory, path = birefnet_subfolder_export
        assert path == str(directory / "onnx" / "model.onnx")
        assert (directory / "onnx" / "model.onnx").is_file()
        assert (directory / "onnx" / "config.json").is_file()
        # no card inside the subfolder: the repo it joins already has one, and the
        # Hub renders the root README anyway
        assert not (directory / "onnx" / "README.md").exists()
        # nothing at all outside the subfolder
        assert sorted(p.name for p in directory.iterdir()) == ["onnx"]

    def test_save_pretrained_subfolder_config_content(
        self, birefnet, birefnet_subfolder_export
    ):
        directory, _ = birefnet_subfolder_export
        with open(directory / "onnx" / "config.json") as f:
            saved = json.load(f)
        assert saved == asdict(birefnet.config)

    def test_save_pretrained_model_card_false_writes_none(self, birefnet, tmp_path):
        birefnet.onnx_save_pretrained(tmp_path, model_card=False)
        assert (tmp_path / "model.onnx").is_file()
        assert (tmp_path / "config.json").is_file()
        assert not (tmp_path / "README.md").exists()

    def test_save_pretrained_model_card_true_overrides_the_subfolder_default(
        self, birefnet, tmp_path
    ):
        """`model_card=True` beats the "no card in a subfolder" default."""
        birefnet.onnx_save_pretrained(tmp_path, subfolder="onnx", model_card=True)
        card = (tmp_path / "onnx" / "README.md").read_text()
        assert "- onnx" in card.split("---")[1]
        # the snippet names the subfolder, or the load would not find the graph
        assert 'onnx_from_pretrained("<repo id>", subfolder="onnx")' in card
        assert "`onnx/` folder" in card

    def test_from_pretrained_subfolder_matches_the_torch_model(
        self, birefnet, birefnet_subfolder_export
    ):
        directory, _ = birefnet_subfolder_export
        loaded = BiRefNet.onnx_from_pretrained(
            directory, subfolder="onnx", providers=["CPUExecutionProvider"]
        )
        assert isinstance(loaded, OnnxModel)
        assert loaded.config == birefnet.config
        pixel_values = torch.randn(1, 3, 128, 128)
        with torch.no_grad():
            expected = birefnet(pixel_values)["logits"]
        actual = loaded(pixel_values=pixel_values)["logits"]
        assert actual.shape == expected.shape
        assert torch.allclose(actual, expected, atol=1e-4)

    def test_from_pretrained_without_a_subfolder_raises(
        self, birefnet_subfolder_export
    ):
        """The error names `subfolder=`, since that is the fix here."""
        directory, _ = birefnet_subfolder_export
        with pytest.raises(FileNotFoundError, match="subfolder="):
            BiRefNet.onnx_from_pretrained(directory)

    def test_from_pretrained_subfolder_falls_back_to_the_root_config(
        self, birefnet, birefnet_subfolder_export, tmp_path
    ):
        """A repo keeps `config.json` at its root; the subfolder may have none."""
        directory, _ = birefnet_subfolder_export
        shutil.copytree(directory, tmp_path / "repo")
        (tmp_path / "repo" / "onnx" / "config.json").rename(
            tmp_path / "repo" / "config.json"
        )
        loaded = BiRefNet.onnx_from_pretrained(
            tmp_path / "repo", subfolder="onnx", providers=["CPUExecutionProvider"]
        )
        assert isinstance(loaded.config, BiRefNetConfig)
        assert loaded.config == birefnet.config

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

    def test_push_to_hub_subfolder_scopes_the_upload_and_patches_the_card(
        self, birefnet, monkeypatch
    ):
        """A subfolder push touches `onnx/` plus the root card, nothing else."""
        import huggingface_hub
        from huggingface_hub import ModelCard, ModelCardData

        uploaded = {}
        card = ModelCard("# sam3-prompted\n\nA hand-written card.\n")
        card.data = ModelCardData(tags=["nobg", "nobg-birefnet"])
        pushed = []

        class FakeApi:
            def __init__(self, token=None):
                pass

            def create_repo(self, repo_id, private=None, exist_ok=False):
                return SimpleNamespace(repo_id=repo_id)

            def upload_folder(self, *, repo_id, folder_path, path_in_repo=None, **kw):
                uploaded["path_in_repo"] = path_in_repo
                uploaded["folder_name"] = folder_path.name
                uploaded["files"] = sorted(p.name for p in folder_path.iterdir())
                return f"https://huggingface.co/{repo_id}/commit/deadbeef"

        monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
        monkeypatch.setattr(
            ModelCard, "load", classmethod(lambda cls, repo_id, token=None: card)
        )
        monkeypatch.setattr(
            ModelCard,
            "push_to_hub",
            lambda self, repo_id, **kwargs: pushed.append(repo_id),
        )

        url = birefnet.onnx_push_to_hub("tester/birefnet", subfolder="onnx")

        # the folder handed to `upload_folder` is the subfolder itself, landing at
        # `onnx/` in the repo -- so the torch weights beside it are never in the
        # commit, and neither is a root README
        assert uploaded["path_in_repo"] == "onnx"
        assert uploaded["folder_name"] == "onnx"
        assert "model.onnx" in uploaded["files"]
        assert "config.json" in uploaded["files"]
        assert "README.md" not in uploaded["files"]
        assert url.endswith("deadbeef")

        # the existing card is patched in place, in a separate commit
        assert pushed == ["tester/birefnet"]
        assert card.data.tags == ["nobg", "nobg-birefnet", "onnx"]
        assert card.text.startswith("# sam3-prompted")
        assert 'onnx_from_pretrained("tester/birefnet", subfolder="onnx")' in card.text

    def test_patch_model_card_is_idempotent(self, birefnet, monkeypatch):
        """Re-pushing must not stack a second copy of the usage section."""
        from huggingface_hub import ModelCard, ModelCardData

        card = ModelCard("# birefnet\n\nA hand-written card.\n")
        card.data = ModelCardData(tags=["nobg"])
        monkeypatch.setattr(
            ModelCard, "load", classmethod(lambda cls, repo_id, token=None: card)
        )
        monkeypatch.setattr(ModelCard, "push_to_hub", lambda self, repo_id, **kw: None)

        for _ in range(2):
            birefnet._onnx_patch_model_card("tester/birefnet", subfolder="onnx")
        assert card.text.count(mixin._ONNX_USAGE_HEADING) == 1
        assert card.data.tags == ["nobg", "onnx"]

    def test_patch_model_card_generates_one_when_the_repo_has_none(
        self, birefnet, monkeypatch
    ):
        """A repo the export is the first thing in still gets a card."""
        from huggingface_hub import ModelCard
        from huggingface_hub.utils import EntryNotFoundError

        def missing(cls, repo_id, token=None):
            raise EntryNotFoundError("no README.md in this repo")

        pushed = []
        monkeypatch.setattr(ModelCard, "load", classmethod(missing))
        monkeypatch.setattr(
            ModelCard, "push_to_hub", lambda self, repo_id, **kw: pushed.append(self)
        )

        birefnet._onnx_patch_model_card("tester/birefnet", subfolder="onnx")

        (card,) = pushed
        assert "onnx" in card.data.tags
        assert 'onnx_from_pretrained("tester/birefnet", subfolder="onnx")' in card.text

    def test_push_to_hub_survives_a_broken_model_card(self, birefnet, monkeypatch):
        """The weights are already up, so a card failure is a warning, not a raise."""
        import huggingface_hub
        from huggingface_hub import ModelCard

        def explode(cls, repo_id, token=None):
            raise RuntimeError("the Hub is having a day")

        class FakeApi:
            def __init__(self, token=None):
                pass

            def create_repo(self, repo_id, private=None, exist_ok=False):
                return SimpleNamespace(repo_id=repo_id)

            def upload_folder(self, *, repo_id, folder_path, **kwargs):
                return f"https://huggingface.co/{repo_id}/commit/deadbeef"

        monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
        monkeypatch.setattr(ModelCard, "load", classmethod(explode))

        url = birefnet.onnx_push_to_hub("tester/birefnet", subfolder="onnx")
        assert url.endswith("deadbeef")

    def test_push_to_hub_keeps_the_torch_card_untagged(self, birefnet, tmp_path):
        """The `onnx` tag is per-export, not a mutation of the class's card data."""
        birefnet.onnx_save_pretrained(tmp_path / "onnx")
        birefnet.save_pretrained(tmp_path / "torch")
        torch_card = (tmp_path / "torch" / "README.md").read_text()
        assert "- onnx" not in torch_card.split("---")[1]


class TestOnnxExternalData:
    """`_onnx_consolidate_external_data`, the fix for per-tensor sidecars.

    Only a >2 GB export actually spills, which is far too big for a test, so the
    spilled layout is built directly with `onnx.save_model`: `size_threshold=0`
    plus `all_tensors_to_one_file=False` is exactly what the TorchScript
    exporter's C++ serializer produces, one file named after each tensor. On
    `nobg/sam3-prompted` that is 609 files, initializers and graph `Constant`
    attributes both, so both kinds are covered here.
    """

    @pytest.fixture
    def constant_graph(self):
        """A graph whose weights are half initializer, half `Constant` attribute.

        The real export's sidecar names include
        `..._attention_Constant_24_attr__value`, i.e. the spill is not limited to
        initializers -- so the tensor walk behind the delete list has to descend
        into node attributes, and this is what proves it does.
        """
        import numpy as np
        from onnx import TensorProto, helper, numpy_helper

        rng = np.random.default_rng(1)
        value = numpy_helper.from_array(rng.random((32, 32), dtype=np.float32), "value")
        weight = numpy_helper.from_array(rng.random((32, 32), dtype=np.float32), "w")
        return helper.make_model(
            helper.make_graph(
                [
                    helper.make_node("Constant", [], ["c"], value=value),
                    helper.make_node("MatMul", ["x", "w"], ["h"]),
                    helper.make_node("MatMul", ["h", "c"], ["y"]),
                ],
                "constant_chain",
                [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 32])],
                [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 32])],
                [weight],
            ),
            opset_imports=[helper.make_opsetid("", mixin.DEFAULT_ONNX_OPSET)],
        )

    @pytest.fixture
    def graph(self):
        """A chain of `MatMul`s, i.e. a graph that is mostly initializers."""
        import numpy as np
        from onnx import TensorProto, helper, numpy_helper

        # 4 KB each, comfortably over `onnx.save_model`'s 1 KB size threshold, so
        # they really do land in sidecars instead of staying inline.
        rng = np.random.default_rng(0)
        weights = [
            numpy_helper.from_array(rng.random((32, 32), dtype=np.float32), f"w_{i}")
            for i in range(4)
        ]
        nodes, previous = [], "x"
        for i, weight in enumerate(weights):
            output = "y" if i == len(weights) - 1 else f"hidden_{i}"
            nodes.append(helper.make_node("MatMul", [previous, weight.name], [output]))
            previous = output
        return helper.make_model(
            helper.make_graph(
                nodes,
                "chain",
                [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 32])],
                [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 32])],
                weights,
            ),
            opset_imports=[helper.make_opsetid("", mixin.DEFAULT_ONNX_OPSET)],
        )

    def _run(self, path):
        import numpy as np
        import onnxruntime as ort

        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        return session.run(None, {"x": np.ones((1, 32), dtype=np.float32)})[0]

    def test_collapses_per_tensor_sidecars(self, graph, tmp_path):
        import onnx

        path = tmp_path / "model.onnx"
        onnx.save_model(
            graph,
            path,
            save_as_external_data=True,
            all_tensors_to_one_file=False,
            size_threshold=0,
        )
        assert sorted(p.name for p in tmp_path.iterdir()) == [
            "model.onnx",
            "w_0",
            "w_1",
            "w_2",
            "w_3",
        ]
        expected = self._run(path)

        data_path = mixin._onnx_consolidate_external_data(path)

        assert data_path == tmp_path / "model.onnx.data"
        assert sorted(p.name for p in tmp_path.iterdir()) == [
            "model.onnx",
            "model.onnx.data",
        ]
        assert (tmp_path / "model.onnx.data").stat().st_size > 0
        # same weights, same answer -- the rewrite is a relayout, not a re-quantize
        assert self._run(path) == pytest.approx(expected)

    def test_collapses_constant_attribute_sidecars(self, constant_graph, tmp_path):
        """A `Constant` node's spilled attribute is collected and cleaned up too."""
        import onnx

        path = tmp_path / "model.onnx"
        onnx.save_model(
            constant_graph,
            path,
            save_as_external_data=True,
            all_tensors_to_one_file=False,
            size_threshold=0,
            convert_attribute=True,
        )
        # one sidecar for the initializer, one for the Constant's tensor attribute
        assert len(list(tmp_path.iterdir())) == 3
        expected = self._run(path)

        assert mixin._onnx_consolidate_external_data(path) == (
            tmp_path / "model.onnx.data"
        )
        # the attribute lands back inline (`convert_attribute` stays False on the
        # rewrite), so only the initializer is in the data file -- but neither
        # sidecar survives
        assert sorted(p.name for p in tmp_path.iterdir()) == [
            "model.onnx",
            "model.onnx.data",
        ]
        assert self._run(path) == pytest.approx(expected)

    def test_tensor_walk_matches_onnx_s_own(self, constant_graph, tmp_path):
        """Drift guard on `_onnx_all_tensors` vs the private helper it replaces.

        A tensor it misses is a sidecar left behind, so the two must agree.
        """
        import onnx
        from onnx.external_data_helper import _get_all_tensors

        path = tmp_path / "model.onnx"
        onnx.save_model(constant_graph, path)
        proto = onnx.load(path, load_external_data=False)

        # Materialize both before comparing identities: the yielded protos are
        # views onto their parent, and a `{id(t) for t in ...}` comprehension can
        # see an id reused once a view is collected mid-iteration.
        mine = list(mixin._onnx_all_tensors(proto))
        theirs = list(_get_all_tensors(proto))

        assert {id(t) for t in mine} == {id(t) for t in theirs}
        assert sorted(t.name for t in mine) == ["value", "w"]

    def test_leaves_a_single_data_file_alone(self, graph, tmp_path):
        import onnx

        path = tmp_path / "model.onnx"
        onnx.save_model(
            graph,
            path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location="model.onnx.data",
            size_threshold=0,
        )
        before = (path.stat().st_mtime_ns, path.read_bytes())

        assert mixin._onnx_consolidate_external_data(path) == (
            tmp_path / "model.onnx.data"
        )
        assert (path.stat().st_mtime_ns, path.read_bytes()) == before

    def test_returns_none_without_external_data(self, graph, tmp_path):
        """A graph small enough to hold its own weights needs no data file."""
        import onnx

        path = tmp_path / "model.onnx"
        onnx.save_model(graph, path)
        before = (path.stat().st_mtime_ns, path.read_bytes())

        assert mixin._onnx_consolidate_external_data(path) is None
        assert (path.stat().st_mtime_ns, path.read_bytes()) == before
        assert sorted(p.name for p in tmp_path.iterdir()) == ["model.onnx"]

    def test_is_a_no_op_on_a_normal_export(self, birefnet_export):
        """A real sub-2 GB export is left byte-for-byte as the exporter wrote it.

        `torch.onnx.export`'s `external_data=` defaults to True, so the
        `torch.export` path writes a single `model.onnx.data` for every model,
        however small — which is already the target layout, hence nothing to do.
        """
        directory, path = birefnet_export
        before = {
            p.name: (p.stat().st_mtime_ns, p.stat().st_size)
            for p in directory.iterdir()
        }

        mixin._onnx_consolidate_external_data(path)

        assert {
            p.name: (p.stat().st_mtime_ns, p.stat().st_size)
            for p in directory.iterdir()
        } == before


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
