import json
import os
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from nobg import AutoModel, Sam3, auto
from nobg.loss import birefnet_loss, dice_loss, sam3_loss, sigmoid_focal_loss
from nobg.sam3 import modeling_sam3
from nobg.sam3.modeling_sam3 import (
    Sam3Config,
    _fields_from_hf_config,
    _remap_upstream_state_dict,
)


@pytest.fixture
def small_config():
    """A real-but-tiny SAM3 (~0.27 M params) so the suite stays fast and offline."""
    return Sam3Config(
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


@pytest.fixture
def small_model(small_config):
    return Sam3(config=small_config).eval()


@pytest.fixture
def instance_model(small_config):
    """A model on the `aggregate="max"` path — the instance-union branch.

    The default is `"semantic"`, so the query-scoring/threshold behaviour has to
    be exercised explicitly.
    """
    small_config.aggregate = "max"
    return Sam3(config=small_config).eval()


@pytest.fixture
def inputs():
    """pixel_values plus pre-tokenized ids, so no tokenizer download is needed."""
    torch.manual_seed(0)
    return {
        "pixel_values": torch.randn(1, 3, 112, 112),
        "input_ids": torch.randint(0, 1000, (1, 8)),
    }


class TestSam3Config:
    def test_init_default_config(self):
        config = Sam3Config()
        assert config.image_size == 1008
        assert config.vision_patch_size == 14
        assert config.vision_global_attn_indexes == [7, 15, 23, 31]
        assert config.num_queries == 200
        assert config.hidden_size == 256
        assert config.default_prompt
        assert config.aggregate == "semantic"
        assert config.nobg_version

    def test_default_mutable_not_shared(self):
        a, b = Sam3Config(), Sam3Config()
        a.vision_global_attn_indexes.append(99)
        assert b.vision_global_attn_indexes == [7, 15, 23, 31]

    def test_bad_aggregate_raises(self):
        with pytest.raises(ValueError, match="aggregate"):
            Sam3Config(aggregate="median")

    def test_indivisible_image_size_raises(self):
        with pytest.raises(ValueError, match="divisible"):
            Sam3Config(image_size=113, vision_patch_size=14)


class TestSam3:
    def test_init_default_config(self, monkeypatch):
        """`Sam3()` falls back to Sam3Config and forwards it to transformers.

        A full-size SAM3 is ~850 M params, so the backbone is stubbed out and only
        the config plumbing is asserted.
        """
        seen = {}

        def fake_sam3_model(hf_config):
            seen["config"] = hf_config
            return torch.nn.Linear(1, 1)

        monkeypatch.setattr(modeling_sam3, "Sam3Model", fake_sam3_model)
        model = Sam3()
        assert isinstance(model.config, Sam3Config)
        assert model.config.image_size == 1008
        backbone = seen["config"].vision_config.backbone_config
        assert backbone.image_size == 1008
        assert backbone.patch_size == 14
        assert backbone.global_attn_indexes == [7, 15, 23, 31]
        assert seen["config"].detr_decoder_config.num_queries == 200
        assert seen["config"].text_config.hidden_size == 1024
        # backbone_feature_sizes derive from the ViT grid (1008/14 = 72)
        assert seen["config"].vision_config.backbone_feature_sizes == [
            [288, 288],
            [144, 144],
            [72, 72],
        ]

    def test_init_custom_config(self, small_model):
        assert small_model.config.image_size == 112
        assert small_model.config.num_queries == 10
        assert small_model.config.vision_hidden_size == 32

    def test_sub_configs_not_stored(self, small_model):
        """Only the nobg dataclass is kept; transformers configs stay local to __init__."""
        assert isinstance(small_model.config, Sam3Config)
        assert not hasattr(small_model, "vision_config")
        assert not hasattr(small_model, "text_config")

    def test_forward(self, small_model, inputs):
        with torch.no_grad():
            out = small_model(**inputs)
        assert out["logits"].shape == (1, 1, 112, 112)
        assert torch.isfinite(out["logits"]).all()
        # instance-level outputs pass through untouched
        assert out["pred_masks"].shape[:2] == (1, 10)
        assert out["pred_boxes"].shape == (1, 10, 4)
        assert out["pred_logits"].shape == (1, 10)
        assert out["presence_logits"].shape == (1, 1)
        assert out["semantic_seg"].shape[:2] == (1, 1)

    def test_forward_batch(self, small_model):
        pixel_values = torch.randn(3, 3, 112, 112)
        input_ids = torch.randint(0, 1000, (3, 8))
        with torch.no_grad():
            out = small_model(pixel_values=pixel_values, input_ids=input_ids)
        assert out["logits"].shape == (3, 1, 112, 112)

    def test_logits_match_pixel_values_size(self, small_model, inputs):
        """The matte is upsampled from the decoder grid back to the input size."""
        with torch.no_grad():
            out = small_model(**inputs)
        # decoder emits a coarse grid; logits are full input resolution
        assert out["pred_masks"].shape[-2:] != out["logits"].shape[-2:]
        assert out["logits"].shape[-2:] == inputs["pixel_values"].shape[-2:]

    def test_wrong_input_size_raises(self, small_model):
        """The ViT's rotary buffers are sized from config.image_size, so enforce it."""
        with pytest.raises(ValueError, match="built for"):
            small_model(
                pixel_values=torch.randn(1, 3, 70, 70),
                input_ids=torch.randint(0, 1000, (1, 8)),
            )

    def test_forward_without_input_ids_raises(self, small_model):
        with pytest.raises(ValueError, match="prompt-driven"):
            small_model(pixel_values=torch.randn(1, 3, 112, 112))

    def test_prompt_changes_matte(self, small_model, inputs):
        other_ids = torch.randint(0, 1000, (1, 8)) + 1
        with torch.no_grad():
            a = small_model(**inputs)["logits"]
            b = small_model(pixel_values=inputs["pixel_values"], input_ids=other_ids)[
                "logits"
            ]
        assert not torch.allclose(a, b)

    def test_threshold_changes_matte(self, instance_model, inputs):
        with torch.no_grad():
            keep_all = instance_model(**inputs, score_threshold=0.0)["logits"]
            keep_one = instance_model(**inputs, score_threshold=1.0)["logits"]
        assert not torch.allclose(keep_all, keep_one)

    def test_threshold_ignored_under_semantic_aggregate(self, small_model, inputs):
        """The semantic head has no per-query scores, so the knob is inert."""
        with torch.no_grad():
            low = small_model(**inputs, score_threshold=0.0)["logits"]
            high = small_model(**inputs, score_threshold=1.0)["logits"]
        assert torch.equal(low, high)

    def test_semantic_is_the_default_and_reads_semantic_seg(self, small_model, inputs):
        """Default matte is SAM3's semantic head, not the instance union."""
        assert small_model.config.aggregate == "semantic"
        with torch.no_grad():
            out = small_model(**inputs)
        expected = torch.nn.functional.interpolate(
            out["semantic_seg"], size=(112, 112), mode="bilinear", align_corners=False
        )
        assert torch.allclose(out["logits"], expected, atol=1e-6)

    def test_threshold_filtering_everything_still_yields_matte(
        self, instance_model, inputs
    ):
        """A threshold no query can clear falls back to the top-scoring query."""
        with torch.no_grad():
            out = instance_model(**inputs, score_threshold=1.0)
        assert out["logits"].shape == (1, 1, 112, 112)
        assert torch.isfinite(out["logits"]).all()

    def test_aggregate_mean(self, small_config, inputs):
        small_config.aggregate = "mean"
        model = Sam3(config=small_config).eval()
        with torch.no_grad():
            out = model(**inputs, score_threshold=0.0)
        assert out["logits"].shape == (1, 1, 112, 112)
        assert torch.isfinite(out["logits"]).all()

    def test_aggregate_max_equals_max_over_kept_logits(self, instance_model, inputs):
        """max over logits == logit of max probability, so no sigmoid round trip."""
        with torch.no_grad():
            out = instance_model(**inputs, score_threshold=0.0)
        expected = out["pred_masks"].max(dim=1).values.unsqueeze(1)
        expected = torch.nn.functional.interpolate(
            expected, size=(112, 112), mode="bilinear", align_corners=False
        )
        assert torch.allclose(out["logits"], expected, atol=1e-5)

    def test_forward_with_labels_returns_loss(self, small_model, inputs):
        labels = (torch.rand(1, 1, 112, 112) > 0.5).float()
        out = small_model(**inputs, labels=labels)
        assert out["loss"].ndim == 0
        assert torch.isfinite(out["loss"])

    def test_criterion_is_sam3_loss_not_birefnet(self, small_model, inputs):
        """SAM3 uses its own focal+dice objective, matching how it was trained."""
        assert small_model.criterion is sam3_loss
        labels = (torch.rand(1, 1, 112, 112) > 0.5).float()
        with torch.no_grad():
            out = small_model(**inputs, labels=labels)
            expected = sam3_loss([out["logits"]], labels)
            birefnet = birefnet_loss([out["logits"]], labels)
        assert out["loss"].item() == pytest.approx(expected.item(), abs=1e-6)
        assert not torch.isclose(out["loss"], birefnet, atol=1e-3)

    def test_criterion_hot_swap(self, small_model, inputs):
        calls = []

        def fake_loss(scaled_preds, gt):
            calls.append(len(scaled_preds))
            return torch.tensor(1.234)

        small_model.criterion = fake_loss
        labels = torch.zeros(1, 1, 112, 112)
        out = small_model(**inputs, labels=labels)
        assert out["loss"].item() == pytest.approx(1.234)
        assert calls == [1]  # single aggregated matte, no multi-scale heads
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
            saved = json.load(f)
        assert saved["image_size"] == 112
        assert saved["num_queries"] == 10
        assert saved["vision_hidden_size"] == 32
        assert saved["vision_global_attn_indexes"] == [1]
        assert saved["aggregate"] == "semantic"
        assert saved["score_threshold"] == pytest.approx(0.3)
        assert saved["default_prompt"]
        assert "nobg_version" in saved

    def test_from_pretrained_roundtrip(self, small_model, tmp_path):
        small_model.save_pretrained(tmp_path)
        loaded = Sam3.from_pretrained(tmp_path)
        assert isinstance(loaded.config, Sam3Config)
        assert loaded.config.image_size == 112
        assert loaded.config.num_queries == 10
        assert loaded.config.vision_global_attn_indexes == [1]

    def test_weights_preserved_after_roundtrip(self, small_model, tmp_path, inputs):
        with torch.no_grad():
            expected = small_model(**inputs)
        small_model.save_pretrained(tmp_path)
        loaded = Sam3.from_pretrained(tmp_path).eval()
        with torch.no_grad():
            actual = loaded(**inputs)
        assert torch.allclose(expected["logits"], actual["logits"])

    def test_from_origin_same_shape(self, small_model, tmp_path, inputs):
        small_model.save_pretrained(tmp_path)
        clone = Sam3.from_origin(tmp_path).eval()
        with torch.no_grad():
            a = small_model(**inputs)["logits"]
            b = clone(**inputs)["logits"]
        assert torch.allclose(a, b, atol=1e-5)

    def test_from_origin_from_module_instance(self, small_model, inputs):
        clone = Sam3.from_origin(small_model).eval()
        assert clone.config.num_queries == small_model.config.num_queries
        with torch.no_grad():
            a = small_model(**inputs)["logits"]
            b = clone(**inputs)["logits"]
        assert torch.allclose(a, b, atol=1e-5)

    def test_from_origin_override_keeps_matching_weights(self, small_model, tmp_path):
        """Growing num_queries keeps every shape-compatible weight."""
        small_model.save_pretrained(tmp_path)
        grown = Sam3.from_origin(tmp_path, num_queries=16)
        assert grown.config.num_queries == 16
        # the backbone is untouched by the query count, so its weights carry over
        key = "model.vision_encoder.backbone.embeddings.patch_embeddings.projection.weight"
        assert torch.equal(small_model.state_dict()[key], grown.state_dict()[key])
        # the query embeddings themselves are reshaped, so they start fresh
        assert grown.state_dict()["model.detr_decoder.query_embed.weight"].shape == (
            16,
            32,
        )
        with torch.no_grad():
            out = grown.eval()(
                pixel_values=torch.randn(1, 3, 112, 112),
                input_ids=torch.randint(0, 1000, (1, 8)),
            )
        assert out["pred_logits"].shape == (1, 16)
        assert torch.isfinite(out["logits"]).all()

    def test_from_origin_config_and_overrides_conflict(self, small_model, small_config):
        with pytest.raises(ValueError, match="not both"):
            Sam3.from_origin(small_model, config=small_config, num_queries=4)


class TestUpstreamLayout:
    """`facebook/sam3` is a video checkpoint; the detector must be lifted out of it."""

    def test_remap_strips_prefix_and_drops_tracker(self):
        got = _remap_upstream_state_dict(
            {
                "detector_model.vision_encoder.weight": torch.zeros(1),
                "detector_model.detr_decoder.query_embed.weight": torch.zeros(2),
                "tracker_model.memory_attention.weight": torch.zeros(3),
                "tracker_neck.conv.weight": torch.zeros(4),
            }
        )
        assert set(got) == {
            "model.vision_encoder.weight",
            "model.detr_decoder.query_embed.weight",
        }

    def test_from_origin_loads_upstream_layout(self, small_model, tmp_path, inputs):
        """Rewrite a nobg checkpoint into upstream layout; from_origin must still match.

        This pins the shape of a real `facebook/sam3` load: a nested `sam3_video`
        config plus `detector_model.*` weights beside tracker weights we never build.
        """
        upstream = small_model.state_dict()
        c = small_model.config
        save_file(
            {f"detector_model.{k[len('model.') :]}": v for k, v in upstream.items()}
            | {"tracker_model.memory_attention.weight": torch.zeros(3)},
            str(tmp_path / "model.safetensors"),
        )
        (tmp_path / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "sam3_video",
                    "detector_config": {
                        "model_type": "sam3",
                        "vision_config": {
                            "fpn_hidden_size": c.fpn_hidden_size,
                            "backbone_config": {
                                "hidden_size": c.vision_hidden_size,
                                "intermediate_size": c.vision_intermediate_size,
                                "num_hidden_layers": c.vision_num_hidden_layers,
                                "num_attention_heads": c.vision_num_attention_heads,
                                "image_size": c.image_size,
                                "patch_size": c.vision_patch_size,
                                "window_size": c.vision_window_size,
                                "global_attn_indexes": c.vision_global_attn_indexes,
                                "pretrain_image_size": c.vision_pretrain_image_size,
                            },
                        },
                        "text_config": {
                            "vocab_size": c.text_vocab_size,
                            "hidden_size": c.text_hidden_size,
                            "intermediate_size": c.text_intermediate_size,
                            "projection_dim": c.text_projection_dim,
                            "num_hidden_layers": c.text_num_hidden_layers,
                            "num_attention_heads": c.text_num_attention_heads,
                        },
                        "geometry_encoder_config": {
                            "num_layers": c.geometry_num_layers
                        },
                        "detr_encoder_config": {
                            "hidden_size": c.hidden_size,
                            "intermediate_size": c.intermediate_size,
                            "num_attention_heads": c.num_attention_heads,
                            "num_layers": c.detr_encoder_num_layers,
                        },
                        "detr_decoder_config": {
                            "num_queries": c.num_queries,
                            "num_layers": c.detr_decoder_num_layers,
                        },
                        "mask_decoder_config": {
                            "num_upsampling_stages": c.num_upsampling_stages
                        },
                    },
                }
            )
        )

        loaded = Sam3.from_origin(tmp_path).eval()
        assert loaded.config.image_size == 112
        assert loaded.config.num_queries == 10
        # every weight transferred, so outputs are bit-identical to the source
        for key, value in upstream.items():
            assert torch.equal(loaded.state_dict()[key], value), key
        with torch.no_grad():
            assert torch.allclose(
                small_model(**inputs)["logits"], loaded(**inputs)["logits"]
            )


class TestFieldsFromHFConfig:
    def test_reads_nested_sam3_video_config(self):
        """`facebook/sam3` ships a sam3_video composite; read the nested detector."""
        raw = {
            "model_type": "sam3_video",
            "detector_config": {
                "model_type": "sam3",
                "vision_config": {
                    "fpn_hidden_size": 256,
                    "backbone_config": {
                        "hidden_size": 1024,
                        "image_size": 1008,
                        "patch_size": 14,
                        "global_attn_indexes": [7, 15, 23, 31],
                    },
                },
                "text_config": {"hidden_size": 1024, "vocab_size": 49408},
                "detr_encoder_config": {"hidden_size": 256, "num_layers": 6},
                "detr_decoder_config": {"num_queries": 200, "num_layers": 6},
                "mask_decoder_config": {"num_upsampling_stages": 3},
            },
        }
        got = _fields_from_hf_config(raw)
        assert got["image_size"] == 1008
        assert got["vision_hidden_size"] == 1024
        assert got["vision_global_attn_indexes"] == [7, 15, 23, 31]
        assert got["fpn_hidden_size"] == 256
        assert got["num_queries"] == 200
        assert got["detr_encoder_num_layers"] == 6
        assert got["text_vocab_size"] == 49408
        assert got["num_upsampling_stages"] == 3
        # only known fields, all constructible
        assert Sam3Config(**got).image_size == 1008

    def test_reads_flat_sam3_config(self):
        raw = {
            "model_type": "sam3",
            "vision_config": {"backbone_config": {"image_size": 504, "patch_size": 14}},
        }
        got = _fields_from_hf_config(raw)
        assert got["image_size"] == 504
        assert got["vision_patch_size"] == 14

    def test_missing_keys_fall_back_to_defaults(self):
        got = _fields_from_hf_config({"model_type": "sam3"})
        assert got == {}
        assert Sam3Config(**got).image_size == 1008


class TestAutoModel:
    def test_automodel_dispatches_sam3(self, monkeypatch, small_model, tmp_path):
        small_model.save_pretrained(tmp_path)
        monkeypatch.setattr(
            auto, "model_info", lambda *a, **k: SimpleNamespace(tags=["nobg-sam3"])
        )
        loaded = AutoModel.from_pretrained(tmp_path)
        assert isinstance(loaded, Sam3)
        assert loaded.config.num_queries == 10

    def test_automodel_dispatches_bare_sam3_tag(
        self, monkeypatch, small_model, tmp_path
    ):
        small_model.save_pretrained(tmp_path)
        monkeypatch.setattr(
            auto, "model_info", lambda *a, **k: SimpleNamespace(tags=["sam3"])
        )
        assert isinstance(AutoModel.from_pretrained(tmp_path), Sam3)


class TestSam3Loss:
    """SAM 3's semantic-seg objective: focal + dice, written from the published
    formulations (Meta's own code is SAM-licensed, nobg is Apache-2.0)."""

    @pytest.fixture
    def gt(self):
        g = torch.zeros(2, 1, 32, 32)
        g[:, :, 8:24, 8:24] = 1.0
        return g

    def test_zero_at_perfect_prediction(self, gt):
        confident = (gt * 2 - 1) * 20  # large-magnitude logits of the right sign
        assert sam3_loss([confident], gt).item() == pytest.approx(0.0, abs=1e-4)

    def test_inverted_is_worse_than_random(self, gt):
        inverted = (1 - gt) * 2 - 1
        torch.manual_seed(0)
        assert sam3_loss([inverted], gt) > sam3_loss([torch.randn_like(gt)], gt)

    def test_gradients_flow(self, gt):
        torch.manual_seed(0)
        pred = torch.randn_like(gt).requires_grad_(True)
        sam3_loss([pred], gt).backward()
        assert torch.isfinite(pred.grad).all()
        assert pred.grad.abs().sum() > 0

    def test_resizes_prediction_to_label(self, gt):
        """SAM3 predicts at 288^2 for a 1008^2 input, so this path is the norm."""
        torch.manual_seed(0)
        loss = sam3_loss([torch.randn(2, 1, 8, 8)], gt)
        assert loss.ndim == 0 and torch.isfinite(loss)

    def test_weights_are_the_published_ratio(self, gt):
        """20 x focal + 30 x dice, per Meta's SemanticSegCriterion weight_dict."""
        torch.manual_seed(0)
        pred = torch.randn_like(gt)
        expected = 20.0 * sigmoid_focal_loss(pred, gt, alpha=0.6, gamma=2.0)
        expected = expected + 30.0 * dice_loss(pred, gt)
        assert sam3_loss([pred], gt).item() == pytest.approx(expected.item(), abs=1e-6)

    def test_focal_matches_reference_formulation(self, gt):
        """Pins the focal formula against the arXiv:1708.02002 definition."""
        torch.manual_seed(0)
        pred = torch.randn_like(gt)
        alpha, gamma = 0.6, 2.0
        prob = pred.sigmoid()
        ce = torch.nn.functional.binary_cross_entropy_with_logits(
            pred, gt, reduction="none"
        )
        p_t = prob * gt + (1 - prob) * (1 - gt)
        ref = ce * (1 - p_t).pow(gamma) * (alpha * gt + (1 - alpha) * (1 - gt))
        got = sigmoid_focal_loss(pred, gt, alpha=alpha, gamma=gamma)
        assert got.item() == pytest.approx(ref.mean().item(), abs=1e-9)

    def test_alpha_negative_disables_class_weighting(self, gt):
        torch.manual_seed(0)
        pred = torch.randn_like(gt)
        weighted = sigmoid_focal_loss(pred, gt, alpha=0.5)
        unweighted = sigmoid_focal_loss(pred, gt, alpha=-1.0)
        assert unweighted.item() == pytest.approx(2 * weighted.item(), abs=1e-6)

    def test_dice_is_scale_free(self):
        """Dice normalizes by mask area; a tiny foreground is not down-weighted."""
        losses = []
        for frac in (0.02, 0.5):
            g = torch.zeros(2, 1, 32, 32)
            g.view(2, -1)[:, : int(32 * 32 * frac)] = 1.0
            losses.append(dice_loss(torch.zeros_like(g), g).item())
        # An all-negative prediction misses everything either way, so both are
        # high -- but the sparse case is *worse*, the opposite of raw BCE.
        assert losses[0] > losses[1]

    def test_dice_zero_on_exact_overlap(self, gt):
        assert dice_loss((gt * 2 - 1) * 20, gt).item() == pytest.approx(0.0, abs=1e-3)

    def test_comparable_magnitude_to_birefnet_loss(self, gt):
        """Same lr works for both; a 10x scale gap would need retuning."""
        torch.manual_seed(0)
        pred = torch.randn_like(gt)
        ratio = sam3_loss([pred], gt).item() / birefnet_loss([pred], gt).item()
        assert 0.5 < ratio < 2.0
