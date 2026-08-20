import json
import os

import numpy as np
import pytest
import torch
from PIL import Image
from transformers import CLIPTokenizer
from transformers.models.sam3.image_processing_sam3 import Sam3ImageProcessor

from nobg import AutoProcessor, Sam3, Sam3Processor
from nobg.sam3.image_processing_sam3 import DEFAULT_PROMPT
from nobg.sam3.modeling_sam3 import DEFAULT_TOKENIZER, Sam3Config


def _small_config(vocab_size: int, **overrides) -> Sam3Config:
    """A ~0.27 M-param SAM3, small enough to run in a test."""
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
        text_vocab_size=vocab_size,
        text_hidden_size=32,
        text_intermediate_size=64,
        text_projection_dim=32,
        text_num_hidden_layers=2,
        text_num_attention_heads=2,
        **overrides,
    )


@pytest.fixture
def tokenizer(tmp_path_factory):
    """A minimal CLIP tokenizer built on disk, so nothing is downloaded."""
    d = tmp_path_factory.mktemp("tok")
    vocab = {"<|startoftext|>": 0, "<|endoftext|>": 1}
    for i, char in enumerate("abcdefghijklmnopqrstuvwxyz"):
        vocab[char] = 2 + i
        vocab[char + "</w>"] = 28 + i
    (d / "vocab.json").write_text(json.dumps(vocab))
    (d / "merges.txt").write_text("#version: 0.2\n")
    return CLIPTokenizer(str(d / "vocab.json"), str(d / "merges.txt"))


@pytest.fixture
def processor(tokenizer):
    return Sam3Processor(
        Sam3ImageProcessor(
            size={"height": 112, "width": 112},
            mask_size={"height": 32, "width": 32},
        ),
        tokenizer,
    )


@pytest.fixture
def rgb_image():
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 255, (48, 64, 3), dtype=np.uint8))


class TestSam3Processor:
    def test_default_attributes(self, processor):
        assert processor.default_prompt == DEFAULT_PROMPT
        assert processor.target_size == 112
        assert processor.point_pad_value == -10
        assert processor.image_processor.size["height"] == 112

    def test_custom_default_prompt(self, tokenizer):
        proc = Sam3Processor(
            Sam3ImageProcessor(size={"height": 112, "width": 112}),
            tokenizer,
            default_prompt="a dog",
        )
        assert proc.default_prompt == "a dog"

    def test_preprocess_without_text_uses_default_prompt(self, processor, rgb_image):
        """This is what makes SAM3 usable prompt-free, like BiRefNetImageProcessor."""
        out = processor(images=rgb_image, return_tensors="pt")
        assert out["pixel_values"].shape == (1, 3, 112, 112)
        assert out["input_ids"].shape == (1, 32)  # upstream pads to max_length=32
        assert out["attention_mask"].shape == (1, 32)
        expected = processor.tokenizer(
            DEFAULT_PROMPT, padding="max_length", max_length=32, return_tensors="pt"
        )
        assert torch.equal(out["input_ids"], expected["input_ids"])

    def test_explicit_text_overrides_default(self, processor, rgb_image):
        default = processor(images=rgb_image, return_tensors="pt")
        explicit = processor(images=rgb_image, text="a dog", return_tensors="pt")
        assert not torch.equal(default["input_ids"], explicit["input_ids"])

    def test_preprocess_batch(self, processor):
        imgs = [Image.new("RGB", (64, 48)), Image.new("RGB", (30, 90))]
        out = processor(images=imgs, return_tensors="pt")
        assert out["pixel_values"].shape == (2, 3, 112, 112)

    def test_original_sizes_recorded(self, processor, rgb_image):
        out = processor(images=rgb_image, return_tensors="pt")
        assert out["original_sizes"].tolist() == [[48, 64]]

    def test_save_pretrained(self, processor, tmp_path):
        processor.save_pretrained(tmp_path)
        files = os.listdir(tmp_path)
        assert "processor_config.json" in files
        with open(tmp_path / "processor_config.json") as f:
            saved = json.load(f)
        assert saved["processor_class"] == "Sam3Processor"
        assert saved["default_prompt"] == DEFAULT_PROMPT
        assert saved["target_size"] == 112
        assert saved["image_processor"]["size"] == {"height": 112, "width": 112}

    def test_from_pretrained_roundtrip(self, tokenizer, tmp_path):
        proc = Sam3Processor(
            Sam3ImageProcessor(size={"height": 56, "width": 56}),
            tokenizer,
            default_prompt="a cat",
        )
        proc.save_pretrained(tmp_path)
        loaded = Sam3Processor.from_pretrained(tmp_path)
        assert isinstance(loaded, Sam3Processor)
        assert loaded.default_prompt == "a cat"
        assert loaded.image_processor.size["height"] == 56
        out = loaded(images=Image.new("RGB", (100, 80)), return_tensors="pt")
        assert out["pixel_values"].shape == (1, 3, 56, 56)

    def test_auto_processor_local_dir_roundtrip(self, processor, tmp_path):
        # AutoProcessor dispatches on the recorded image_processor_type without Hub access.
        processor.save_pretrained(tmp_path)
        loaded = AutoProcessor.from_pretrained(tmp_path)
        assert isinstance(loaded, Sam3Processor)
        assert loaded.image_processor.size["height"] == 112

    def test_auto_processor_accepts_upstream_fast_type(self, processor, tmp_path):
        """A raw `facebook/sam3` repo records Sam3ImageProcessorFast; still ours."""
        processor.save_pretrained(tmp_path)
        path = tmp_path / "processor_config.json"
        saved = json.loads(path.read_text())
        saved["image_processor"]["image_processor_type"] = "Sam3ImageProcessorFast"
        path.write_text(json.dumps(saved))
        loaded = AutoProcessor.from_pretrained(tmp_path)
        assert isinstance(loaded, Sam3Processor)

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
        mattes = processor.post_process_alpha_matting(
            {"logits": torch.randn(1, 1, 16, 16)}
        )
        assert mattes[0].shape == (16, 16)

    def test_post_process_target_sizes_length_mismatch_raises(self, processor):
        with pytest.raises(ValueError):
            processor.post_process_alpha_matting(
                {"logits": torch.randn(2, 1, 16, 16)}, target_sizes=[(10, 10)]
            )

    def test_cutout(self, processor):
        image = Image.new("RGB", (40, 30), (10, 20, 30))
        alpha = torch.rand(30, 40)
        cut = processor.cutout(image, alpha)
        assert cut.mode == "RGBA"
        assert cut.size == (40, 30)
        got = torch.from_numpy(np.array(cut.getchannel("A")))
        expected = (alpha.clamp(0, 1) * 255).to(torch.uint8)
        assert torch.equal(got.to(torch.uint8), expected)

    def test_refine_foreground(self, processor):
        image = torch.rand(3, 30, 40)
        refined = processor.refine_foreground(image, torch.rand(30, 40), r=5)
        assert refined.shape == image.shape
        assert torch.isfinite(refined).all()

    def test_cutout_refine(self, processor):
        image = Image.new("RGB", (40, 30), (10, 20, 30))
        alpha = torch.rand(30, 40)
        cut = processor.cutout(image, alpha, refine=True)
        assert cut.mode == "RGBA"
        assert cut.size == (40, 30)
        # Refinement rewrites RGB but must leave the matte alone.
        got = torch.from_numpy(np.array(cut.getchannel("A")))
        expected = (alpha.clamp(0, 1) * 255).to(torch.uint8)
        assert torch.equal(got.to(torch.uint8), expected)

    def test_instance_post_processing_still_available(self, processor):
        """Wrapping must not cost the inherited open-vocabulary behaviour."""
        assert hasattr(processor.image_processor, "post_process_instance_segmentation")
        assert hasattr(processor.image_processor, "post_process_semantic_segmentation")

    def test_end_to_end_with_small_model(self, processor):
        model = Sam3(config=_small_config(processor.tokenizer.vocab_size)).eval()
        image = Image.new("RGB", (200, 150), (120, 130, 140))

        inputs = processor(images=image, return_tensors="pt")
        # `model(**inputs)` is the documented flow, so the processor's extra
        # `original_sizes` key must pass through harmlessly.
        assert "original_sizes" in inputs
        with torch.no_grad():
            out = model(**inputs)
        assert out["logits"].shape == (1, 1, 112, 112)

        matte = processor.post_process_alpha_matting(
            out, target_sizes=[(image.height, image.width)]
        )[0]
        assert matte.shape == (150, 200)
        cut = processor.cutout(image, matte)
        assert cut.mode == "RGBA"
        assert cut.size == (200, 150)


class TestSam3Predict:
    @pytest.fixture
    def model(self, processor):
        # Seeded: on a 0.27 M-param random model the prompt shifts the matte by
        # only ~1e-4, and on an unlucky init it drops below the atol that
        # test_prompt_changes_the_matte asserts. Same reason test_sam3.py seeds.
        torch.manual_seed(0)
        return Sam3(config=_small_config(processor.tokenizer.vocab_size))

    def test_single_image_returns_cutout(self, model, processor):
        image = Image.new("RGB", (60, 40), (10, 20, 30))
        cut = model.predict(processor, image)
        assert isinstance(cut, Image.Image)
        assert cut.mode == "RGBA"
        assert cut.size == (60, 40)

    def test_list_returns_list_at_original_sizes(self, model, processor):
        images = [Image.new("RGB", (60, 40)), Image.new("RGB", (33, 77))]
        cuts = model.predict(processor, images)
        assert [c.size for c in cuts] == [(60, 40), (33, 77)]

    def test_return_type_alpha(self, model, processor):
        alpha = model.predict(
            processor, Image.new("RGB", (60, 40)), return_type="alpha"
        )
        assert alpha.shape == (40, 60)
        assert alpha.min() >= 0 and alpha.max() <= 1

    def test_prompt_changes_the_matte(self, model, processor):
        image = Image.new("RGB", (60, 40), (90, 100, 110))
        default = model.predict(processor, image, return_type="alpha")
        prompted = model.predict(processor, image, prompt="a dog", return_type="alpha")
        assert not torch.allclose(default, prompted, atol=1e-6)

    def test_prompt_can_be_positional(self, model, processor):
        image = Image.new("RGB", (60, 40), (90, 100, 110))
        keyword = model.predict(processor, image, prompt="a dog", return_type="alpha")
        positional = model.predict(processor, image, "a dog", return_type="alpha")
        assert torch.equal(keyword, positional)

    def test_score_threshold_is_plumbed_through(self, model, processor):
        # Only the instance-union path scores queries; the default "semantic"
        # aggregate ignores the threshold entirely.
        model.config.aggregate = "max"
        image = Image.new("RGB", (60, 40), (90, 100, 110))
        low = model.predict(processor, image, score_threshold=0.0, return_type="alpha")
        high = model.predict(processor, image, score_threshold=1.0, return_type="alpha")
        assert not torch.allclose(low, high, atol=1e-6)

    def test_shared_prompt_is_broadcast_over_a_batch(self, model, processor):
        # One prompt for N images gives input_ids of batch 1 against pixel_values
        # of batch N; without broadcasting this fails inside attention.
        images = [Image.new("RGB", (60, 40)), Image.new("RGB", (33, 77))]
        cuts = model.predict(processor, images, "a dog", batch_size=2)
        assert [c.size for c in cuts] == [(60, 40), (33, 77)]

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

    def test_matches_the_manual_pipeline(self, model, processor):
        model.eval()
        image = Image.new("RGB", (60, 40), (90, 100, 110))
        inputs = processor(images=[image], return_tensors="pt")
        with torch.no_grad():
            out = model(
                pixel_values=inputs["pixel_values"],
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
        expected = processor.post_process_alpha_matting(
            out, target_sizes=[(image.height, image.width)]
        )[0]
        got = model.predict(processor, image, return_type="alpha")
        assert torch.equal(expected, got)

    def test_restores_training_mode(self, model, processor):
        model.train()
        model.predict(processor, Image.new("RGB", (60, 40)))
        assert model.training

    def test_rejects_a_preprocessed_batch(self, model, processor):
        inputs = processor(images=Image.new("RGB", (60, 40)), return_tensors="pt")
        with pytest.raises(TypeError, match="raw images"):
            model.predict(processor, inputs)


class TestSam3PredictBoxes:
    """`boxes` is the fourth positional argument: predict(processor, image, prompt, boxes)."""

    @pytest.fixture
    def model(self, processor):
        torch.manual_seed(0)
        return Sam3(config=_small_config(processor.tokenizer.vocab_size))

    def test_boxes_change_the_matte(self, model, processor):
        image = Image.new("RGB", (60, 40), (90, 100, 110))
        without = model.predict(processor, image, return_type="alpha")
        with_boxes = model.predict(
            processor, image, None, [[5, 5, 30, 30]], return_type="alpha"
        )
        assert not torch.allclose(without, with_boxes, atol=1e-6)

    def test_boxes_can_be_a_keyword(self, model, processor):
        image = Image.new("RGB", (60, 40), (90, 100, 110))
        positional = model.predict(
            processor, image, None, [[5, 5, 30, 30]], return_type="alpha"
        )
        keyword = model.predict(
            processor, image, boxes=[[5, 5, 30, 30]], return_type="alpha"
        )
        assert torch.equal(positional, keyword)

    def test_boxes_combine_with_a_prompt(self, model, processor):
        image = Image.new("RGB", (60, 40), (90, 100, 110))
        cut = model.predict(processor, image, "a dog", [[5, 5, 30, 30]])
        assert cut.size == (60, 40)

    def test_several_boxes_for_one_image(self, model, processor):
        image = Image.new("RGB", (60, 40), (90, 100, 110))
        one = model.predict(
            processor, image, None, [[5, 5, 30, 30]], return_type="alpha"
        )
        two = model.predict(
            processor,
            image,
            None,
            [[5, 5, 30, 30], [31, 10, 55, 35]],
            return_type="alpha",
        )
        assert not torch.allclose(one, two, atol=1e-6)

    def test_one_box_list_per_image(self, model, processor):
        images = [Image.new("RGB", (60, 40)), Image.new("RGB", (33, 77))]
        boxes = [[[5, 5, 30, 30]], [[2, 2, 20, 40]]]
        cuts = model.predict(processor, images, None, boxes)
        assert [c.size for c in cuts] == [(60, 40), (33, 77)]

    def test_boxes_are_sliced_per_chunk(self, model, processor):
        # The batch_size seam: boxes are per-image, so chunking must slice them
        # alongside the images rather than resending the whole list.
        images = [Image.new("RGB", (60, 40)), Image.new("RGB", (33, 77))]
        boxes = [[[5, 5, 30, 30]], [[2, 2, 20, 40]]]
        one = model.predict(processor, images, None, boxes, return_type="alpha")
        two = model.predict(
            processor, images, None, boxes, return_type="alpha", batch_size=2
        )
        for a, b in zip(one, two, strict=True):
            assert torch.allclose(a, b, atol=1e-5)

    def test_box_count_mismatch_raises(self, model, processor):
        images = [Image.new("RGB", (60, 40)), Image.new("RGB", (33, 77))]
        with pytest.raises(ValueError, match="boxes for 1 images"):
            model.predict(processor, images, None, [[[5, 5, 30, 30]]])

    def test_flat_box_raises(self, model, processor):
        # A single un-nested box is the natural mistake; it must not be read as
        # a list of four one-element boxes.
        with pytest.raises(ValueError, match="levels of nesting"):
            model.predict(processor, Image.new("RGB", (60, 40)), None, [5, 5, 30, 30])

    def test_empty_boxes_raises(self, model, processor):
        with pytest.raises(TypeError, match="non-empty"):
            model.predict(processor, Image.new("RGB", (60, 40)), None, [])

    def test_matches_the_manual_pipeline_with_boxes(self, model, processor):
        model.eval()
        image = Image.new("RGB", (60, 40), (90, 100, 110))
        inputs = processor(
            images=[image], input_boxes=[[[5, 5, 30, 30]]], return_tensors="pt"
        )
        with torch.no_grad():
            out = model(
                pixel_values=inputs["pixel_values"],
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                input_boxes=inputs["input_boxes"],
                input_boxes_labels=inputs["input_boxes_labels"],
            )
        expected = processor.post_process_alpha_matting(
            out, target_sizes=[(image.height, image.width)]
        )[0]
        got = model.predict(
            processor, image, None, [[5, 5, 30, 30]], return_type="alpha"
        )
        assert torch.equal(expected, got)


class TestSam3Process:
    """`process` is `predict` with the processor built from the model's own config.

    SAM3 also needs a tokenizer, which no config records, so `AutoTokenizer` is
    served from the on-disk fixture instead of the Hub — that keeps these offline
    while still pinning the resolution order.
    """

    @pytest.fixture
    def hub(self, monkeypatch, tokenizer):
        import transformers

        class Hub:
            def __init__(self):
                self.requested = []
                self.missing = set()

            def from_pretrained(self, source, *args, **kwargs):
                self.requested.append(str(source))
                if str(source) in self.missing:
                    raise OSError(f"{source} is not a local folder and is not gated")
                return tokenizer

        hub = Hub()
        monkeypatch.setattr(
            transformers.AutoTokenizer, "from_pretrained", hub.from_pretrained
        )
        return hub

    @pytest.fixture
    def model(self, tokenizer):
        torch.manual_seed(0)
        return Sam3(config=_small_config(tokenizer.vocab_size))

    def test_default_processor_follows_the_config(self, model, hub):
        proc = model.default_processor()
        assert isinstance(proc, Sam3Processor)
        assert proc.image_processor.size["height"] == 112
        assert proc.default_prompt == model.config.default_prompt

    def test_custom_default_prompt_is_carried_over(self, tokenizer, hub):
        model = Sam3(config=_small_config(tokenizer.vocab_size, default_prompt="a dog"))
        assert model.default_processor().default_prompt == "a dog"

    def test_fresh_model_falls_back_to_the_clip_tokenizer(self, model, hub):
        model.default_processor()
        assert model._tokenizer_source is None
        assert hub.requested == [DEFAULT_TOKENIZER]

    def test_a_tokenizer_instance_is_used_as_is(self, model, hub, tokenizer):
        proc = model.default_processor(tokenizer)
        assert proc.tokenizer is tokenizer
        assert hub.requested == []  # nothing resolved, nothing downloaded

    def test_a_repo_id_is_loaded(self, model, hub):
        model.default_processor("some/clip")
        assert hub.requested == ["some/clip"]

    def test_the_recorded_origin_wins_over_the_fallback(self, model, hub):
        model._tokenizer_source = "some/sam3"
        model.default_processor()
        assert hub.requested == ["some/sam3"]

    def test_falls_back_when_the_origin_has_no_tokenizer(self, model, hub):
        # A nobg checkpoint directory holds weights and a config, but no tokenizer.
        model._tokenizer_source = "some/sam3"
        hub.missing.add("some/sam3")
        model.default_processor()
        assert hub.requested == ["some/sam3", DEFAULT_TOKENIZER]

    def test_raises_when_nothing_yields_a_tokenizer(self, model, hub):
        model._tokenizer_source = "some/sam3"
        hub.missing.update({"some/sam3", DEFAULT_TOKENIZER})
        with pytest.raises(OSError, match="could not load a tokenizer"):
            model.default_processor()

    def test_the_resolved_processor_is_cached(self, model, hub):
        assert model.default_processor() is model.default_processor()
        assert hub.requested == [DEFAULT_TOKENIZER]  # loaded once, not twice

    def test_an_explicit_tokenizer_is_not_cached(self, model, hub, tokenizer):
        # Explicit beats cached, and never becomes the cache: two different
        # tokenizers in a row must give two different processors.
        assert model.default_processor(tokenizer) is not model.default_processor(
            tokenizer
        )
        assert model.default_processor() is not model.default_processor(tokenizer)

    def test_a_too_large_vocabulary_is_rejected(self, tokenizer, hub):
        # The text embedding has one row per `text_vocab_size`; a bigger vocabulary
        # would index off the end inside the text tower.
        model = Sam3(config=_small_config(tokenizer.vocab_size - 1))
        with pytest.raises(ValueError, match="larger than"):
            model.default_processor()

    def test_from_origin_records_a_local_origin(self, model, tmp_path, hub):
        model.save_pretrained(tmp_path)
        loaded = Sam3.from_origin(tmp_path)
        assert loaded._tokenizer_source == str(tmp_path)
        loaded.default_processor()
        assert hub.requested == [str(tmp_path)]

    def test_from_origin_carries_the_source_across_a_module(self, model):
        model._tokenizer_source = "some/sam3"
        assert Sam3.from_origin(model)._tokenizer_source == "some/sam3"

    def test_single_image_returns_cutout(self, model, hub):
        cut = model.process(Image.new("RGB", (60, 40), (10, 20, 30)))
        assert isinstance(cut, Image.Image)
        assert cut.mode == "RGBA"
        assert cut.size == (60, 40)

    def test_accepts_a_path(self, model, hub, tmp_path):
        path = tmp_path / "in.png"
        Image.new("RGB", (60, 40), (1, 2, 3)).save(path)
        assert model.process(str(path)).size == (60, 40)

    def test_list_returns_list_at_original_sizes(self, model, hub):
        images = [Image.new("RGB", (60, 40)), Image.new("RGB", (33, 77))]
        cuts = model.process(images, batch_size=2)
        assert [c.size for c in cuts] == [(60, 40), (33, 77)]

    def test_matches_predict_with_the_same_processor(self, model, hub, tokenizer):
        image = Image.new("RGB", (60, 40), (90, 100, 110))
        processor = Sam3Processor(
            Sam3ImageProcessor(size={"height": 112, "width": 112}), tokenizer
        )
        expected = model.predict(processor, image, return_type="alpha")
        assert torch.equal(expected, model.process(image, return_type="alpha"))

    def test_prompt_is_the_second_positional_argument(self, model, hub):
        image = Image.new("RGB", (60, 40), (90, 100, 110))
        default = model.process(image, return_type="alpha")
        prompted = model.process(image, "a dog", return_type="alpha")
        assert not torch.allclose(default, prompted, atol=1e-6)

    def test_boxes_are_the_third_positional_argument(self, model, hub):
        image = Image.new("RGB", (60, 40), (90, 100, 110))
        without = model.process(image, return_type="alpha")
        boxed = model.process(image, None, [[5, 5, 30, 30]], return_type="alpha")
        assert not torch.allclose(without, boxed, atol=1e-6)

    def test_score_threshold_is_plumbed_through(self, model, hub):
        model.config.aggregate = "max"
        image = Image.new("RGB", (60, 40), (90, 100, 110))
        low = model.process(image, score_threshold=0.0, return_type="alpha")
        high = model.process(image, score_threshold=1.0, return_type="alpha")
        assert not torch.allclose(low, high, atol=1e-6)

    def test_rejects_bad_arguments(self, model, hub):
        with pytest.raises(ValueError, match="return_type"):
            model.process(Image.new("RGB", (60, 40)), return_type="rgb")
