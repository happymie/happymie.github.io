"""Check actual upstream parity and model-specific training semantics."""
import hashlib
import importlib
import json
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from biomed_lab.data import demo_batch
from biomed_lab.models.foundation_models import SPECS, SAMLesson, build_model

torch.set_num_threads(2)


@pytest.mark.parametrize("num", [30, 31, 32, 33, 95])
def test_original_architecture_has_finite_backward_and_optimizer_step(num):
    torch.manual_seed(13)
    model = build_model(num, {}).train()
    batch = demo_batch(kind="image_text" if num in (30, 31, 95) else "prompt_shapes")
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=1e-3)
    loss = model.training_loss(batch)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    gradients = [p.grad for p in parameters if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    before = parameters[0].detach().clone()
    optimizer.step()
    assert not torch.equal(before, parameters[0])


def test_clip_reads_eos_embedding_and_cannot_read_future_padding_tokens():
    torch.manual_seed(2)
    model = build_model(30, {}).eval()
    tokens = demo_batch(kind="image_text")["tokens"]
    mapped = model.prepare_tokens(tokens)
    eos_positions = (tokens == 2).long().argmax(-1)
    assert torch.equal(mapped.argmax(-1), eos_positions)
    assert not torch.equal(tokens.argmax(-1), eos_positions)
    with torch.no_grad():
        net = model.network
        x = net.token_embedding(mapped) + net.positional_embedding
        x = net.transformer(x.permute(1, 0, 2)).permute(1, 0, 2)
        x = net.ln_final(x)
        explicit_eos_readout = x[torch.arange(len(tokens)), eos_positions] @ net.text_projection
        torch.testing.assert_close(net.encode_text(mapped), explicit_eos_readout)
        changed = mapped.clone()
        changed[:, 5:] = 7
        torch.testing.assert_close(net.encode_text(changed), net.encode_text(mapped))


def test_clip_preserves_original_forward_and_symmetric_pair_objective():
    torch.manual_seed(10)
    model = build_model(30, {}).eval()
    batch = demo_batch(kind="image_text")
    with torch.no_grad():
        wrapped = model(batch["image"], batch["tokens"])
        original = model.network(model.prepare_image(batch["image"]), model.prepare_tokens(batch["tokens"]))
        for actual, expected in zip(wrapped, original):
            torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(wrapped[1], wrapped[0].T)
        target = torch.arange(batch["image"].shape[0])
        expected_loss = (F.cross_entropy(original[0], target) + F.cross_entropy(original[1], target)) / 2
        torch.testing.assert_close(model.training_loss(batch), expected_loss)
    model.training_loss(batch).backward()
    assert model.network.visual.conv1.weight.grad.abs().sum() > 0
    assert model.network.token_embedding.weight.grad.abs().sum() > 0


def test_clip_rejects_missing_end_token_and_reserved_high_id():
    model = build_model(30, {})
    with pytest.raises(ValueError, match="EOS"):
        model.prepare_tokens(torch.zeros(2, 16, dtype=torch.long))
    bad = demo_batch(kind="image_text")["tokens"].clone()
    bad[:, 5] = 31
    with pytest.raises(ValueError, match="Reserve"):
        model.prepare_tokens(bad)


def test_clip_short_context_discards_only_empty_padding():
    model = build_model(30, {"context_length": 4})
    tokens = demo_batch(kind="image_text")["tokens"]
    assert model.prepare_tokens(tokens).shape == (2, 4)
    tokens[:, 8] = 7
    with pytest.raises(ValueError, match="trailing padding"):
        model.prepare_tokens(tokens)


def test_sam_matches_unmodified_author_inference_on_each_prompted_image():
    torch.manual_seed(44)
    model = build_model(32, {}).eval()
    batch = demo_batch(kind="prompt_shapes")
    author_inputs = [dict(
        image=image.repeat(3, 1, 1) * 255,
        original_size=(32, 32),
        point_coords=point[None, None] * 31,
        point_labels=torch.ones(1, 1, dtype=torch.long),
    ) for image, point in zip(batch["image"], batch["prompt_point"])]
    with torch.no_grad():
        actual = model(batch["image"], points=batch["prompt_point"])
        expected = model.network(author_inputs, multimask_output=True)
    assert actual["mask_logits"].shape == (2, 3, 32, 32)
    for i, record in enumerate(expected):
        torch.testing.assert_close(actual["low_res_logits"][i:i+1], record["low_res_logits"])
        torch.testing.assert_close(actual["iou_predictions"][i:i+1], record["iou_predictions"])
        assert torch.equal(actual["mask_logits"][i:i+1] > 0, record["masks"])


@pytest.mark.parametrize("num", [32, 33])
def test_paired_batch_does_not_expand_to_b_squared_or_mix_images(num):
    torch.manual_seed(34)
    model = build_model(num, {}).eval()
    batch = demo_batch(batch_size=3, kind="prompt_shapes")
    with torch.no_grad():
        together = model(batch["image"], boxes=batch["box"], multimask_output=False)
        apart = [model(batch["image"][i:i+1], boxes=batch["box"][i:i+1], multimask_output=False)
                 for i in range(3)]
    for key in together:
        assert together[key].shape[0] == 3
        torch.testing.assert_close(together[key], torch.cat([x[key] for x in apart]), atol=1e-6, rtol=1e-4)


def test_sam_prediction_responds_to_point_location_and_dense_mask_prompt():
    torch.manual_seed(8)
    model = build_model(32, {}).eval()
    batch = demo_batch(kind="prompt_shapes")
    with torch.no_grad():
        embedding = model.encode_image(batch["image"])
        a = model.decode(embedding, points=batch["prompt_point"])
        b = model.decode(embedding, points=1 - batch["prompt_point"])
        c = model.decode(embedding, points=batch["prompt_point"], mask_inputs=a["low_res_logits"][:, :1])
    assert not torch.allclose(a["mask_logits"], b["mask_logits"])
    assert not torch.allclose(a["mask_logits"], c["mask_logits"])


def test_sam_interactive_training_reaches_dense_prompt_and_single_mask_heads():
    torch.manual_seed(12)
    model = build_model(32, {"interactive_rounds": 2})
    model.training_loss(demo_batch(kind="prompt_shapes")).backward()
    net = model.network
    assert net.image_encoder.patch_embed.proj.weight.grad.abs().sum() > 0
    assert net.prompt_encoder.mask_downscaling[0].weight.grad.abs().sum() > 0
    assert net.mask_decoder.output_hypernetworks_mlps[0].layers[-1].weight.grad.abs().sum() > 0
    assert net.mask_decoder.iou_prediction_head.layers[-1].weight.grad.abs().sum() > 0


def test_sam_ambiguity_loss_backpropagates_only_best_mask_candidate():
    target = torch.zeros(2, 8, 8)
    target[:, 2:6, 2:6] = 1
    logits = torch.stack((target * 10 - 5, torch.full_like(target, -2), torch.full_like(target, 2)), dim=1).requires_grad_()
    quality = torch.zeros(2, 3, requires_grad=True)
    SAMLesson.sam_loss({"mask_logits": logits, "iou_predictions": quality}, target).backward()
    assert logits.grad[:, 0].abs().sum() > 0
    assert torch.equal(logits.grad[:, 1:], torch.zeros_like(logits.grad[:, 1:]))
    assert quality.grad.abs().sum() > 0


def test_correction_clicks_are_errors_with_correct_foreground_labels():
    target = torch.zeros(2, 8, 8)
    target[:, 2:5, 3:6] = 1
    prediction = torch.full((2, 1, 8, 8), -10.)
    point, label = SAMLesson.correction_point(prediction, target)
    coordinates = (point[:, 0] * 7).round().long()
    assert torch.equal(label, torch.ones_like(label))
    assert all(target[i, xy[1], xy[0]] == 1 for i, xy in enumerate(coordinates))
    correct = (target[:, None] * 2 - 1) * 10
    _, padded = SAMLesson.correction_point(correct, target)
    assert torch.equal(padded, torch.full_like(padded, -1))


def test_medsam_freezes_prompt_encoder_and_keeps_original_input_convention():
    torch.manual_seed(15)
    model = build_model(33, {})
    batch = demo_batch(kind="prompt_shapes")
    torch.testing.assert_close(model.prepare_image(batch["image"]), batch["image"].repeat(1, 3, 1, 1))
    model.training_loss(batch).backward()
    assert all(not p.requires_grad and p.grad is None for p in model.network.prompt_encoder.parameters())
    assert model.network.image_encoder.patch_embed.proj.weight.grad.abs().sum() > 0
    assert model.network.mask_decoder.mask_tokens.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.network.mask_decoder.iou_prediction_head.parameters())
    with pytest.raises(ValueError, match="box-only"):
        model(batch["image"], points=batch["prompt_point"])


@pytest.mark.parametrize("num,overrides", [(30, {"vision_width": 32}), (32, {"prompt_dim": 30}), (33, {"image_size": 31})])
def test_invalid_reduced_shapes_fail_before_silent_architecture_changes(num, overrides):
    with pytest.raises(ValueError):
        build_model(num, overrides)


def test_source_manifests_retain_original_files_and_licenses():
    vendor = Path(__file__).resolve().parents[1] / "vendor"
    for package in ("foundation_clip", "foundation_sam", "foundation_medsam"):
        path = vendor / package
        provenance = json.loads((path / "PROVENANCE.json").read_text())
        assert len(provenance["commit"]) == 40
        assert (path / "LICENSE").stat().st_size > 1000
        for record in provenance["files"]:
            assert hashlib.sha256((path / record["path"]).read_bytes()).hexdigest() == record["sha256"]
            assert provenance["commit"] in record["url"]
    assert set(SPECS) == {30, 31, 32, 33, 95}
    for unavailable in (34, 39, 40, 98, 112):
        with pytest.raises(NotImplementedError, match="source route"):
            build_model(unavailable, {})


def test_align_uses_mbconv_se_bert_and_actual_smoothed_symmetric_loss():
    torch.manual_seed(39)
    model = build_model(95, {}).eval()
    batch = demo_batch(kind="image_text")
    # Use distinct descriptions to exercise nonzero contrastive gradients.
    batch["tokens"][0, 2], batch["tokens"][1, 2] = 4, 5
    output = model(batch["image"], batch["tokens"], return_loss=True)
    names = {type(m).__name__ for m in model.network.modules()}
    assert {"AlignVisionBlock", "AlignVisionSqueezeExciteLayer", "AlignTextSelfAttention"} <= names
    torch.testing.assert_close(output.logits_per_text, output.logits_per_image.T)
    target = torch.arange(2)
    expected = (F.cross_entropy(output.logits_per_image, target, label_smoothing=.1)
                + F.cross_entropy(output.logits_per_text, target, label_smoothing=.1)) / 2
    torch.testing.assert_close(output.loss, expected)
    output.loss.backward()
    assert model.network.text_projection.weight.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.network.vision_model.parameters())


def test_blip2_trains_queries_and_crossattention_while_both_towers_stay_frozen():
    torch.manual_seed(22)
    model = build_model(31, {}).train()
    batch = demo_batch(kind="image_text")
    before = {k: v.clone() for k, v in model.network.vision_model.state_dict().items()}
    loss = model.training_loss(batch)
    loss.backward()
    net = model.network
    assert not net.vision_model.training and not net.language_model.training
    assert net.qformer.training
    for module in (net.vision_model, net.language_model):
        assert all(not p.requires_grad and p.grad is None for p in module.parameters())
    assert net.query_tokens.grad.abs().sum() > 0
    assert net.language_projection.weight.grad.abs().sum() > 0
    assert net.qformer.encoder.layer[0].crossattention.attention.key.weight.grad.abs().sum() > 0
    for key, value in net.vision_model.state_dict().items():
        assert torch.equal(value, before[key])


def test_blip2_caption_loss_ignores_padding_and_visual_prefix_and_is_causal():
    torch.manual_seed(28)
    model = build_model(31, {}).eval()
    batch = demo_batch(kind="image_text")
    inputs, attention, labels = model.prepare_tokens(batch["tokens"])
    assert (labels[:, :5] == -100).all()
    assert (labels[~attention] == -100).all()
    with torch.no_grad():
        out = model(batch["image"], batch["tokens"])
        expected = F.cross_entropy(out.logits[:, :-1].reshape(-1, 32), labels[:, 1:].reshape(-1))
        torch.testing.assert_close(out.loss, expected)
        changed = batch["tokens"].clone()
        changed[:, 2] = 7
        later = model(batch["image"], changed)
        # Position 5 is BOS+one position after the four visual tokens; changing
        # position 6 cannot affect this earlier causal output.
        torch.testing.assert_close(out.logits[:, :6], later.logits[:, :6])
        changed_image = model(1 - batch["image"], batch["tokens"])
        assert not torch.allclose(out.logits, changed_image.logits)


def test_canonical_foundation_library_is_pinned_and_source_hashes_match():
    import transformers
    assert transformers.__version__ == "4.46.3"
    path = Path(__file__).resolve().parents[1] / "vendor" / "foundation_transformers"
    provenance = json.loads((path / "PROVENANCE.json").read_text())
    assert (path / "LICENSE").stat().st_size > 1000
    assert provenance["commit"] == "052e652d6d53c2b26ffde87e039b723949a53493"
    for record in provenance["installed_files"]:
        module = importlib.import_module(record["module"])
        assert hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() == record["sha256"]
