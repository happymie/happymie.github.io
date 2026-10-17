"""Source-first, reduced foundation architectures for synthetic CPU lessons.

All architecture files under vendor/foundation_* are unmodified author releases.
This module supplies reduced constructor arguments, explicit synthetic I/O, and
training objectives. Random initialization is deliberate: these are architecture
and objective exercises, not replicas of pretrained foundation capabilities.
"""
from __future__ import annotations

from contextlib import nullcontext
from functools import partial
import importlib
import math

import torch
from torch import nn
from torch.nn import functional as F


_CLIP_SOURCE = "https://github.com/openai/CLIP/tree/d05afc436d78f1c48dc0dbf8e5980a9d471f35f6"
_SAM_SOURCE = "https://github.com/facebookresearch/segment-anything/tree/dca509fe793f601edb92606367a655c15ac00fdf"
_MEDSAM_SOURCE = "https://github.com/bowang-lab/MedSAM/tree/d71e8a1a99ad751840a22a7fa3ecfb4166fb1488"
_TRANSFORMERS_SOURCE = "https://github.com/huggingface/transformers/tree/052e652d6d53c2b26ffde87e039b723949a53493/src/transformers/models"


def _spec(name, task, source, mechanism, limitations, **config):
    return dict(name=name, task=task, implementation="upstream", source_url=source,
                mechanism=mechanism, limitations=limitations, config=config)


SPECS = {
    30: _spec(
        "CLIP", "image_text", _CLIP_SOURCE,
        "Original OpenAI ViT image encoder, causal text Transformer, EOT pooling, learned logit scale and symmetric image–text contrastive loss.",
        ["Unmodified MIT author architecture with smaller dimensions and random initialization; no pretrained weights or foundation-pretraining claim.",
         "Synthetic grayscale images are repeated into RGB. A 32-token teaching vocabulary replaces byte-pair tokenization; EOS=2 is explicitly remapped to the largest token ID required by original CLIP pooling.",
         "Paired diagonal contrastive targets are retained. Repeated short synthetic descriptions can be false negatives; tiny batches do not establish semantic retrieval or zero-shot quality."],
        image_size=32, patch_size=8, vision_width=64, vision_layers=2,
        text_width=64, text_heads=4, text_layers=2, embed_dim=32,
        context_length=16, vocab_size=32),
    31: _spec(
        "BLIP-2 (stage-two bridge architecture)", "image_text", _TRANSFORMERS_SOURCE + "/blip_2",
        "Pinned Transformers BLIP-2 ViT, learned query tokens, Q-Former cross-attention, projection and frozen OPT causal language model; train only the stage-two visual-language bridge.",
        ["Canonical Apache-2.0 library implementation, not copied Salesforce LAVIS training code. All modules start randomly initialized at reduced dimensions; frozen random ViT/OPT have no pretrained visual or language knowledge.",
         "Only the stage-two bridge architecture and causal caption loss are exercised. Stage-one image-text contrastive, image-text matching and image-grounded generation pretraining, instruction tuning and real caption data are absent.",
         "Synthetic vocabulary replaces the original tokenizer. Vision initialization and learned query tokens use normal std=0.02 to avoid the library checkpoint-oriented near-zero/zero defaults. No image-captioning or question-answering capability is claimed."],
        image_size=32, patch_size=8, width=32, depth=2, heads=4,
        query_tokens=4, context_length=16, vocab_size=32),
    32: _spec(
        "Segment Anything (SAM)", "prompt_segmentation", _SAM_SOURCE,
        "Original window/global ViT, Fourier point/box prompt encoder, two-way attention decoder, three ambiguity masks, IoU head, and error-guided correction with previous-mask prompts.",
        ["Unmodified Apache-2.0 author architecture at reduced image size/width/depth with random initialization; no MAE initialization, SA-1B pretraining, or zero-shot performance claim.",
         "Teaching training uses an initial supplied foreground point and two prompt rounds by default. The 11-round mixed point/box schedule, noisy boxes, large-scale data engine and original optimizer schedule are not reproduced.",
         "Mask supervision uses minimum-over-candidate 20× focal + Dice; IoU regression supervises every returned candidate. Default supplied synthetic prompts come from ground truth and are explicit inputs, not predicted detections."],
        image_size=32, patch_size=8, width=64, depth=2, heads=4,
        prompt_dim=32, decoder_depth=2, decoder_heads=4, decoder_mlp_dim=64,
        window_size=2, interactive_rounds=2),
    33: _spec(
        "MedSAM", "prompt_segmentation", _MEDSAM_SOURCE,
        "Original MedSAM image/prompt/mask modules, box-only prompts, frozen prompt encoder, and author squared-prediction Dice plus BCE training objective.",
        ["Unmodified Apache-2.0 MedSAM architecture at reduced dimensions. Random initialization replaces SAM/MedSAM checkpoints; this is the fine-tuning mechanism on synthetic data, not a pretrained medical model.",
         "The original MedSAM [0,1] input convention is retained. Synthetic grayscale is repeated into RGB; supplied tight boxes replace the original randomly perturbed training boxes and medical preprocessing.",
         "The prompt encoder stays frozen as in the author training script. IoU scores are returned for inspection but are not trained by MedSAM's Dice+BCE objective. No clinical data or performance claim."],
        image_size=32, patch_size=8, width=64, depth=2, heads=4,
        prompt_dim=32, decoder_depth=2, decoder_heads=4, decoder_mlp_dim=64,
        window_size=2),
    95: _spec(
        "ALIGN", "image_text", _TRANSFORMERS_SOURCE + "/align",
        "Pinned Transformers ALIGN: seven-stage EfficientNet MBConv/SE image tower, BERT text tower, normalized embeddings, learned temperature and symmetric label-smoothed contrastive loss.",
        ["Canonical Apache-2.0 library implementation follows the Kakao Brain reproduction; it is not Google's unreleased original training implementation or checkpoint.",
         "Reduced width/depth with random initialization, 32-token synthetic vocabulary and two paired images replace web-scale noisy image-text training. All seven EfficientNet stage types remain; each has at least one block.",
         "Vision initializer std=0.1 replaces the library's checkpoint-oriented 0.02, whose deep uncalibrated random tower can underflow during evaluation. Synthetic images use [-1,1] normalization without original data augmentation; no foundation retrieval or zero-shot performance is established."],
        image_size=32, width_coefficient=0.25, depth_coefficient=0.25,
        text_width=32, text_depth=2, text_heads=4, context_length=16, vocab_size=32),
}


def _integer(config, key, low, high):
    value = config[key]
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ValueError(f"{key} must be an integer in [{low}, {high}]")
    return value


def _rgb(image, image_size):
    if image.ndim != 4 or image.shape[1] not in (1, 3):
        raise ValueError("Expected B×1×H×W or B×3×H×W images")
    if image.shape[-2:] != (image_size, image_size):
        raise ValueError(f"Expected square {image_size}×{image_size} input; resize images and prompts together")
    return image.repeat(1, 3, 1, 1) if image.shape[1] == 1 else image


def _transformers():
    package = importlib.import_module("transformers")
    if package.__version__ != "4.46.3":
        raise RuntimeError("These source-pinned lessons require transformers==4.46.3")
    return package


def _teaching_tokens(tokens, context_length, vocab_size):
    if tokens.ndim != 2 or tokens.dtype != torch.long:
        raise ValueError("Teaching text must be a B×L int64 token tensor")
    if tokens.shape[1] > context_length:
        if (tokens[:, context_length:] != 0).any():
            raise ValueError("Only trailing padding may be removed from teaching text")
        tokens = tokens[:, :context_length]
    if (tokens < 0).any() or (tokens >= vocab_size - 1).any():
        raise ValueError("Reserve the largest vocabulary ID for visual placeholders")
    if not (tokens[:, 0] == 1).all() or not ((tokens == 2).sum(-1) == 1).all():
        raise ValueError("Teaching text needs BOS=1 first and exactly one EOS=2")
    return tokens


class ALIGNLesson(nn.Module):
    """Actual EfficientNet/BERT ALIGN class from the pinned canonical library."""

    def __init__(self, config):
        super().__init__()
        tr = _transformers()
        from transformers.models.align.modeling_align import round_filters

        c = config
        self.image_size = _integer(c, "image_size", 32, 256)
        self.context_length = _integer(c, "context_length", 4, 128)
        self.vocab_size = _integer(c, "vocab_size", 8, 4096)
        width = _integer(c, "text_width", 16, 256)
        heads = _integer(c, "text_heads", 1, 16)
        if width % heads:
            raise ValueError("text_width must divide by text_heads")
        for key in ("width_coefficient", "depth_coefficient"):
            value = c[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not .125 <= value <= 1:
                raise ValueError(f"{key} must be finite and in [0.125, 1]")
        vision = tr.AlignVisionConfig(image_size=self.image_size,
            width_coefficient=c["width_coefficient"], depth_coefficient=c["depth_coefficient"],
            drop_connect_rate=0, initializer_range=.1)
        # ALIGN has no vision projection: text projection must equal the actual
        # last EfficientNet channel count after the author's filter rounding.
        projection = round_filters(vision, vision.out_channels[-1])
        vision.hidden_dim = projection
        text = tr.AlignTextConfig(vocab_size=self.vocab_size, hidden_size=width,
            num_hidden_layers=_integer(c, "text_depth", 1, 8), num_attention_heads=heads,
            intermediate_size=width * 2, max_position_embeddings=self.context_length,
            hidden_dropout_prob=0, attention_probs_dropout_prob=0, pad_token_id=0)
        self.network = tr.AlignModel(tr.AlignConfig.from_text_vision_configs(
            text, vision, projection_dim=projection))

    def forward(self, image, tokens, return_loss=False):
        tokens = _teaching_tokens(tokens, self.context_length, self.vocab_size)
        return self.network(pixel_values=_rgb(image, self.image_size) * 2 - 1,
            input_ids=tokens, attention_mask=tokens.ne(0), return_loss=return_loss)

    def training_loss(self, batch):
        if batch["image"].shape[0] < 2:
            raise ValueError("ALIGN contrastive loss needs at least two paired examples")
        return self(batch["image"], batch["tokens"], return_loss=True).loss


class BLIP2Lesson(nn.Module):
    """Real Q-Former bridge; randomly initialized frozen towers are explicit."""

    def __init__(self, config):
        super().__init__()
        tr = _transformers()
        c = config
        self.image_size = _integer(c, "image_size", 16, 256)
        patch = _integer(c, "patch_size", 2, 32)
        width = _integer(c, "width", 16, 256)
        heads = _integer(c, "heads", 1, 16)
        depth = _integer(c, "depth", 1, 8)
        self.context_length = _integer(c, "context_length", 4, 128)
        self.vocab_size = _integer(c, "vocab_size", 8, 4096)
        self.num_query_tokens = _integer(c, "query_tokens", 2, 32)
        if self.image_size % patch or width % heads:
            raise ValueError("image_size must divide by patch_size, and width by heads")
        vision = tr.Blip2VisionConfig(hidden_size=width, intermediate_size=width * 2,
            num_hidden_layers=depth, num_attention_heads=heads, image_size=self.image_size,
            patch_size=patch, attention_dropout=0, initializer_range=.02)
        qformer = tr.Blip2QFormerConfig(hidden_size=width, intermediate_size=width * 2,
            num_hidden_layers=depth, num_attention_heads=heads, hidden_dropout_prob=0,
            attention_probs_dropout_prob=0, max_position_embeddings=self.context_length)
        text = tr.OPTConfig(hidden_size=width, word_embed_proj_dim=width,
            ffn_dim=width * 2, num_hidden_layers=depth, num_attention_heads=heads,
            vocab_size=self.vocab_size, max_position_embeddings=self.context_length + self.num_query_tokens,
            bos_token_id=1, eos_token_id=2, pad_token_id=0, dropout=0, attention_dropout=0,
            use_cache=False)
        combined = tr.Blip2Config.from_vision_qformer_text_configs(vision, qformer, text,
            num_query_tokens=self.num_query_tokens, image_token_index=self.vocab_size - 1)
        self.network = tr.Blip2ForConditionalGeneration(combined)
        # Library defaults expect pretrained towers and leave query tokens zero.
        # A documented normal initialization gives the tiny random-init exercise
        # distinguishable queries without modifying any upstream architecture.
        nn.init.normal_(self.network.query_tokens, std=.02)
        self.network.vision_model.requires_grad_(False)
        self.network.language_model.requires_grad_(False)
        self.register_buffer("pixel_mean", torch.tensor([.48145466, .4578275, .40821073])[None, :, None, None])
        self.register_buffer("pixel_std", torch.tensor([.26862954, .26130258, .27577711])[None, :, None, None])
        self.train()

    def train(self, mode=True):
        super().train(mode)
        if hasattr(self, "network"):
            self.network.vision_model.eval()
            self.network.language_model.eval()
        return self

    def prepare_tokens(self, tokens):
        tokens = _teaching_tokens(tokens, self.context_length, self.vocab_size)
        prefix = tokens.new_full((tokens.shape[0], self.num_query_tokens), self.vocab_size - 1)
        inputs = torch.cat((prefix, tokens), dim=1)
        labels = inputs.clone()
        labels[labels == 0] = -100
        # Visual placeholders and BOS are conditioning, not caption targets.
        labels[:, :self.num_query_tokens + 1] = -100
        return inputs, inputs.ne(0), labels

    def forward(self, image, tokens):
        inputs, attention, labels = self.prepare_tokens(tokens)
        image = (_rgb(image, self.image_size) - self.pixel_mean) / self.pixel_std
        return self.network(pixel_values=image, input_ids=inputs, attention_mask=attention, labels=labels)

    def training_loss(self, batch):
        return self(batch["image"], batch["tokens"]).loss


class CLIPLesson(nn.Module):
    """The released CLIP class, with a documented synthetic tokenizer boundary."""

    def __init__(self, config):
        super().__init__()
        from vendor.foundation_clip.model import CLIP

        c = config
        self.image_size = _integer(c, "image_size", 16, 256)
        patch = _integer(c, "patch_size", 2, 32)
        vision_width = _integer(c, "vision_width", 64, 512)
        text_width = _integer(c, "text_width", 16, 512)
        text_heads = _integer(c, "text_heads", 1, 16)
        self.context_length = _integer(c, "context_length", 4, 128)
        self.vocab_size = _integer(c, "vocab_size", 8, 4096)
        if self.image_size % patch or vision_width % 64 or text_width % text_heads:
            raise ValueError("image_size must divide by patch_size; vision_width by 64; text_width by text_heads")
        self.network = CLIP(
            embed_dim=_integer(c, "embed_dim", 8, 512),
            image_resolution=self.image_size,
            vision_layers=_integer(c, "vision_layers", 1, 8),
            vision_width=vision_width, vision_patch_size=patch,
            context_length=self.context_length, vocab_size=self.vocab_size,
            transformer_width=text_width, transformer_heads=text_heads,
            transformer_layers=_integer(c, "text_layers", 1, 8),
        )
        # These are the actual values in OpenAI's pinned clip/clip.py.
        self.register_buffer("pixel_mean", torch.tensor([.48145466, .4578275, .40821073])[None, :, None, None])
        self.register_buffer("pixel_std", torch.tensor([.26862954, .26130258, .27577711])[None, :, None, None])

    def prepare_tokens(self, tokens):
        """Map teaching EOS=2 to CLIP's largest ID, preserving EOT readout.

        This accepts the documented synthetic vocabulary, not free text or real
        OpenAI BPE IDs. Silently using the class token as EOT would change CLIP.
        """
        if tokens.ndim != 2 or tokens.dtype != torch.long:
            raise ValueError("Teaching text must be a B×L int64 token tensor")
        if tokens.shape[1] > self.context_length:
            if (tokens[:, self.context_length:] != 0).any():
                raise ValueError("Input text exceeds context_length; only trailing padding may be removed")
            tokens = tokens[:, :self.context_length]
        if (tokens < 0).any() or (tokens >= self.vocab_size - 1).any():
            raise ValueError("Reserve the largest vocabulary ID for remapped EOS")
        if not ((tokens == 2).sum(-1) == 1).all():
            raise ValueError("Each teaching sequence must contain exactly one EOS=2")
        result = F.pad(tokens, (0, self.context_length - tokens.shape[1])).clone()
        result[result == 2] = self.vocab_size - 1
        return result

    def prepare_image(self, image):
        return (_rgb(image, self.image_size) - self.pixel_mean) / self.pixel_std

    def forward(self, image, tokens):
        return self.network(self.prepare_image(image), self.prepare_tokens(tokens))

    def training_loss(self, batch):
        if batch["image"].shape[0] < 2:
            raise ValueError("CLIP contrastive loss needs at least two paired examples")
        image_logits, text_logits = self(batch["image"], batch["tokens"])
        target = torch.arange(image_logits.shape[0], device=image_logits.device)
        return (F.cross_entropy(image_logits, target) + F.cross_entropy(text_logits, target)) / 2

    @torch.no_grad()
    def after_optimizer_step(self):
        # CLIP paper §2.5 clips the exponentiated temperature to at most 100.
        self.network.logit_scale.clamp_(max=4.605170185988092)


def _sam_components(config, medical):
    upstream = importlib.import_module("vendor.foundation_medsam" if medical else "vendor.foundation_sam")
    c = config
    size = _integer(c, "image_size", 16, 256)
    patch = _integer(c, "patch_size", 2, 32)
    width = _integer(c, "width", 16, 256)
    depth = _integer(c, "depth", 1, 8)
    heads = _integer(c, "heads", 1, 16)
    prompt_dim = _integer(c, "prompt_dim", 16, 256)
    decoder_heads = _integer(c, "decoder_heads", 1, 16)
    window = _integer(c, "window_size", 1, 16)
    if size % patch or width % heads or prompt_dim % 8 or (prompt_dim // 2) % decoder_heads:
        raise ValueError("SAM needs image_size divisible by patch_size, width by heads, prompt_dim by 8, and prompt_dim/2 by decoder_heads")
    grid = size // patch
    return upstream.Sam(
        image_encoder=upstream.ImageEncoderViT(
            img_size=size, patch_size=patch, in_chans=3, embed_dim=width,
            depth=depth, num_heads=heads, mlp_ratio=4, out_chans=prompt_dim,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), qkv_bias=True,
            use_rel_pos=True, use_abs_pos=True, window_size=window,
            global_attn_indexes=(depth - 1,),
        ),
        prompt_encoder=upstream.PromptEncoder(
            embed_dim=prompt_dim, image_embedding_size=(grid, grid),
            input_image_size=(size, size), mask_in_chans=8,
        ),
        mask_decoder=upstream.MaskDecoder(
            transformer_dim=prompt_dim, num_multimask_outputs=3,
            transformer=upstream.TwoWayTransformer(
                depth=_integer(c, "decoder_depth", 1, 4),
                embedding_dim=prompt_dim, num_heads=decoder_heads,
                mlp_dim=_integer(c, "decoder_mlp_dim", 16, 1024),
            ),
            iou_head_depth=3, iou_head_hidden_dim=prompt_dim,
        ),
    )


def _check_prompt(tensor, batch_size, tail, name):
    if tensor.shape[0] != batch_size or tensor.shape[-1] != tail:
        raise ValueError(f"{name} has an incompatible batch or coordinate dimension")
    if not torch.isfinite(tensor).all() or (tensor < 0).any() or (tensor > 1).any():
        raise ValueError(f"{name} coordinates must be finite and normalized to [0,1]")


class SAMLesson(nn.Module):
    """Actual SAM components, including interactive mask and point prompting."""

    def __init__(self, config, medical=False):
        super().__init__()
        self.image_size = config["image_size"]
        self.medical = medical
        self.interactive_rounds = 1 if medical else _integer(config, "interactive_rounds", 1, 11)
        self.network = _sam_components(config, medical)
        if medical:
            self.network.prompt_encoder.requires_grad_(False)

    def prepare_image(self, image):
        rgb = _rgb(image, self.image_size)
        # MedSAM's author training script expects preprocessed [0,1] inputs and
        # calls the image encoder directly. SAM uses its original normalization.
        return rgb if self.medical else self.network.preprocess(rgb * 255)

    def encode_image(self, image):
        return self.network.image_encoder(self.prepare_image(image))

    def decode(self, embedding, *, points=None, point_labels=None, boxes=None,
               mask_inputs=None, multimask_output=True):
        n = embedding.shape[0]
        if points is None and boxes is None:
            raise ValueError("Supply an explicit point or box prompt")
        if self.medical and (boxes is None or points is not None):
            raise ValueError("The MedSAM lesson uses the author's box-only prompting")
        if points is not None:
            if points.ndim == 2:
                points = points[:, None, :]
            _check_prompt(points, n, 2, "point")
            if point_labels is None:
                point_labels = torch.ones(points.shape[:2], device=points.device, dtype=torch.long)
            if point_labels.shape != points.shape[:2] or not torch.isin(point_labels, point_labels.new_tensor([-1, 0, 1])).all():
                raise ValueError("Point labels must match B×N points and contain -1, 0, or 1")
            point_arg = (points * (self.image_size - 1), point_labels)
        else:
            point_arg = None
        if boxes is not None:
            if boxes.ndim != 2:
                raise ValueError("Box prompts must have shape B×4")
            _check_prompt(boxes, n, 4, "box")
            if (boxes[:, 2:] < boxes[:, :2]).any():
                raise ValueError("Box maximum coordinates must follow minimum coordinates")
            boxes = boxes * (self.image_size - 1)
        prompt_encoder = self.network.prompt_encoder
        with torch.no_grad() if self.medical else nullcontext():
            sparse, dense = prompt_encoder(points=point_arg, boxes=boxes, masks=mask_inputs)
        image_pe = prompt_encoder.get_dense_pe()
        if self.medical:
            # MedSAM's released decoder has an explicit paired-batch branch.
            low, quality = self.network.mask_decoder(
                image_embeddings=embedding, image_pe=image_pe,
                sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
                multimask_output=False,
            )
        else:
            # Original SAM expands ONE image for many prompt sets internally.
            # Passing B images and B prompts at once would create a B² error.
            masks, scores = [], []
            for i in range(n):
                low_i, score_i = self.network.mask_decoder(
                    image_embeddings=embedding[i:i+1], image_pe=image_pe,
                    sparse_prompt_embeddings=sparse[i:i+1],
                    dense_prompt_embeddings=dense[i:i+1],
                    multimask_output=multimask_output,
                )
                masks.append(low_i)
                scores.append(score_i)
            low, quality = torch.cat(masks), torch.cat(scores)
        logits = F.interpolate(low, (self.image_size, self.image_size), mode="bilinear", align_corners=False)
        return {"mask_logits": logits, "low_res_logits": low, "iou_predictions": quality}

    def forward(self, image, *, points=None, point_labels=None, boxes=None,
                mask_inputs=None, multimask_output=True):
        return self.decode(self.encode_image(image), points=points,
                           point_labels=point_labels, boxes=boxes,
                           mask_inputs=mask_inputs, multimask_output=multimask_output)

    @staticmethod
    def sam_loss(output, target):
        """Appendix-A focal:Dice=20:1 and candidate mask-quality regression."""
        logits = output["mask_logits"]
        truth = target[:, None].float().expand_as(logits)
        prob = logits.sigmoid()
        bce = F.binary_cross_entropy_with_logits(logits, truth, reduction="none")
        pt = prob * truth + (1 - prob) * (1 - truth)
        alpha = .25 * truth + .75 * (1 - truth)
        focal = (alpha * (1 - pt).square() * bce).mean((-2, -1))
        intersection = (prob * truth).sum((-2, -1))
        dice = 1 - (2 * intersection + 1) / (prob.sum((-2, -1)) + truth.sum((-2, -1)) + 1)
        candidate_loss = 20 * focal + dice
        selected_loss = candidate_loss.min(dim=1).values.mean()
        with torch.no_grad():
            binary = logits > 0
            gt = truth.bool()
            numerator = (binary & gt).sum((-2, -1)).float()
            denominator = (binary | gt).sum((-2, -1)).float()
            true_iou = torch.where(denominator > 0, numerator / denominator.clamp_min(1), torch.ones_like(denominator))
        return selected_loss + F.mse_loss(output["iou_predictions"], true_iou)

    @staticmethod
    @torch.no_grad()
    def correction_point(mask_logits, target):
        """Uniformly sample an error pixel; type is false-negative/positive."""
        error = (mask_logits[:, 0] > 0) != target.bool()
        points, labels = [], []
        for sample_error, gt in zip(error, target):
            ij = sample_error.nonzero()
            if ij.shape[0] == 0:
                points.append(mask_logits.new_zeros(2))
                labels.append(-1)  # no-error padding, not a background click
                continue
            chosen = ij[torch.randint(len(ij), (), device=ij.device)]
            row, col = chosen.unbind()
            points.append(torch.stack((col / max(gt.shape[1] - 1, 1), row / max(gt.shape[0] - 1, 1))))
            labels.append(int(gt[row, col] > 0))
        return torch.stack(points)[:, None], torch.tensor(labels, device=target.device)[:, None]

    def training_loss(self, batch):
        target = batch["mask"]
        if self.medical:
            # Exactly the loss settings in MedSAM/train_one_gpu.py; MONAI is
            # imported only when this model is run, keeping catalog reads cheap.
            from monai.losses import DiceLoss
            output = self(batch["image"], boxes=batch["box"], multimask_output=False)
            logits = output["mask_logits"]
            truth = target[:, None].float()
            dice = DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
            return dice(logits, truth) + F.binary_cross_entropy_with_logits(logits, truth)
        embedding = self.encode_image(batch["image"])
        points = batch["prompt_point"][:, None, :]
        labels = torch.ones(points.shape[:2], dtype=torch.long, device=points.device)
        previous = None
        losses = []
        for round_index in range(self.interactive_rounds):
            output = self.decode(embedding, points=points, point_labels=labels,
                                 mask_inputs=previous, multimask_output=(round_index == 0))
            losses.append(self.sam_loss(output, target))
            if round_index + 1 == self.interactive_rounds:
                break
            with torch.no_grad():
                choice = output["iou_predictions"].argmax(dim=1)
                rows = torch.arange(embedding.shape[0], device=embedding.device)
                previous = output["low_res_logits"][rows, choice][:, None].detach()
                mask = output["mask_logits"][rows, choice][:, None].detach()
                next_points, next_labels = self.correction_point(mask, target)
                points = torch.cat((points, next_points), dim=1)
                labels = torch.cat((labels, next_labels), dim=1)
        return torch.stack(losses).mean()


def build_model(num, config):
    if num not in SPECS:
        raise NotImplementedError(f"No faithful foundation CPU adapter for model {num}; use its source route")
    c = dict(SPECS[num]["config"])
    extra = set(config) - set(c)
    if extra:
        raise ValueError(f"Unsupported foundation settings: {sorted(extra)}")
    c.update(config)
    if num == 30:
        return CLIPLesson(c)
    if num == 31:
        return BLIP2Lesson(c)
    if num == 95:
        return ALIGNLesson(c)
    return SAMLesson(c, medical=(num == 33))
