from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .augmentations import NATIVE_SPECTRAL_DIM, haar_high_frequency_perturbation


def linear_fp32(layer: nn.Linear, inputs: torch.Tensor) -> torch.Tensor:
    """Evaluate the final scoring layer in FP32 even inside bf16/fp16 autocast."""
    device_type = inputs.device.type
    with torch.autocast(device_type=device_type, enabled=False):
        bias = None if layer.bias is None else layer.bias.float()
        return F.linear(inputs.float(), layer.weight.float(), bias)


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad = False
        self.lora_a = nn.Linear(base.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False)
        self.scale = alpha / rank
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.base(inputs) + self.lora_b(self.lora_a(inputs)) * self.scale


def inject_lora_qkv(model: nn.Module, last_blocks: int, rank: int, alpha: float) -> int:
    blocks = getattr(model, "blocks", None)
    if blocks is None or last_blocks <= 0 or rank <= 0:
        return 0
    replaced = 0
    for block in blocks[-last_blocks:]:
        attention = getattr(block, "attn", None)
        qkv = getattr(attention, "qkv", None)
        if isinstance(qkv, nn.Linear):
            attention.qkv = LoRALinear(qkv, rank, alpha)
            replaced += 1
    return replaced


class TinyBackbone(nn.Module):
    num_features = 64

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, self.num_features, 3, stride=2, padding=1),
            nn.GELU(),
        )

    def forward_features(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.network(inputs)
        return features.flatten(2).transpose(1, 2)

    def forward_head(self, features: torch.Tensor, pre_logits: bool = True) -> torch.Tensor:
        return features.mean(dim=1)


def create_backbone(name: str, pretrained: bool, image_size: int) -> nn.Module:
    if name == "tiny":
        return TinyBackbone()
    import timm

    return timm.create_model(name, pretrained=pretrained, num_classes=0, img_size=image_size)


def freeze(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = False


def pooled_features(model: nn.Module, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    features = model.forward_features(inputs)
    if isinstance(features, dict):
        pooled = features.get("x_norm_clstoken")
        tokens = features.get("x_norm_patchtokens")
        if pooled is None or tokens is None:
            raise RuntimeError(f"Unsupported feature dictionary: {sorted(features)}")
        return pooled, tokens
    if features.ndim == 4:
        tokens = features.flatten(2).transpose(1, 2)
    elif features.ndim == 3:
        tokens = features
    else:
        pooled = model.forward_head(features, pre_logits=True)
        return pooled, pooled[:, None, :]
    pooled = model.forward_head(features, pre_logits=True)
    if pooled.ndim > 2:
        pooled = pooled.flatten(1)
    return pooled, tokens


class AttentionPool(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.LayerNorm(dimension), nn.Linear(dimension, dimension // 4), nn.GELU(), nn.Linear(dimension // 4, 1)
        )

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weights = torch.softmax(self.score(tokens).squeeze(-1), dim=1)
        pooled = torch.sum(tokens * weights.unsqueeze(-1), dim=1)
        return pooled, weights


class TraceDetector(nn.Module):
    def __init__(self, config: dict, image_size: int, semantic_image_size: int) -> None:
        super().__init__()
        settings = config["model"]
        self.use_semantic = bool(settings.get("use_semantic_branch", True))
        self.use_wavelet = bool(settings.get("use_wavelet_branch", True))
        self.use_three_expert = bool(settings.get("use_three_expert_moe", False))
        if self.use_three_expert and not self.use_semantic:
            raise ValueError("three-expert MoE requires the semantic branch")
        self.forensic = create_backbone(
            settings["forensic_backbone"], settings.get("pretrained", True), image_size
        )
        if settings.get("freeze_forensic", True):
            freeze(self.forensic)
        replaced = inject_lora_qkv(
            self.forensic,
            settings.get("lora_last_blocks", 0),
            settings.get("lora_rank", 0),
            settings.get("lora_alpha", 1),
        )
        self.lora_modules = replaced
        forensic_dim = int(getattr(self.forensic, "num_features"))
        self.register_buffer(
            "forensic_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "forensic_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )
        semantic_dim = 0
        if self.use_semantic:
            self.semantic = create_backbone(
                settings["semantic_backbone"], settings.get("pretrained", True), semantic_image_size
            )
            if settings.get("freeze_semantic", True):
                freeze(self.semantic)
            semantic_dim = int(getattr(self.semantic, "num_features"))
            self.register_buffer("semantic_mean", torch.full((1, 3, 1, 1), 0.5))
            self.register_buffer("semantic_std", torch.full((1, 3, 1, 1), 0.5))
        self.tile_pool = AttentionPool(forensic_dim)
        self.tile_head = nn.Linear(forensic_dim, 1)
        self.consistency_head = nn.Linear(forensic_dim, 1)
        fusion_dim = forensic_dim * 2 + semantic_dim + (1 if self.use_wavelet else 0)
        hidden = settings["hidden_dim"]
        dropout = settings.get("dropout", 0.2)
        self.fusion = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden, 1)
        self.degradation_classifier = nn.Linear(hidden, settings.get("degradation_classes", 8))
        self.degradation_regressor = nn.Linear(hidden, 1)
        if self.use_three_expert:
            expert_hidden = int(settings.get("expert_hidden_dim", 256))
            gate_hidden = int(settings.get("gate_hidden_dim", 128))
            spatial_dim = forensic_dim * 2 + (1 if self.use_wavelet else 0)
            self.spatial_expert_features = nn.Sequential(
                nn.LayerNorm(spatial_dim), nn.Linear(spatial_dim, expert_hidden), nn.GELU(), nn.Dropout(dropout)
            )
            self.semantic_expert_features = nn.Sequential(
                nn.LayerNorm(semantic_dim), nn.Linear(semantic_dim, expert_hidden), nn.GELU(), nn.Dropout(dropout)
            )
            self.frequency_expert_features = nn.Sequential(
                nn.LayerNorm(NATIVE_SPECTRAL_DIM),
                nn.Linear(NATIVE_SPECTRAL_DIM, expert_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.spatial_expert_classifier = nn.Linear(expert_hidden, 1)
            self.semantic_expert_classifier = nn.Linear(expert_hidden, 1)
            self.frequency_expert_classifier = nn.Linear(expert_hidden, 1)
            gate_dim = spatial_dim + semantic_dim + NATIVE_SPECTRAL_DIM
            self.gate_features = nn.Sequential(
                nn.LayerNorm(gate_dim), nn.Linear(gate_dim, gate_hidden), nn.GELU(), nn.Dropout(dropout)
            )
            self.gate_classifier = nn.Linear(gate_hidden, 3)

    def normalize_forensic(self, inputs: torch.Tensor) -> torch.Tensor:
        return (inputs - self.forensic_mean) / self.forensic_std

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        global_view = batch["global_view"]
        clean_global = batch["clean_global"]
        tiles = batch["tiles"]
        batch_size, tile_count = tiles.shape[:2]
        forensic_parts = [global_view, tiles.flatten(0, 1)]
        if self.training:
            forensic_parts.append(clean_global)
        forensic_inputs = torch.cat(forensic_parts, dim=0)
        forensic_embedding, _ = pooled_features(
            self.forensic, self.normalize_forensic(forensic_inputs)
        )
        global_embedding = forensic_embedding[:batch_size]
        tile_embeddings = forensic_embedding[batch_size : batch_size + batch_size * tile_count]
        tile_embeddings = tile_embeddings.view(batch_size, tile_count, -1)
        clean_embedding = forensic_embedding[-batch_size:] if self.training else global_embedding
        tile_pooled, tile_attention = self.tile_pool(tile_embeddings)
        components = [global_embedding, tile_pooled]
        clean_semantic_embedding = None
        if self.use_semantic:
            semantic_view = (batch["semantic_view"] - self.semantic_mean) / self.semantic_std
            if self.use_three_expert and self.training:
                clean_semantic_view = (
                    batch.get("clean_semantic_view", batch["semantic_view"]) - self.semantic_mean
                ) / self.semantic_std
                semantic_inputs = torch.cat([semantic_view, clean_semantic_view], dim=0)
                semantic_embeddings, _ = pooled_features(self.semantic, semantic_inputs)
                semantic_embedding = semantic_embeddings[:batch_size]
                clean_semantic_embedding = semantic_embeddings[batch_size:]
            else:
                semantic_embedding, _ = pooled_features(self.semantic, semantic_view)
                if self.use_three_expert:
                    clean_semantic_embedding = semantic_embedding
            components.append(semantic_embedding)
        wavelet_similarity = None
        if self.use_wavelet:
            perturbed = haar_high_frequency_perturbation(global_view)
            perturbed_embedding, _ = pooled_features(
                self.forensic, self.normalize_forensic(perturbed)
            )
            wavelet_similarity = F.cosine_similarity(global_embedding, perturbed_embedding, dim=-1)
            components.append(wavelet_similarity[:, None])
        hidden = self.fusion(torch.cat(components, dim=-1))
        legacy_logits = linear_fp32(self.classifier, hidden).squeeze(-1)
        expert_logits = None
        expert_gate = None
        clean_logits = None
        if self.use_three_expert:
            spatial_parts = [global_embedding, tile_pooled]
            clean_spatial_parts = [clean_embedding, clean_embedding]
            if self.use_wavelet:
                spatial_parts.append(wavelet_similarity[:, None])
                clean_spatial_parts.append(torch.ones_like(wavelet_similarity[:, None]))
            spatial_input = torch.cat(spatial_parts, dim=-1)
            clean_spatial_input = torch.cat(clean_spatial_parts, dim=-1)
            native_spectral = batch["native_spectral"].float()
            clean_native_spectral = batch.get("clean_native_spectral", native_spectral).float()

            spatial_hidden = self.spatial_expert_features(spatial_input)
            semantic_hidden = self.semantic_expert_features(semantic_embedding)
            frequency_hidden = self.frequency_expert_features(native_spectral)
            expert_logits = torch.stack(
                [
                    linear_fp32(self.spatial_expert_classifier, spatial_hidden).squeeze(-1),
                    linear_fp32(self.semantic_expert_classifier, semantic_hidden).squeeze(-1),
                    linear_fp32(self.frequency_expert_classifier, frequency_hidden).squeeze(-1),
                ],
                dim=-1,
            )
            gate_input = torch.cat([spatial_input, semantic_embedding, native_spectral], dim=-1)
            gate_hidden = self.gate_features(gate_input)
            expert_gate = torch.softmax(linear_fp32(self.gate_classifier, gate_hidden), dim=-1)
            logits = torch.sum(expert_gate * expert_logits, dim=-1)

            clean_spatial_hidden = self.spatial_expert_features(clean_spatial_input)
            clean_semantic_hidden = self.semantic_expert_features(clean_semantic_embedding)
            clean_frequency_hidden = self.frequency_expert_features(clean_native_spectral)
            clean_expert_logits = torch.stack(
                [
                    linear_fp32(self.spatial_expert_classifier, clean_spatial_hidden).squeeze(-1),
                    linear_fp32(self.semantic_expert_classifier, clean_semantic_hidden).squeeze(-1),
                    linear_fp32(self.frequency_expert_classifier, clean_frequency_hidden).squeeze(-1),
                ],
                dim=-1,
            )
            clean_gate_input = torch.cat(
                [clean_spatial_input, clean_semantic_embedding, clean_native_spectral], dim=-1
            )
            clean_gate_hidden = self.gate_features(clean_gate_input)
            clean_gate = torch.softmax(linear_fp32(self.gate_classifier, clean_gate_hidden), dim=-1)
            clean_logits = torch.sum(clean_gate * clean_expert_logits, dim=-1)
        else:
            logits = legacy_logits

        outputs = {
            "logits": logits.float(),
            "tile_logits": self.tile_head(tile_embeddings).squeeze(-1),
            "tile_attention": tile_attention,
            "degradation_logits": self.degradation_classifier(hidden),
            "degradation_severity": torch.sigmoid(self.degradation_regressor(hidden).squeeze(-1)),
            "embedding": global_embedding,
            "clean_embedding": clean_embedding,
            "aug_consistency_logits": self.consistency_head(global_embedding).squeeze(-1),
            "clean_consistency_logits": self.consistency_head(clean_embedding).squeeze(-1),
            "wavelet_similarity": wavelet_similarity
            if wavelet_similarity is not None
            else torch.ones(batch_size, device=global_view.device),
        }
        if expert_logits is not None:
            outputs.update(
                {
                    "expert_logits": expert_logits,
                    "expert_gate": expert_gate,
                    "clean_logits": clean_logits.float(),
                }
            )
        return outputs

    def trainable_parameter_summary(self) -> dict[str, int]:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        trainable_names = {name for name, parameter in self.named_parameters() if parameter.requires_grad}
        return {
            name: tensor.detach().cpu()
            for name, tensor in self.state_dict().items()
            if name in trainable_names
        }
