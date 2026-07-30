"""Compact masked temporal models for Phase 6 viability experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class TemporalModelError(ValueError):
    """Raised when a temporal batch or architecture violates mask contracts."""


@dataclass(frozen=True, slots=True)
class TemporalArchitecture:
    """Small CPU-compatible GRU architecture declared by configuration."""

    radar_channels: int
    optical_channels: int
    index_channels: int
    sensor_projection_dim: int = 24
    index_projection_dim: int = 16
    hidden_dim: int = 64
    num_layers: int = 1
    dropout: float = 0.15

    def __post_init__(self) -> None:
        if min(self.radar_channels, self.optical_channels, self.index_channels) < 1:
            raise TemporalModelError("all temporal input groups require at least one channel")
        if min(self.sensor_projection_dim, self.index_projection_dim, self.hidden_dim) < 1:
            raise TemporalModelError("temporal hidden dimensions must be positive")
        if self.num_layers not in {1, 2}:
            raise TemporalModelError("compact temporal model supports one or two GRU layers")
        if not 0.0 <= self.dropout < 1.0:
            raise TemporalModelError("temporal dropout must be in [0, 1)")


@dataclass(frozen=True, slots=True)
class TemporalForwardOutput:
    """Logits plus pooled representation and interpretable sensor gate."""

    logits: Tensor
    embedding: Tensor
    optical_gate: Tensor


def masked_mean_pool(values: Tensor, padding_mask: Tensor) -> Tensor:
    """Pool valid temporal positions while excluding explicit right padding."""

    if values.ndim != 3 or padding_mask.ndim != 2:
        raise TemporalModelError("masked pooling expects [batch,time,channels] and [batch,time]")
    if values.shape[:2] != padding_mask.shape:
        raise TemporalModelError("masked pooling values and padding mask are misaligned")
    valid = (~padding_mask).to(values.dtype).unsqueeze(-1)
    counts = valid.sum(dim=1)
    if torch.any(counts <= 0):
        raise TemporalModelError("every temporal sample needs at least one non-padded position")
    return (values * valid).sum(dim=1) / counts


class SensorGatedGRU(nn.Module):
    """Dual-sensor projections fused by availability-aware element-wise gating."""

    def __init__(self, architecture: TemporalArchitecture) -> None:
        super().__init__()
        self.architecture = architecture
        sensor_dim = architecture.sensor_projection_dim
        index_dim = architecture.index_projection_dim
        self.radar_projection = nn.Sequential(
            nn.Linear(architecture.radar_channels * 2 + 1, sensor_dim),
            nn.LayerNorm(sensor_dim),
            nn.GELU(),
        )
        self.optical_projection = nn.Sequential(
            nn.Linear(architecture.optical_channels * 2 + 1, sensor_dim),
            nn.LayerNorm(sensor_dim),
            nn.GELU(),
        )
        self.index_projection = nn.Sequential(
            nn.Linear(architecture.index_channels * 2, index_dim),
            nn.LayerNorm(index_dim),
            nn.GELU(),
        )
        self.sensor_gate = nn.Linear(sensor_dim * 2 + 2, sensor_dim)
        temporal_input_dim = sensor_dim + index_dim + 5
        self.temporal_encoder = nn.GRU(
            input_size=temporal_input_dim,
            hidden_size=architecture.hidden_dim,
            num_layers=architecture.num_layers,
            batch_first=True,
            dropout=architecture.dropout if architecture.num_layers > 1 else 0.0,
        )
        head_dim = max(16, architecture.hidden_dim // 2)
        self.classifier = nn.Sequential(
            nn.LayerNorm(architecture.hidden_dim),
            nn.Linear(architecture.hidden_dim, head_dim),
            nn.GELU(),
            nn.Dropout(architecture.dropout),
            nn.Linear(head_dim, 1),
        )

    @staticmethod
    def _validate_batch(batch: dict[str, Tensor]) -> None:
        required = {
            "radar_values",
            "optical_values",
            "index_values",
            "radar_feature_mask",
            "optical_feature_mask",
            "index_mask",
            "radar_mask",
            "optical_mask",
            "padding_mask",
            "relative_positions",
            "month_encoding",
        }
        missing = sorted(required - set(batch))
        if missing:
            raise TemporalModelError(f"temporal batch is missing tensors: {missing}")
        batch_size, time_steps = batch["padding_mask"].shape
        for name in ("radar_mask", "optical_mask", "relative_positions"):
            if batch[name].shape != (batch_size, time_steps):
                raise TemporalModelError(f"temporal tensor '{name}' is misaligned")
        if batch["month_encoding"].shape != (batch_size, time_steps, 2):
            raise TemporalModelError("month encoding must have two cyclic channels")
        valid_lengths = (~batch["padding_mask"]).sum(dim=1)
        if torch.any(valid_lengths < 1):
            raise TemporalModelError("a temporal batch contains an empty sequence")
        # Phase 2 uses right padding only; packed GRU relies on this invariant.
        expected_padding = torch.arange(time_steps, device=valid_lengths.device).unsqueeze(0)
        expected_padding = expected_padding >= valid_lengths.unsqueeze(1)
        if not torch.equal(expected_padding, batch["padding_mask"]):
            raise TemporalModelError("temporal sequences must use contiguous right padding")

    def forward(
        self,
        batch: dict[str, Tensor],
        *,
        ablate_radar: bool = False,
        ablate_optical: bool = False,
        ablate_indices: bool = False,
    ) -> TemporalForwardOutput:
        """Return one logit per window with optional sensor masking diagnostics."""

        self._validate_batch(batch)
        radar_values = batch["radar_values"]
        optical_values = batch["optical_values"]
        index_values = batch["index_values"]
        radar_feature_mask = batch["radar_feature_mask"].to(radar_values.dtype)
        optical_feature_mask = batch["optical_feature_mask"].to(optical_values.dtype)
        index_mask = batch["index_mask"].to(index_values.dtype)
        radar_available = batch["radar_mask"].to(radar_values.dtype)
        optical_available = batch["optical_mask"].to(optical_values.dtype)

        if ablate_radar:
            radar_values = torch.zeros_like(radar_values)
            radar_feature_mask = torch.zeros_like(radar_feature_mask)
            radar_available = torch.zeros_like(radar_available)
        if ablate_optical:
            optical_values = torch.zeros_like(optical_values)
            optical_feature_mask = torch.zeros_like(optical_feature_mask)
            optical_available = torch.zeros_like(optical_available)
        if ablate_indices:
            index_values = torch.zeros_like(index_values)
            index_mask = torch.zeros_like(index_mask)

        radar_input = torch.cat(
            (radar_values, radar_feature_mask, radar_available.unsqueeze(-1)), dim=-1
        )
        optical_input = torch.cat(
            (optical_values, optical_feature_mask, optical_available.unsqueeze(-1)), dim=-1
        )
        index_input = torch.cat((index_values, index_mask), dim=-1)
        radar_hidden = self.radar_projection(radar_input)
        optical_hidden = self.optical_projection(optical_input)
        index_hidden = self.index_projection(index_input)

        raw_gate = torch.sigmoid(
            self.sensor_gate(
                torch.cat(
                    (
                        radar_hidden,
                        optical_hidden,
                        radar_available.unsqueeze(-1),
                        optical_available.unsqueeze(-1),
                    ),
                    dim=-1,
                )
            )
        )
        both = (radar_available > 0.5) & (optical_available > 0.5)
        optical_only = (radar_available <= 0.5) & (optical_available > 0.5)
        gate = torch.where(both.unsqueeze(-1), raw_gate, torch.zeros_like(raw_gate))
        gate = torch.where(optical_only.unsqueeze(-1), torch.ones_like(gate), gate)
        fused_sensor = gate * optical_hidden + (1.0 - gate) * radar_hidden

        relative = batch["relative_positions"].unsqueeze(-1)
        time_features = torch.cat(
            (
                batch["month_encoding"],
                relative,
                radar_available.unsqueeze(-1),
                optical_available.unsqueeze(-1),
            ),
            dim=-1,
        )
        temporal_input = torch.cat((fused_sensor, index_hidden, time_features), dim=-1)
        temporal_input = temporal_input.masked_fill(batch["padding_mask"].unsqueeze(-1), 0.0)
        lengths = (~batch["padding_mask"]).sum(dim=1).to(torch.int64)
        packed = pack_padded_sequence(
            temporal_input,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, _ = self.temporal_encoder(packed)
        encoded, _ = pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=batch["padding_mask"].shape[1],
        )
        embedding = masked_mean_pool(encoded, batch["padding_mask"])
        logits = self.classifier(embedding).squeeze(-1)
        gate = gate.masked_fill(batch["padding_mask"].unsqueeze(-1), 0.0)
        return TemporalForwardOutput(logits=logits, embedding=embedding, optical_gate=gate)


def count_trainable_parameters(model: nn.Module) -> int:
    """Return the exact number of optimized parameters."""

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def architecture_from_dict(
    value: dict[str, Any],
    *,
    radar_channels: int,
    optical_channels: int,
    index_channels: int,
) -> TemporalArchitecture:
    """Build a validated architecture from a scalar experiment mapping."""

    allowed = {
        "sensor_projection_dim",
        "index_projection_dim",
        "hidden_dim",
        "num_layers",
        "dropout",
    }
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise TemporalModelError(f"unsupported temporal architecture keys: {unknown}")
    return TemporalArchitecture(
        radar_channels=radar_channels,
        optical_channels=optical_channels,
        index_channels=index_channels,
        sensor_projection_dim=int(value.get("sensor_projection_dim", 24)),
        index_projection_dim=int(value.get("index_projection_dim", 16)),
        hidden_dim=int(value.get("hidden_dim", 64)),
        num_layers=int(value.get("num_layers", 1)),
        dropout=float(value.get("dropout", 0.15)),
    )
