"""Machine-readable provenance for engineered features."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Literal

import pandas as pd

FeatureKind = Literal["monthly", "aggregate", "metadata"]
OutputRepresentation = Literal["tabular", "sequence"]


class FeatureRegistryError(ValueError):
    """Raised when feature provenance is incomplete or ambiguous."""


@dataclass(frozen=True, slots=True)
class FeatureDefinition:
    """Scientific provenance and validity semantics for one output feature."""

    name: str
    feature_group: str
    source_bands: tuple[str, ...]
    formula: str
    temporal_aggregation: str | None
    validity_rule: str
    expected_dtype: str
    feature_kind: FeatureKind
    output_representation: OutputRepresentation
    version: str

    def __post_init__(self) -> None:
        if not self.name or not self.feature_group or not self.formula:
            raise FeatureRegistryError("feature name, group, and formula must be non-empty")
        if not self.validity_rule or not self.expected_dtype or not self.version:
            raise FeatureRegistryError("feature validity, dtype, and version must be documented")
        if len(self.source_bands) != len(set(self.source_bands)):
            raise FeatureRegistryError(f"feature '{self.name}' repeats a source band")


@dataclass(frozen=True, slots=True)
class FeatureRegistry:
    """Stable ordered collection of feature definitions."""

    definitions: tuple[FeatureDefinition, ...]

    def __post_init__(self) -> None:
        if not self.definitions:
            raise FeatureRegistryError("feature registry must not be empty")
        names = [definition.name for definition in self.definitions]
        if len(names) != len(set(names)):
            duplicates = sorted({name for name in names if names.count(name) > 1})
            raise FeatureRegistryError(f"feature registry contains duplicate names: {duplicates}")

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Return feature names in deterministic output order."""

        return tuple(definition.name for definition in self.definitions)

    @property
    def fingerprint(self) -> str:
        """Hash the complete ordered registry schema."""

        payload = json.dumps(
            [asdict(definition) for definition in self.definitions],
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(payload).hexdigest()

    def to_frame(self) -> pd.DataFrame:
        """Return a normalized table suitable for ignored audit artifacts."""

        return pd.DataFrame(
            [
                {
                    **asdict(definition),
                    "source_bands": ",".join(definition.source_bands),
                }
                for definition in self.definitions
            ]
        )


def combine_feature_registries(*registries: FeatureRegistry) -> FeatureRegistry:
    """Combine representation registries while retaining their supplied order."""

    return FeatureRegistry(
        definitions=tuple(
            definition for registry in registries for definition in registry.definitions
        )
    )
