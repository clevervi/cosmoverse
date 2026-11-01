"""
Configuration management and schema validation for COSMOVERSE experiments.
Supports YAML/JSON serialization, default presets, and scientific provenance tracking.
"""

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import BaseModel, Field

from cosmoverse.core.parameters import (
    BaryonParameters,
    CosmologicalParameters,
    ObservationalNuisanceParameters,
    SimulationParameters,
)


class CosmoverseConfig(BaseModel):
    """
    Complete configuration container for an end-to-end Cosmoverse run.
    Respects the three-level separation:
      1. Level 1 (Physics): cosmology, simulation, baryons
      2. Level 2 (Observational process): observational_nuisance
      3. Level 3 (Statistical inference): extra_settings
    """
    project_name: str = Field(default="cosmoverse-experiment", description="Experiment identifier")
    description: str = Field(default="", description="Scientific notes on this run")
    roadmap_stage: int = Field(default=1, ge=1, le=6, description="Active COSMOVERSE roadmap stage (1-6)")
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="Run initialization timestamp (UTC)"
    )
    git_hash: Optional[str] = Field(default=None, description="Git commit hash for full reproducibility")

    cosmology: CosmologicalParameters = Field(default_factory=CosmologicalParameters)
    simulation: SimulationParameters = Field(default_factory=SimulationParameters)
    baryons: BaryonParameters = Field(default_factory=BaryonParameters)
    observational_nuisance: ObservationalNuisanceParameters = Field(default_factory=ObservationalNuisanceParameters)

    extra_settings: Dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "CosmoverseConfig":
        """Load configuration from a YAML file."""
        file_path = Path(path)
        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to a YAML file."""
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        if self.git_hash is None:
            self.git_hash = get_git_revision_hash()
        with open(file_path, "w", encoding="utf-8") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_json(cls, path: str | Path) -> "CosmoverseConfig":
        """Load configuration from a JSON file."""
        with open(Path(path), "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)

    def to_json(self, path: str | Path) -> None:
        """Save configuration to a JSON file."""
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        if self.git_hash is None:
            self.git_hash = get_git_revision_hash()
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.model_dump(), f, indent=2)


def get_git_revision_hash() -> Optional[str]:
    """Retrieves current git commit SHA if in a repository."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=2,
        )
        return res.stdout.strip()
    except Exception:
        return None
