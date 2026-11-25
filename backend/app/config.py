"""Provider configuration loading without exposing credential values."""

from dataclasses import dataclass
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BACKEND_DIR / ".env")


@dataclass(frozen=True)
class ModelConfig:
    id: str
    display_name: str
    base_url: str
    model: str
    api_key_env: str
    requires_api_key: bool = True
    enabled: bool = True
    requests_per_minute: int = 30

    @property
    def api_key(self) -> str | None:
        configured = os.getenv(self.api_key_env)
        if configured:
            return configured
        return "ollama" if not self.requires_api_key else None


def load_models(path: Path | None = None) -> list[ModelConfig]:
    config_path = path or BACKEND_DIR / "config" / "models.yaml"
    with config_path.open(encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}
    configured_models = [ModelConfig(**item) for item in raw.get("models", [])]
    optional_ids = {
        model_id.strip()
        for model_id in os.getenv("TRUSTMETER_INCLUDE_OPTIONAL_MODELS", "").split(",")
        if model_id.strip()
    }
    configured_ids = {model.id for model in configured_models}
    unknown_optional_ids = optional_ids - configured_ids
    if unknown_optional_ids:
        raise ValueError(
            "Unknown optional model ID(s): " + ", ".join(sorted(unknown_optional_ids))
        )
    models = [
        model for model in configured_models
        if model.enabled or model.id in optional_ids
    ]
    if not models:
        raise ValueError(f"No model providers configured in {config_path}.")
    if any(model.requests_per_minute < 1 for model in models):
        raise ValueError("Provider requests_per_minute must be at least 1.")
    provider_limits: dict[tuple[str, str], int] = {}
    for model in models:
        provider_key = (model.base_url.rstrip("/"), model.api_key_env)
        configured_limit = provider_limits.setdefault(
            provider_key, model.requests_per_minute
        )
        if configured_limit != model.requests_per_minute:
            raise ValueError(
                "Models sharing a provider URL and key variable must use the same "
                "requests_per_minute value."
            )
    ids = [model.id for model in models]
    if len(set(ids)) != len(ids):
        raise ValueError("Provider model IDs must be unique.")
    return models
