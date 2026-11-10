"""
Config loader for RUNorm.
Loads pipeline.yaml and language configs from the configs/ directory.
"""

from pathlib import Path
from typing import Any, Dict
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_CONFIGS_DIR = _PROJECT_ROOT / "configs"


def load_pipeline_config() -> Dict[str, Any]:
    """Loads the main pipeline configuration from configs/pipeline.yaml."""
    path = _CONFIGS_DIR / "pipeline.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Pipeline config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_language_config(language_code: str = "ur_roman") -> Dict[str, Any]:
    """
    Loads a language-specific configuration from configs/languages/<language_code>.yaml.

    Args:
        language_code: The language identifier (e.g., 'ur_roman').

    Returns:
        Parsed YAML config as a dict.
    """
    path = _CONFIGS_DIR / "languages" / f"{language_code}.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"Language config not found for '{language_code}': {path}"
        )
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
