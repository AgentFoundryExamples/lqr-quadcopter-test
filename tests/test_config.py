"""Tests for configuration loading utilities."""

import os
from unittest import mock


def test_get_default_config_has_required_keys():
    """Test that default config has all required keys."""
    from src.utils import get_default_config

    config = get_default_config()

    assert "seed" in config
    assert "episode_length" in config
    assert "target" in config
    assert "radius_requirement" in config["target"]
    assert "motion_type" in config["target"]
    assert "success_criteria" in config


def test_get_default_config_values():
    """Test that default config has expected values."""
    from src.utils import get_default_config

    config = get_default_config()

    assert config["seed"] == 42
    assert config["episode_length"] == 30.0
    assert config["target"]["radius_requirement"] == 0.5
    assert config["success_criteria"]["min_on_target_ratio"] == 0.8


def test_load_config_without_file():
    """Test config loading with defaults only."""
    from src.utils import load_config

    config = load_config(config_path=None, load_env=False)

    assert config["seed"] == 42
    assert config["episode_length"] == 30.0


def test_load_config_env_override():
    """Test that environment variables override defaults."""
    from src.utils import load_config

    with mock.patch.dict(os.environ, {"QUADCOPTER_SEED": "123"}):
        config = load_config(config_path=None, load_env=True)
        assert config["seed"] == 123


def test_load_config_target_env_override():
    """Test that target-specific env vars are applied."""
    from src.utils import load_config

    with mock.patch.dict(os.environ, {"QUADCOPTER_TARGET_RADIUS": "1.0"}):
        config = load_config(config_path=None, load_env=True)
        assert config["target"]["radius_requirement"] == 1.0
