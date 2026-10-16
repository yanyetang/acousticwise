from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_package_imports():
    import acousticwise

    assert acousticwise.__name__ == "acousticwise"


def test_default_config_has_audio_settings():
    config = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text())

    assert config["audio"]["sample_rate"] == 32000
    assert config["audio"]["window_seconds"] == 5.0
