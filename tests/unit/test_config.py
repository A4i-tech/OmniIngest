import os

import pytest

from omni_ingest.core.config import ProfileManager


def test_profile_loading(tmp_path):
    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir()

    profile_content = """
    name: test_profile
    settings:
      secret_key: ${env.TEST_SECRET}
      default: value
    """
    (profiles_dir / "test_profile.yaml").write_text(profile_content)

    os.environ["TEST_SECRET"] = "my_secret_123"

    manager = ProfileManager(profiles_dir=str(profiles_dir))
    config = manager.load_profile("test_profile")

    assert config["name"] == "test_profile"
    assert config["settings"]["secret_key"] == "my_secret_123"
    assert config["settings"]["default"] == "value"

    del os.environ["TEST_SECRET"]


def test_profile_not_found():
    manager = ProfileManager(profiles_dir="non_existent")
    with pytest.raises(FileNotFoundError):
        manager.load_profile("missing_profile")
