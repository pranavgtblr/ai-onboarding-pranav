import os
from unittest import mock

from phase_0_baseline.config import Settings


def test_settings_defaults() -> None:
    settings = Settings()
    assert settings.app_name == "Phase 0 Baseline API"
    assert settings.app_env in ["development", "staging", "production", "test"]
    assert isinstance(settings.port, int)


def test_settings_from_env_vars() -> None:
    env_overrides = {
        "APP_NAME": "Custom AI App",
        "APP_ENV": "production",
        "DEBUG": "true",
        "PORT": "9000",
        "API_KEY": "supersecretkey123",
    }
    with mock.patch.dict(os.environ, env_overrides, clear=False):
        custom_settings = Settings()
        assert custom_settings.app_name == "Custom AI App"
        assert custom_settings.app_env == "production"
        assert custom_settings.debug is True
        assert custom_settings.port == 9000
        assert custom_settings.api_key == "supersecretkey123"
