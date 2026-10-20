import pytest

from tesla_fleet_chat.config import ConfigError, load_settings

BASE = {"TFC_VEHICLES": "Model Y=VINY,Model 3=VIN3"}


def test_parse_vehicles():
    s = load_settings(dict(BASE))
    assert [v.name for v in s.vehicles] == ["Model Y", "Model 3"]
    assert s.server_port == 8642 and s.ai_mode == "auto"
    assert s.ai_base_url == "http://localhost:11434/v1"


def test_bad_region():
    with pytest.raises(ConfigError):
        load_settings({**BASE, "TESLA_REGION": "moon"})


def test_bad_ai_mode():
    with pytest.raises(ConfigError):
        load_settings({**BASE, "TFC_AI_MODE": "sometimes"})


def test_ai_base_trailing_slash_trimmed():
    s = load_settings({**BASE, "TFC_AI_BASE_URL": "http://x/v1/"})
    assert s.ai_base_url == "http://x/v1"


def test_dry_run_parse():
    assert load_settings({**BASE, "TFC_DRY_RUN": "1"}).dry_run is True
