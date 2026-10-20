"""The setup wizard must produce a config the app can load — no developer
knowledge required from the (scripted) user."""
from tesla_fleet_chat.config import load_settings
from tesla_fleet_chat.setup import run_setup


def _scripted(answers):
    it = iter(answers)
    return lambda prompt="": next(it)


def _parse_env(path):
    env = {}
    for line in path.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            env[k] = v
    return env


def test_wizard_two_cars_ollama(tmp_path):
    answers = [
        "cid-123", "secret-xyz",          # step 1
        "Model Y", "VINY", "Model 3", "VIN3", "",  # step 2 (blank = done)
        "2",                              # step 3: ollama
        "", "",                           # step 4: default port, no password
    ]
    env_file = tmp_path / ".env"
    run_setup(input_fn=_scripted(answers), out=lambda *a: None,
              env_path=str(env_file))
    settings = load_settings(_parse_env(env_file))
    assert [v.name for v in settings.vehicles] == ["Model Y", "Model 3"]
    assert settings.ai_base_url == "http://localhost:11434/v1"
    assert settings.server_port == 8642


def test_wizard_no_ai_mode(tmp_path):
    answers = ["cid", "sec", "Solo", "VIN1", "", "1", "", ""]
    env_file = tmp_path / ".env"
    run_setup(input_fn=_scripted(answers), out=lambda *a: None,
              env_path=str(env_file))
    settings = load_settings(_parse_env(env_file))
    assert settings.ai_mode == "never"
    assert len(settings.vehicles) == 1


def test_wizard_keeps_existing_when_declined(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("TFC_VEHICLES=Keep=VIN0\n")
    run_setup(input_fn=_scripted(["n"]), out=lambda *a: None,
              env_path=str(env_file))
    assert "Keep" in env_file.read_text()
