import math

from tesla_fleet_chat.config import Settings, VehicleConfig
from tesla_fleet_chat.service import CAPABILITY_BLURB, FleetChatService
from tesla_fleet_chat.simulator import FakeTeslaClient, demo_vehicles


def make_service(ai_mode="never", llm=None, dry_run=False):
    settings = Settings(vehicles=[VehicleConfig("Model Y", "VINY"),
                                  VehicleConfig("Model 3", "VIN3")],
                        ai_mode=ai_mode, dry_run=dry_run)
    fake = FakeTeslaClient(demo_vehicles())
    service = FleetChatService(settings, client_factory=lambda: fake,
                               llm_factory=(lambda: llm) if llm is not None else None)
    return service, fake


def chat(service, text):
    body = {"model": "tesla-fleet-chat",
            "messages": [{"role": "user", "content": text}]}
    return service.chat_completions(body)


def test_chat_shape():
    service, _ = make_service()
    resp = chat(service, "status?")
    assert resp["object"] == "chat.completion"
    msg = resp["choices"][0]["message"]
    assert msg["role"] == "assistant" and "Model Y" in msg["content"]
    assert resp["model"] == "tesla-fleet-chat"


def test_models_shape():
    service, _ = make_service()
    models = service.models_list()
    assert models["object"] == "list"
    assert models["data"][0]["id"] == "tesla-fleet-chat"


def test_embeddings_shape_and_determinism():
    service, _ = make_service()
    r1 = service.embeddings({"input": "charging the model y"})
    r2 = service.embeddings({"input": "charging the model y"})
    vec = r1["data"][0]["embedding"]
    assert len(vec) == 384
    assert vec == r2["data"][0]["embedding"]
    assert abs(math.sqrt(sum(x * x for x in vec)) - 1.0) < 1e-6


def test_embeddings_batch():
    service, _ = make_service()
    r = service.embeddings({"input": ["one", "two"]})
    assert len(r["data"]) == 2 and r["data"][1]["index"] == 1


class FakeLlm:
    def __init__(self, reply):
        self.reply = reply
        self.seen_brief = ""

    def chat(self, brief, user_message):
        self.seen_brief = brief
        return self.reply


def test_llm_fallback_answer_gets_brief():
    llm = FakeLlm("The Model Y is your fuller car tonight.")
    service, _ = make_service(ai_mode="auto", llm=llm)
    resp = chat(service, "Tell me a pirate tale about my automobiles")
    assert resp["choices"][0]["message"]["content"].startswith("The Model Y")
    assert "battery 82%" in llm.seen_brief


def test_llm_command_block_executes_after_validation():
    llm = FakeLlm('```json\n{"command": "charge_start", "vehicle": "Model Y"}\n```')
    service, fake = make_service(ai_mode="auto", llm=llm)
    resp = chat(service, "get the Y juicing please")
    assert fake.command_log == [("charge_start", "VINY")]
    assert "charge_start" in resp["choices"][0]["message"]["content"]


def test_llm_disallowed_command_rejected():
    llm = FakeLlm('```json\n{"command": "open_frunk", "vehicle": "Model Y"}\n```')
    service, fake = make_service(ai_mode="auto", llm=llm)
    chat(service, "pop the frunk")
    assert fake.command_log == []


def test_llm_never_mode_uses_capability_blurb():
    llm = FakeLlm("should not be used")
    service, _ = make_service(ai_mode="never")
    resp = chat(service, "tell me a poem about my cars")
    assert resp["choices"][0]["message"]["content"] == CAPABILITY_BLURB
