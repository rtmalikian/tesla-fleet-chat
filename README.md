# Tesla Fleet Chat — Chat With Your Teslas Through Any OpenAI Client (OpenAI-Compatible Fleet API Server)

**Free, open-source server that turns your Tesla fleet into an OpenAI-compatible API endpoint.** Point any OpenAI client — ChatGPT-style UIs, Shortcuts, scripts, voice assistants — at it and ask *"which car has more battery?"* or command *"start charging the Model Y"*, answered and executed against the **official Tesla Fleet API**. Part of an AI-architect's Tesla app series: every app embeds OpenAI-compatible endpoints with a pluggable LLM backend.

Works with Model 3, Model Y, Model S, Model X, and Cybertruck.

▶ **[Try the live demo](https://rtmalikian.github.io/tesla-fleet-chat/)** — chat with a pretend fleet in your browser, no install, no car needed.

## Why?

Tesla's voice commands only work **inside the car**, and the Tesla app has no conversational interface at all. Meanwhile every AI tool you already use speaks one protocol: the OpenAI API. This app bridges them — your cars become a "model" any OpenAI-compatible client can chat with:

- **Ask anything** — battery, charging state, cabin temp, lock/sentry status, odometer, grounded in live no-wake Fleet API reads
- **Command by sentence** — start/stop charging, precondition, sentry on/off — via a strict allowlist, with dry-run mode
- **Pluggable brain** — a deterministic rules engine answers the common questions with **zero AI involved**; connect any OpenAI-compatible LLM backend (local **Ollama**/llama.cpp by default, or OpenAI/Groq/LM Studio) and it handles free-form questions with a live "fleet brief" as context. LLM-proposed commands are validated against the same allowlist before anything touches your car
- **Privacy by architecture** — vehicle data is sent **only** to the backend *you* configure. With `TFC_AI_MODE=never` (or no backend), nothing leaves your machine except Tesla API calls. Runs happily on a Raspberry Pi

## AI endpoints

```bash
# run it
tfleet serve            # listens on http://127.0.0.1:8642

# use it like the OpenAI API (any client works — set base URL + any key)
curl http://127.0.0.1:8642/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "tesla-fleet-chat",
       "messages": [{"role": "user", "content": "which car has more battery?"}]}'

curl http://127.0.0.1:8642/v1/models
curl http://127.0.0.1:8642/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": "charging the Model Y overnight"}'
```

- `POST /v1/chat/completions` — OpenAI-shaped responses (streaming not supported)
- `GET /v1/models` — advertises the fleet as a model
- `POST /v1/embeddings` — deterministic local feature-hashing embeddings (dim 384, zero external calls) for building your own retrieval over fleet text
- Optional bearer auth (`TFC_SERVER_API_KEY`) — clients authenticate exactly like OpenAI

## Quick start

```bash
git clone https://github.com/rtmalikian/tesla-fleet-chat && cd tesla-fleet-chat
python3 -m venv .venv && .venv/bin/pip install -e .
.venv/bin/tfleet setup  # guided setup — plain-language questions, writes .env for you
.venv/bin/tfleet auth   # one-time Tesla authorization + virtual-key pairing
.venv/bin/tfleet serve
# or one-shot, no server:
.venv/bin/tfleet ask "start charging the Model Y"
.venv/bin/tfleet simulate   # full demo against a fake fleet — no car, no keys
```

Optional LLM backend (default: local Ollama):

```bash
ollama serve && ollama pull llama3.1   # or set TFC_AI_BASE_URL to any OpenAI-compatible API
```

## How it works

1. On each chat message the server takes a **no-wake snapshot** of every configured car (battery, charging, climate, sentry, locks, odometer).
2. The **rules engine** tries to answer/execute deterministically — exact data, zero hallucination surface.
3. Unmatched questions escalate to the **LLM backend** (if configured) with the snapshot as its only facts. If the model proposes a command, it must come back as a JSON block naming an **allowlisted command** and an exact vehicle name — validated, then executed.
4. Anything else falls back to a capability list, so the chat never silently fails.

## Cost

Fleet API commands are pay-per-use with Tesla's $10/month developer discovery credit; questions are no-wake data reads and commands are a handful per day in normal chat use — personal use sits comfortably inside the credit. The embeddings endpoint and rules engine are 100% local and free.

## Configuration

Everything lives in `.env` (git-ignored — secrets never leave your machine). See `.env.example`: `TFC_VEHICLES`, `TFC_SERVER_PORT`, `TFC_SERVER_API_KEY`, `TFC_AI_BASE_URL` / `TFC_AI_API_KEY` / `TFC_AI_MODEL` / `TFC_AI_MODE` (`auto` rules-first | `always` LLM-first | `never` rules-only), `TFC_DRY_RUN`.

## Part of a series

One Tesla pain point per day, each with embedded OpenAI-compatible AI: calendar-aware preconditioning (`tesla-calendar-climate`), phantom-drain alerts (`tesla-drain-detective`), per-location charge limits, rain-closing windows, luxury scenes, BYD-style V2L planning, and more — all on this GitHub profile.

## License

MIT © Raphael Malikian — free to use, fork, and ship.
