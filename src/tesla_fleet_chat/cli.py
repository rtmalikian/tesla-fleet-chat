"""CLI: tfleet auth | serve | ask | status | simulate"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

from .config import ConfigError, load_settings

__version__ = "0.1.0"


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")


def _client(settings):
    from .tesla_client import TeslaClient
    return TeslaClient(settings.client_id, settings.client_secret,
                       region=settings.region, redirect_uri=settings.redirect_uri,
                       token_file=settings.token_file)


def cmd_auth(args, settings) -> int:
    from .tesla_client import TeslaClient
    client = _client(settings)
    print("Open this URL in your browser and authorize:")
    print(client.authorize_url())
    print("Virtual-key pairing: https://developer.tesla.com/docs/fleet-api#pairing")
    return 0


def cmd_serve(args, settings) -> int:
    from .server import serve
    from .service import FleetChatService

    def client_factory():
        return _client(settings)

    service = FleetChatService(settings, client_factory=client_factory)
    print("Vehicle data goes ONLY to your configured LLM backend "
          f"(TFC_AI_BASE_URL={settings.ai_base_url or 'none'}); "
          "rules engine works with no backend at all.")
    serve(service)
    return 0


def cmd_ask(args, settings) -> int:
    from .fleet import snapshot_fleet
    from .intents import answer_question
    from .service import FleetChatService

    client = _client(settings)
    if not settings.vehicles:
        print("TFC_VEHICLES is not set — add name=VIN entries first.", file=sys.stderr)
        return 2
    service = FleetChatService(settings, client_factory=lambda: client)
    body = {"model": settings.model_id,
            "messages": [{"role": "user", "content": args.question}]}
    resp = service.chat_completions(body)
    print(resp["choices"][0]["message"]["content"])
    return 0


def cmd_status(args, settings) -> int:
    from .fleet import snapshot_fleet
    client = _client(settings)
    snap = snapshot_fleet(client, settings.vehicles)
    print(snap.brief())
    return 0


def cmd_simulate(_args, settings) -> int:
    from .simulator import run_demo
    settings.dry_run = False  # fake fleet; actually execute to show commands
    for question, answer, log in run_demo(settings):
        print(f"> {question}")
        print(f"  {answer}")
        for cmd, vin in log:
            print(f"  [fleet command] {cmd} {vin}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tfleet", description="Chat with your Teslas.")
    p.add_argument("--version", action="version", version=f"tfleet {__version__}")
    p.add_argument("--env-file", default=None)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("auth", help="Authorize with your Tesla account (one-time)")
    sub.add_parser("serve", help="Run the OpenAI-compatible API server")
    ask_p = sub.add_parser("ask", help="One-shot question/answer (no server)")
    ask_p.add_argument("question", help="What to ask or command")
    sub.add_parser("status", help="Show the live fleet brief")
    sub.add_parser("simulate", help="Demo Q&A against a fake fleet (no API calls)")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    env = dict(os.environ)
    if args.env_file:
        from dotenv import load_dotenv
        load_dotenv(args.env_file, override=True)
        env = dict(os.environ)
    else:
        from dotenv import load_dotenv
        load_dotenv(override=False)
        env = dict(os.environ)
    try:
        settings = load_settings(env)
    except ConfigError as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        return 2
    handlers = {"auth": cmd_auth, "serve": cmd_serve, "ask": cmd_ask,
                "status": cmd_status, "simulate": cmd_simulate}
    return handlers[args.command](args, settings)


if __name__ == "__main__":
    raise SystemExit(main())
