"""24x7 channel worker: polls Email, Telegram and WhatsApp, runs every new message
through the agents and replies on the same channel.

    python channel_worker.py            # loop forever
    python channel_worker.py --once     # one polling round (handy for demos / cron)

Tickets processed here appear in the Streamlit dashboard and ticket log too,
because everything shares logs/tickets.csv.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import time
from typing import Dict, List

from channels.dispatcher import CHANNEL_CLASSES, Dispatcher
from channels.inbox import Inbox
from engine.config import Settings, get_settings
from engine.observability import setup_logging
from engine.orchestrator import SupportEngine
from engine.state import TicketInput, TicketResult

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
setup_logging(get_settings().log_file.parent)  # also write logs/app.log (rotating)
for noisy in ("autogen_core", "autogen_agentchat", "autogen_core.events", "httpx"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
log = logging.getLogger("channel_worker")


def poll_channels(settings: Settings, channels: List[str] = None) -> Dict[str, object]:
    """Fetch from every configured channel into the unified inbox."""
    dispatcher, inbox = Dispatcher(settings), Inbox(settings)
    report: Dict[str, object] = {}
    for name in channels or list(CHANNEL_CLASSES):
        ch = dispatcher.channel(name)
        if not ch.configured:
            report[name] = "not configured"
            continue
        try:
            report[name] = inbox.add(ch.fetch_new())
        except Exception as exc:
            report[name] = f"error: {type(exc).__name__}: {exc}"[:900]
    return report


async def process_inbox_item(msg_id: str, settings: Settings = None, on_event=None,
                             engine: SupportEngine = None) -> TicketResult | None:
    settings = settings or get_settings()
    inbox = Inbox(settings)
    if not inbox.claim(msg_id):
        return None
    msg = inbox.get(msg_id)
    engine = engine or SupportEngine(settings)
    try:
        result = await engine.process(
            TicketInput(complaint=msg.text, customer_name=msg.sender_name, channel=msg.channel,
                        customer_contact=msg.sender_id, subject=msg.subject, inbox_id=msg.id),
            on_event=on_event)
        inbox.update(msg_id, status="done", ticket_id=result.ticket_id)
        return result
    except Exception as exc:
        inbox.update(msg_id, status="error", ticket_id=f"error: {exc}"[:120])
        raise


async def run_once(settings: Settings) -> None:
    report = poll_channels(settings)
    log.info("poll: %s", report)
    engine = SupportEngine(settings)
    for item in Inbox(settings).list("new"):
        res = await process_inbox_item(item["id"], settings, engine=engine)
        if res:
            log.info("%s %s via %s -> %s", res.ticket_id, res.status, item["channel"],
                     [f"{d['channel']}:{d['status']}" for d in res.dispatch])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="run a single polling round and exit")
    args = parser.parse_args()
    settings = get_settings()
    log.info("Channel worker started (provider=%s, poll every %ss)", settings.llm_provider, settings.worker_poll_seconds)
    while True:
        try:
            asyncio.run(run_once(settings))
        except Exception:
            log.exception("worker round failed")
        if args.once:
            break
        time.sleep(settings.worker_poll_seconds)


if __name__ == "__main__":
    main()
