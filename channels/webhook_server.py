"""Optional FastAPI webhook receiver for WhatsApp (Twilio) and Telegram.

Run:  uvicorn channels.webhook_server:app --port 8000
Expose it (e.g. ngrok http 8000) and point:
  * Twilio WhatsApp sandbox "WHEN A MESSAGE COMES IN" -> https://<host>/webhooks/whatsapp
  * Telegram: https://api.telegram.org/bot<TOKEN>/setWebhook?url=https://<host>/webhooks/telegram
    (when a webhook is set, Telegram disables getUpdates polling)

Messages are put in the unified inbox. With AUTO_PROCESS=true they are answered
straight away by the agents; otherwise the Streamlit Channel Inbox or the
channel worker picks them up.
"""
from __future__ import annotations

import asyncio
import os

from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import PlainTextResponse

from channels.inbox import Inbox
from channels.telegram_channel import parse_update
from channels.whatsapp_channel import parse_twilio_form
from engine.config import get_settings

app = FastAPI(title="Remya Industries Ltd Support Webhooks")
settings = get_settings()
inbox = Inbox(settings)
AUTO_PROCESS = os.getenv("AUTO_PROCESS", "false").lower() in {"1", "true", "yes"}


def _process(msg_id: str) -> None:
    from channel_worker import process_inbox_item

    asyncio.run(process_inbox_item(msg_id, settings))


@app.get("/health")
def health():
    return {"ok": True, "inbox_new": len(inbox.list("new"))}


@app.post("/webhooks/whatsapp")
async def whatsapp(request: Request, tasks: BackgroundTasks):
    form = dict(await request.form())
    msg = parse_twilio_form(form)
    if msg and inbox.add([msg]) and AUTO_PROCESS:
        tasks.add_task(_process, msg.id)
    # empty TwiML: the reply is sent later through the REST API by the Log Agent
    return PlainTextResponse("<Response></Response>", media_type="application/xml")


@app.post("/webhooks/telegram")
async def telegram(request: Request, tasks: BackgroundTasks):
    msg = parse_update(await request.json())
    if msg and inbox.add([msg]) and AUTO_PROCESS:
        tasks.add_task(_process, msg.id)
    return {"ok": True}
