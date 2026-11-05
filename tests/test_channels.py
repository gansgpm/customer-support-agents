from channels.base import InboundMessage, Outbox
from channels.dispatcher import Dispatcher
from channels.email_channel import parse_email, render_email_html
from channels.inbox import Inbox
from channels.telegram_channel import TelegramChannel, parse_update
from channels.whatsapp_channel import WhatsAppChannel, normalize_number, parse_twilio_form


class FakeResponse:
    def __init__(self, payload, status=200):
        self._payload, self.status_code, self.content = payload, status, b"x"

    def json(self):
        return self._payload

    def raise_for_status(self):
        pass


class FakeSession:
    def __init__(self, get_payload=None, post_payload=None):
        self.calls, self.get_payload, self.post_payload = [], get_payload, post_payload

    def get(self, url, **kw):
        self.calls.append(("GET", url, kw))
        return FakeResponse(self.get_payload)

    def post(self, url, **kw):
        self.calls.append(("POST", url, kw))
        return FakeResponse(self.post_payload, 201)


def test_twilio_webhook_and_number():
    m = parse_twilio_form({"From": "whatsapp:+919800000001", "Body": " Where is my order? ", "MessageSid": "SM9"})
    assert m.channel == "whatsapp" and m.text == "Where is my order?" and m.external_id == "SM9"
    assert normalize_number("98000 00001") == "whatsapp:+9800000001"


def test_telegram_update_parsing_ignores_start():
    assert parse_update({"update_id": 1, "message": {"chat": {"id": 5}, "text": "/start"}}) is None
    m = parse_update({"update_id": 2, "message": {"chat": {"id": 5}, "from": {"first_name": "Ravi"}, "text": "hi"}})
    assert m.sender_id == "5" and m.sender_name == "Ravi"


def test_email_parsing_strips_quotes():
    raw = (b"From: Anita <anita@example.com>\r\nSubject: Help\r\nContent-Type: text/plain\r\n\r\n"
           b"My blender is broken.\r\n\r\nOn Tue, Support wrote:\r\n> earlier text\r\n")
    m = parse_email(raw, "7")
    assert m.sender_id == "anita@example.com" and m.text == "My blender is broken."


def test_email_template_wraps_reply():
    html = render_email_html("Remya Industries Ltd", "TKT-0009", "Resolved", "Hi\n\n- step 1\n- step 2", "my complaint")
    assert "TKT-0009" in html and "RESOLVED" in html and "<li>step 1</li>" in html


def test_telegram_send_and_fetch_with_fake_api(settings):
    s = settings.with_overrides(telegram_bot_token="123:abc")
    session = FakeSession(get_payload={"result": [{"update_id": 10, "message": {"chat": {"id": 77}, "text": "hello"}}]},
                          post_payload={"ok": True, "result": {"message_id": 3}})
    ch = TelegramChannel(s, session=session)
    msgs = ch.fetch_new()
    assert msgs[0].sender_id == "77"
    assert (s.data_dir / "telegram_offset.txt").read_text() == "11"
    res = ch.send("77", "reply text")
    assert res.status == "sent" and session.calls[-1][2]["json"]["chat_id"] == "77"


def test_whatsapp_send_with_fake_twilio(settings):
    s = settings.with_overrides(twilio_account_sid="AC123", twilio_auth_token="tok")
    session = FakeSession(post_payload={"sid": "SM1"})
    res = WhatsAppChannel(s, session=session).send("+919800000001", "hi")
    assert res.status == "sent"
    assert session.calls[-1][2]["data"]["To"] == "whatsapp:+919800000001"


def test_unconfigured_channels_simulate_and_record(settings):
    d = Dispatcher(settings)
    out = [d.send_to_customer(ch, to, "TKT-1", "Resolved", "Hello", "c")
           for ch, to in [("email", "a@b.com"), ("whatsapp", "+911"), ("telegram", "9")]]
    assert [o["status"] for o in out] == ["simulated"] * 3
    assert len(Outbox(settings).read()) == 3


def test_inbox_dedup_and_claim(settings):
    inbox = Inbox(settings)
    msg = InboundMessage(channel="telegram", sender_id="1", text="x", external_id="U1")
    assert inbox.add([msg]) == 1 and inbox.add([InboundMessage(channel="telegram", sender_id="1", text="x",
                                                               external_id="U1")]) == 0
    assert inbox.claim(msg.id) and not inbox.claim(msg.id)


def test_telegram_setup_helpers_do_not_consume_updates(settings):
    s = settings.with_overrides(telegram_bot_token="123:abc")

    class Session(FakeSession):
        def get(self, url, **kw):
            self.calls.append(("GET", url, kw))
            if url.endswith("getMe"):
                return FakeResponse({"ok": True, "result": {"username": "remya_support_bot", "first_name": "Desk"}})
            return FakeResponse({"ok": True, "result": [
                {"update_id": 5, "message": {"chat": {"id": 987654321, "first_name": "Remya", "type": "private"},
                                             "text": "hello"}}]})

    session = Session()
    tg = TelegramChannel(s, session=session)
    assert tg.get_me()["username"] == "remya_support_bot"
    chats = tg.find_chats()
    assert chats == [{"chat_id": "987654321", "name": "Remya", "type": "private", "last_message": "hello"}]
    assert "offset" not in session.calls[-1][2]["params"]  # updates stay for the inbox
    assert not (s.data_dir / "telegram_offset.txt").exists()


def test_automated_emails_are_ignored():
    raw = (b"From: Shop <no-reply@shop.com>\r\nSubject: Sale\r\nList-Unsubscribe: <x>\r\n"
           b"Content-Type: text/plain\r\n\r\nBig sale")
    assert parse_email(raw, "1").meta["automated"]
    raw = b"From: Anita <anita@example.com>\r\nSubject: Help\r\nContent-Type: text/plain\r\n\r\nBlender broken"
    assert not parse_email(raw, "2").meta["automated"]


def test_telegram_chat_not_found_is_explained(settings):
    from channels.telegram_channel import check_chat_id

    s = settings.with_overrides(telegram_bot_token="123456789:AAbc")
    session = FakeSession(post_payload={"ok": False, "error_code": 400, "description": "Bad Request: chat not found"})
    res = TelegramChannel(s, session=session).send("123456789", "hi")
    assert res.status == "error" and "BOT's own ID" in res.detail and "press Start" in res.detail
    assert "phone number" in check_chat_id("+91 98765 43210")
    assert "example" in check_chat_id("987654321")
    assert "BotFather" in check_chat_id("93372553")
    assert "not a chat ID" in check_chat_id("hello")
    assert check_chat_id("5550001112") is None and check_chat_id("-1001234567890") is None


class _FakeSMTP:
    """Stands in for smtplib.SMTP (port 587); records what was sent."""
    sent = []

    def __init__(self, host, port, timeout=None, **kw):
        self.port = port

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def ehlo(self):
        pass

    def starttls(self, context=None):
        pass

    def login(self, user, password):
        pass

    def send_message(self, msg):
        _FakeSMTP.sent.append((self.port, msg["To"], msg["From"]))


def _email_settings(settings):
    return settings.with_overrides(email_address="desk@gmail.com", email_password="abcdabcdabcdabcd",
                                   email_smtp_port=465, support_signature="REMYA INDUSTRIES LTD CUSTOMER CARE")


def test_email_retries_on_port_587_when_465_is_cut(settings, monkeypatch):
    import smtplib
    from channels import email_channel as ec

    def cut(*a, **k):
        raise smtplib.SMTPServerDisconnected("Connection unexpectedly closed")

    def no_imap(*a, **k):
        raise OSError("no network in tests")

    _FakeSMTP.sent = []
    ec._WORKING_SMTP_ROUTE.clear()
    monkeypatch.setattr(ec.smtplib, "SMTP_SSL", cut)
    monkeypatch.setattr(ec.smtplib, "SMTP", _FakeSMTP)
    monkeypatch.setattr(ec.imaplib, "IMAP4_SSL", no_imap)
    monkeypatch.setattr(ec, "IMAP4SSLv4", no_imap)
    ch = ec.EmailChannel(_email_settings(settings))
    res = ch.send("anita@example.com", "hello", "Subject")
    assert res.status == "sent" and "port 587" in res.detail
    assert _FakeSMTP.sent[0][:2] == (587, "anita@example.com")
    assert "REMYA INDUSTRIES LTD CUSTOMER CARE" in _FakeSMTP.sent[0][2]
    assert ch.check()["smtp"].startswith("ok on port 587")  # tells the user to set EMAIL_SMTP_PORT=587
    ec._WORKING_SMTP_ROUTE.clear()


def test_email_network_block_and_bad_password_are_explained(settings, monkeypatch):
    import smtplib
    from channels import email_channel as ec

    def cut(*a, **k):
        raise smtplib.SMTPServerDisconnected("Connection unexpectedly closed")

    ec._WORKING_SMTP_ROUTE.clear()
    monkeypatch.setattr(ec.smtplib, "SMTP_SSL", cut)
    monkeypatch.setattr(ec.smtplib, "SMTP", cut)
    res = ec.EmailChannel(_email_settings(settings)).send("anita@example.com", "hello")
    assert res.status == "error" and "antivirus" in res.detail and "port 465" in res.detail and "port 587" in res.detail

    class BadLogin(_FakeSMTP):
        def login(self, user, password):
            raise smtplib.SMTPAuthenticationError(535, b"5.7.8 Username and Password not accepted")

    calls = []
    monkeypatch.setattr(ec.smtplib, "SMTP_SSL", lambda *a, **k: BadLogin("h", 465))
    monkeypatch.setattr(ec.smtplib, "SMTP", lambda *a, **k: calls.append(a) or _FakeSMTP("h", 587))
    res = ec.EmailChannel(_email_settings(settings)).send("anita@example.com", "hello")
    assert res.status == "error" and "App Password" in res.detail and not calls  # a refused login is not retried


def test_app_password_spaces_are_ignored(monkeypatch):
    from engine.config import Settings

    monkeypatch.setenv("EMAIL_APP_PASSWORD", "abcd efgh ijkl mnop")
    assert Settings().email_password == "abcdefghijklmnop"


def test_email_falls_back_to_ipv4_when_the_default_route_is_dropped(settings, monkeypatch):
    import smtplib
    from channels import email_channel as ec

    class Route(_FakeSMTP):
        def __init__(self, host, port, timeout=None, source_address=None, **kw):
            if source_address != ec.IPV4_ONLY:  # IPv6 path drops the connection (seen on some home networks)
                raise smtplib.SMTPServerDisconnected("Connection unexpectedly closed")
            super().__init__(host, port, timeout)

    _FakeSMTP.sent = []
    ec._WORKING_SMTP_ROUTE.clear()
    monkeypatch.setattr(ec.smtplib, "SMTP_SSL", Route)
    monkeypatch.setattr(ec.smtplib, "SMTP", Route)
    res = ec.EmailChannel(_email_settings(settings)).send("anita@example.com", "hello")
    assert res.status == "sent" and "port 465 over IPv4" in res.detail and _FakeSMTP.sent[0][0] == 465
    assert ec._WORKING_SMTP_ROUTE["smtp.gmail.com"] == (465, True)  # next mail goes straight there
    ec._WORKING_SMTP_ROUTE.clear()


def test_gmail_wrapped_reply_header_is_stripped():
    from channels.email_channel import strip_quoted

    text = ("Thanks, the test worked. My blender is still broken.\n\n"
            "On Tue, Nov 3, 2026 at 10:58 PM REMYA INDUSTRIES LTD CUSTOMER CARE <\n"
            "support.desk@example.com> wrote:\n\n> Connection test from your AI support desk\n")
    assert strip_quoted(text) == "Thanks, the test worked. My blender is still broken."
    assert strip_quoted("Hi\n\n-----Original Message-----\nFrom: x") == "Hi"


def test_imap_reads_only_recent_unread_mail_and_logout_never_hides_errors(settings, monkeypatch):
    from datetime import date

    from channels import email_channel as ec

    assert ec.imap_since(2, date(2026, 11, 3)) == "01-Nov-2026"

    class FakeIMAP:
        searched = []

        def select(self, *a, **k):
            return "OK", [b"1"]

        def search(self, charset, *criteria):
            FakeIMAP.searched.append(criteria)
            if FakeIMAP.huge:
                raise ec.imaplib.IMAP4.error("got more than 1000000 bytes")
            return "OK", [b""]

        def logout(self):
            raise ec.imaplib.IMAP4.abort("command: LOGOUT => unexpected response: b'169790 169791'")

    s = settings.with_overrides(email_address="desk@gmail.com", email_password="abcdabcdabcdabcd",
                                email_lookback_days=2)
    ch = ec.EmailChannel(s)
    monkeypatch.setattr(ch, "_imap", lambda: (FakeIMAP(), False))
    FakeIMAP.huge = False
    assert ch.fetch_new() == []
    assert FakeIMAP.searched[-1][:2] == ("UNSEEN", "SINCE")
    FakeIMAP.huge = True
    try:
        ch.fetch_new()
        raise AssertionError("expected an error")
    except RuntimeError as exc:
        assert "EMAIL_LOOKBACK_DAYS" in str(exc) and "LOGOUT" not in str(exc)


def test_personal_mailbox_is_safe_by_default(settings):
    """Without a subject filter or EMAIL_MAILBOX_DEDICATED the desk reads only replies to its own tickets."""
    from channels import email_channel as ec

    base = settings.with_overrides(email_address="me@gmail.com", email_password="abcdabcdabcdabcd")
    default = ec.EmailChannel(base).search_criteria()
    assert default[:2] == ["UNSEEN", "SINCE"] and default[3:] == ["SUBJECT", '"TKT-"']
    assert "only replies to desk tickets" in ec.EmailChannel(base).reading_mode()

    filtered = ec.EmailChannel(base.with_overrides(email_subject_filter="Support")).search_criteria()
    assert filtered[3:] == ["OR", "SUBJECT", '"Support"', "SUBJECT", '"TKT-"']

    dedicated = ec.EmailChannel(base.with_overrides(email_mailbox_dedicated=True)).search_criteria()
    assert dedicated[3:] == []
    assert "every unread mail" in ec.EmailChannel(base.with_overrides(email_mailbox_dedicated=True)).reading_mode()
