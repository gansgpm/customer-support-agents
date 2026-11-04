"""Email channel: IMAP in, SMTP out, replies wrapped in an HTML template.

Gmail: turn on 2-step verification, create an App Password and put it in
EMAIL_APP_PASSWORD. IMAP must be enabled in Gmail settings.
"""
from __future__ import annotations

import email
import html
import imaplib
import re
import smtplib
import socket
import ssl
from email.header import decode_header, make_header
from email.message import EmailMessage
from datetime import date, timedelta
from email.utils import formataddr, parseaddr
from pathlib import Path
from string import Template
from typing import Callable, Dict, List, Optional, Tuple

from channels.base import Channel, InboundMessage

TEMPLATE = Path(__file__).parent / "templates" / "email_reply.html"
BADGES = {
    "Resolved": ("#e3f9e5", "#1f7a35", "RESOLVED"),
    "Escalated": ("#ffe3e3", "#b42318", "ESCALATED TO A SPECIALIST"),
    "Approved": ("#e3f2ff", "#1d4ed8", "REVIEWED BY A SPECIALIST"),
    "Info": ("#f1f3f5", "#52606d", "MESSAGE RECEIVED"),
}


def render_email_html(company: str, ticket_id: str, status: str, body: str, complaint: str,
                      reviewed: bool = False) -> str:
    bg, fg, label = BADGES.get(status, BADGES["Resolved"])
    paragraphs = []
    for block in body.strip().split("\n\n"):
        lines = [l for l in block.split("\n") if l.strip()]
        if lines and all(l.strip().startswith(("-", "*", "•")) for l in lines):
            items = "".join(f"<li>{html.escape(l.strip().lstrip('-*• ').strip())}</li>" for l in lines)
            paragraphs.append(f"<ul style='padding-left:20px;margin:0 0 12px 0;'>{items}</ul>")
        else:
            paragraphs.append("<p style='margin:0 0 12px 0;'>" + "<br>".join(html.escape(l) for l in lines) + "</p>")
    short = complaint.strip().replace("\n", " ")
    short = short if len(short) <= 220 else short[:220] + "..."
    return Template(TEMPLATE.read_text(encoding="utf-8")).safe_substitute(
        company=html.escape(company), ticket_id=html.escape(ticket_id), badge_bg=bg, badge_fg=fg,
        status_label=label, body_html="".join(paragraphs), complaint_short=html.escape(short),
        review_note=" and reviewed by a human specialist" if reviewed else "")


def _decode(value: str) -> str:
    try:
        return str(make_header(decode_header(value or "")))
    except Exception:
        return value or ""


def _plain_body(msg: email.message.Message) -> str:
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain" and "attachment" not in str(part.get("Content-Disposition")):
                return part.get_payload(decode=True).decode(part.get_content_charset() or "utf-8", errors="replace")
        for part in msg.walk():
            if part.get_content_type() == "text/html":
                raw = part.get_payload(decode=True).decode(part.get_content_charset() or "utf-8", errors="replace")
                return re.sub(r"<[^>]+>", " ", raw)
        return ""
    payload = msg.get_payload(decode=True) or b""
    return payload.decode(msg.get_content_charset() or "utf-8", errors="replace")


REPLY_SEPARATOR = re.compile(r"^(-{2,}\s*Original Message\s*-{2,}|_{10,}|From:\s.+|Sent from my \w+)", re.I)


def strip_quoted(text: str) -> str:
    """Drop the quoted previous conversation from a reply email.

    Handles "> quoted" lines, Gmail's "On <date>, <name> <address> wrote:" header (which Gmail often wraps over
    two lines) and Outlook's "-----Original Message----- / From:" blocks.
    """
    lines = text.splitlines()
    out = []
    for i, line in enumerate(lines):
        t = line.strip()
        if line.startswith(">") or REPLY_SEPARATOR.match(t):
            break
        if t.startswith("On ") and (t.endswith("wrote:") or
                                    (i + 1 < len(lines) and lines[i + 1].strip().endswith("wrote:"))):
            break
        out.append(line)
    return "\n".join(out).strip()


AUTOMATED_SENDER = re.compile(r"(no-?reply|do-?not-?reply|mailer-daemon|postmaster|notifications?@|newsletter|"
                              r"bounce|alerts?@|updates?@|marketing@)", re.I)


def is_automated(msg: email.message.Message, sender: str) -> bool:
    """Newsletters, receipts, bounces and auto-replies must never become tickets (or get an auto-reply back)."""
    if AUTOMATED_SENDER.search(sender or ""):
        return True
    if msg.get("List-Unsubscribe") or msg.get("List-Id"):
        return True
    auto = (msg.get("Auto-Submitted") or "").lower()
    if auto and auto != "no":
        return True
    return (msg.get("Precedence") or "").lower() in {"bulk", "list", "junk"} or bool(msg.get("X-Autoreply"))


def parse_email(raw: bytes, uid: str = "") -> InboundMessage:
    msg = email.message_from_bytes(raw)
    name, addr = parseaddr(_decode(msg.get("From", "")))
    subject = _decode(msg.get("Subject", ""))
    body = strip_quoted(_plain_body(msg))
    return InboundMessage(channel="email", sender_id=addr, sender_name=name or addr.split("@")[0], subject=subject,
                          text=body.strip(), external_id=msg.get("Message-ID", uid) or uid,
                          meta={"message_id": msg.get("Message-ID", ""), "uid": uid,
                                "automated": is_automated(msg, addr)})


NETWORK_HELP = (
    "Gmail cut the connection before it could answer (tried the normal route and IPv4, on both mail ports), so this "
    "is a network block on this computer, not a password problem. Run: python -m channels.email_diagnose. Usual causes: (1) antivirus email protection (Quick Heal, K7, Avast, AVG, Kaspersky, Norton, McAfee) "
    "scanning outgoing mail - turn off its 'Email protection' / 'Mail shield' or allow python.exe, then try again; "
    "(2) a VPN or proxy - switch it off; (3) office, college or hostel Wi-Fi that blocks mail ports - try a mobile "
    "hotspot. Check in PowerShell: Test-NetConnection smtp.gmail.com -Port 465 and -Port 587 "
    "(TcpTestSucceeded must be True).")
AUTH_HELP = ("Gmail rejected the login. Use a 16-letter App Password created in the SAME Google account as "
             "EMAIL_ADDRESS (myaccount.google.com/apppasswords, 2-Step Verification must be on), type it without "
             "spaces, save .env and restart the app. Your normal Gmail password does not work here.")
IMAP_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def imap_since(days: int, today: Optional[date] = None) -> str:
    """IMAP SINCE date, e.g. 01-Nov-2026 (English month names whatever the computer's language)."""
    d = (today or date.today()) - timedelta(days=max(0, int(days)))
    return f"{d.day:02d}-{IMAP_MONTHS[d.month - 1]}-{d.year}"


def _logout(imap) -> None:
    try:
        imap.logout()
    except Exception:  # never let a failed goodbye hide the real error
        pass


IPV4_ONLY = ("0.0.0.0", 0)  # binding to an IPv4 address makes socket.create_connection skip IPv6 addresses
Route = Tuple[int, bool]  # (port, IPv4 only)
_WORKING_SMTP_ROUTE: Dict[str, Route] = {}  # remembers which route got through on this computer
_WORKING_IMAP_IPV4: Dict[str, bool] = {}


def _issuer(sock) -> str:
    """Organisation that issued the server's TLS certificate (Google Trust Services for the real Gmail)."""
    try:
        cert = sock.getpeercert() or {}
        fields = {k: v for rdn in cert.get("issuer", ()) for k, v in rdn}
        return fields.get("organizationName") or fields.get("commonName") or "?"
    except Exception:
        return "?"


def route_label(route: Route) -> str:
    port, v4 = route
    return f"port {port}" + (" over IPv4" if v4 else "")


class IMAP4SSLv4(imaplib.IMAP4_SSL):
    """IMAP over SSL that connects over IPv4 only (some home connections drop Gmail traffic over IPv6)."""

    def _create_socket(self, timeout):
        sock = socket.create_connection((self.host, self.port), timeout, source_address=IPV4_ONLY)
        return self.ssl_context.wrap_socket(sock, server_hostname=self.host)


def explain_email_error(exc: BaseException, original: str = "") -> str:
    """Turn smtplib / imaplib errors into a fix the user can act on (the original error is kept in brackets)."""
    text = original or f"{type(exc).__name__}: {exc}"
    low = f"{type(exc).__name__}: {exc}".lower()
    if isinstance(exc, smtplib.SMTPAuthenticationError) or "username and password not accepted" in low \
            or "application-specific password" in low or "authenticationfailed" in low or "invalid credentials" in low:
        return f"{AUTH_HELP} ({text})"
    if isinstance(exc, (smtplib.SMTPServerDisconnected, smtplib.SMTPConnectError, ConnectionError, socket.timeout,
                        TimeoutError, ssl.SSLError, imaplib.IMAP4.abort)) or "unexpectedly closed" in low \
            or "timed out" in low or "socket error: eof" in low:
        return f"{NETWORK_HELP} ({text})"
    if "got more than" in low:
        return ("Too many unread emails matched (the mailbox has a very large unread backlog). Lower "
                "EMAIL_LOOKBACK_DAYS, set EMAIL_SUBJECT_FILTER, or use a separate Gmail account for the desk. "
                f"({text[:120]})")
    if isinstance(exc, socket.gaierror):
        return f"This computer cannot look up the Gmail server name - check the internet connection. ({text})"
    return text


class EmailChannel(Channel):
    name = "email"

    @property
    def configured(self) -> bool:
        return bool(self.settings.email_address and self.settings.email_password)

    def fetch_new(self) -> List[InboundMessage]:
        if not self.configured:
            return []
        out: List[InboundMessage] = []
        imap, _ = self._imap()
        try:
            imap.select("INBOX")
            try:
                _, data = imap.search(None, *self.search_criteria())
            except Exception as exc:
                raise RuntimeError(explain_email_error(exc)) from exc
            for uid in (data[0] or b"").split()[-20:]:  # newest 20 per round
                _, msg_data = imap.fetch(uid, "(BODY.PEEK[])")  # PEEK: does not mark the mail as read
                raw = msg_data[0][1] if msg_data and msg_data[0] else b""
                if not raw:
                    continue
                m = parse_email(raw, uid.decode())
                own = m.sender_id.lower() == self.settings.email_address.lower()
                if m.text and not own and not m.meta.get("automated"):
                    out.append(m)
                    imap.store(uid, "+FLAGS", "\\Seen")  # only real customer mails are marked as read
        finally:
            _logout(imap)
        return out

    def search_criteria(self) -> List[str]:
        """Unread mail from the last EMAIL_LOOKBACK_DAYS days only - an old unread backlog is never imported."""
        criteria = ["UNSEEN", "SINCE", imap_since(self.settings.email_lookback_days)]
        if self.settings.email_subject_filter:
            criteria += ["SUBJECT", f'"{self.settings.email_subject_filter}"']
        return criteria

    # ------------------------------------------------------------------- IMAP
    def _imap(self) -> Tuple[imaplib.IMAP4_SSL, bool]:
        """Logged-in IMAP connection; retries over IPv4 when the normal route is dropped. Returns (imap, ipv4)."""
        host = self.settings.email_imap_host
        first_v4 = self.settings.email_force_ipv4 or _WORKING_IMAP_IPV4.get(host, False)
        failures = []
        for v4 in ([True, False] if first_v4 else [False, True]):
            try:
                imap = (IMAP4SSLv4 if v4 else imaplib.IMAP4_SSL)(host, timeout=20)
            except (OSError, imaplib.IMAP4.abort) as exc:
                failures.append(("IPv4" if v4 else "default", exc))
                continue
            try:
                imap.login(self.settings.email_address, self.settings.email_password)
            except (OSError, imaplib.IMAP4.abort) as exc:
                failures.append(("IPv4" if v4 else "default", exc))
                continue
            except Exception as exc:  # Gmail answered: wrong password, IMAP off, ...
                try:
                    imap.shutdown()
                except Exception:
                    pass
                raise RuntimeError(explain_email_error(exc)) from exc
            _WORKING_IMAP_IPV4[host] = v4
            return imap, v4
        tried = "tried IMAP " + "; ".join(f"{r}: {type(e).__name__}: {e}" for r, e in failures)
        raise RuntimeError(explain_email_error(failures[-1][1], original=tried))

    # ------------------------------------------------------------------- SMTP
    def _smtp_routes(self) -> List[Route]:
        configured = int(self.settings.email_smtp_port or 465)
        ports = [configured] + [p for p in (465, 587) if p != configured]
        default, v4 = [(p, False) for p in ports], [(p, True) for p in ports]
        routes = v4 + default if self.settings.email_force_ipv4 else default + v4
        known = _WORKING_SMTP_ROUTE.get(self.settings.email_smtp_host)
        return [known] + [r for r in routes if r != known] if known in routes else routes

    def _smtp(self, action: Optional[Callable[[smtplib.SMTP], None]] = None) -> Route:
        """Log in to SMTP and run `action`; returns the route (port, ipv4) that worked.

        Gmail listens on 465 (SSL) and 587 (STARTTLS), over IPv6 and IPv4. Antivirus mail scanners, VPNs and some
        home connections cut one of these, so when the connection is dropped BEFORE anything is sent we try the
        next route. Login errors are not retried (every route would give the same answer).
        """
        host = self.settings.email_smtp_host
        failures = []
        for port, v4 in self._smtp_routes():
            sending = False
            extra = {"source_address": IPV4_ONLY} if v4 else {}
            try:
                context = ssl.create_default_context()
                if port == 465:
                    smtp = smtplib.SMTP_SSL(host, 465, context=context, timeout=20, **extra)
                else:
                    smtp = smtplib.SMTP(host, port, timeout=20, **extra)
                with smtp:
                    if port != 465:
                        smtp.ehlo()
                        smtp.starttls(context=context)
                        smtp.ehlo()
                    smtp.login(self.settings.email_address, self.settings.email_password)
                    if action:
                        sending = True
                        action(smtp)
                _WORKING_SMTP_ROUTE[host] = (port, v4)
                return port, v4
            except smtplib.SMTPResponseException as exc:  # Gmail answered: login refused, message rejected, ...
                raise RuntimeError(explain_email_error(exc)) from exc
            except OSError as exc:  # connection dropped / refused / timed out / TLS intercepted
                if sending:  # never risk sending the same mail twice
                    raise RuntimeError(explain_email_error(exc)) from exc
                failures.append(((port, v4), exc))
        tried = "tried " + "; ".join(f"{route_label(r)}: {type(e).__name__}: {e}" for r, e in failures)
        raise RuntimeError(explain_email_error(failures[-1][1], original=tried))

    # ------------------------------------------------------------- setup checks
    def check(self) -> dict:
        """Log in to IMAP and SMTP without sending anything."""
        result = {"configured": self.configured, "imap": "not tested", "smtp": "not tested", "unread": None}
        if not self.configured:
            return result
        try:
            imap, v4 = self._imap()
            try:
                imap.select("INBOX", readonly=True)
                _, data = imap.search(None, *self.search_criteria())
                result["unread"] = len((data[0] or b"").split())
            finally:
                _logout(imap)
            result["imap"] = "ok over IPv4" if v4 and not self.settings.email_force_ipv4 else "ok"
        except Exception as exc:
            result["imap"] = f"failed: {exc}"
        try:
            port, v4 = self._smtp()
            configured = int(self.settings.email_smtp_port or 465)
            tips = []
            if port != configured:
                tips.append(f"set EMAIL_SMTP_PORT={port}")
            if v4 and not self.settings.email_force_ipv4:
                tips.append("set EMAIL_FORCE_IPV4=true")
            result["smtp"] = "ok" if not tips else (
                f"ok on {route_label((port, v4))} - the usual route is cut on this network; the app switches "
                f"automatically, or {' and '.join(tips)} in .env to go straight there")
        except Exception as exc:
            result["smtp"] = f"failed: {exc}"
        return result

    def diagnose(self) -> List[Dict[str, str]]:
        """Try every route to Gmail separately (no mail is sent) - shows exactly where the connection breaks.

        For SMTP it records the step that failed (connect / greeting / ehlo / starttls / login), Gmail's greeting
        and who issued the TLS certificate: anything other than Google Trust Services means a program on this
        computer (usually antivirus mail scanning) is sitting in the middle of the connection.
        """
        rows = []
        host = self.settings.email_smtp_host
        for port in (465, 587):
            for v4 in (False, True):
                extra = {"source_address": IPV4_ONLY} if v4 else {}
                row = {"service": f"SMTP {port}", "route": "IPv4" if v4 else "default (IPv6 first)"}
                step = "connect + greeting"
                try:
                    context = ssl.create_default_context()
                    smtp = (smtplib.SMTP_SSL(host, 465, context=context, timeout=15, **extra) if port == 465
                            else smtplib.SMTP(host, port, timeout=15, **extra))
                    with smtp:
                        row["server_ip"] = str(smtp.sock.getpeername()[0])
                        step = "ehlo"
                        smtp.ehlo()
                        if port != 465:
                            step = "starttls"
                            smtp.starttls(context=context)
                            smtp.ehlo()
                        row["tls_issuer"] = _issuer(smtp.sock)
                        step = "login"
                        smtp.login(self.settings.email_address, self.settings.email_password)
                    row["result"] = "OK - login works"
                except Exception as exc:
                    row["result"] = f"failed at {step}: {type(exc).__name__}: {exc}"[:170]
                rows.append(row)
        for v4 in (False, True):
            row = {"service": "IMAP 993", "route": "IPv4" if v4 else "default (IPv6 first)"}
            try:
                imap = (IMAP4SSLv4 if v4 else imaplib.IMAP4_SSL)(self.settings.email_imap_host, timeout=15)
                row["server_ip"] = str(imap.sock.getpeername()[0])
                row["tls_issuer"] = _issuer(imap.sock)
                imap.login(self.settings.email_address, self.settings.email_password)
                imap.logout()
                row["result"] = "OK - login works"
            except Exception as exc:
                row["result"] = f"{type(exc).__name__}: {exc}"[:170]
            rows.append(row)
        return rows

    def _send(self, to: str, text: str, subject: str = "", html_body: str = "") -> str:
        msg = EmailMessage()
        msg["From"] = formataddr((self.settings.support_signature, self.settings.email_address))
        msg["To"] = to
        msg["Subject"] = subject or f"{self.settings.company_name} Support"
        msg.set_content(text)
        if html_body:
            msg.add_alternative(html_body, subtype="html")
        route = self._smtp(lambda smtp: smtp.send_message(msg))
        return f"smtp ok ({route_label(route)}) -> {to}"
