"""Find out exactly where the connection to Gmail breaks on this computer.

    python -m channels.email_diagnose

Tries SMTP 465, SMTP 587 and IMAP 993, each over the default route (IPv6 first) and over IPv4 only, and logs in
on each - no email is sent and the password is never printed. Reads EMAIL_ADDRESS / EMAIL_APP_PASSWORD from .env.
"""
from __future__ import annotations

from channels.email_channel import EmailChannel
from engine.config import get_settings


def main() -> None:
    settings = get_settings()
    ch = EmailChannel(settings)
    if not ch.configured:
        print("EMAIL_ADDRESS / EMAIL_APP_PASSWORD are not set in .env - fill them in and save the file first.")
        return
    n = len(settings.email_password)
    print(f"Account: {settings.email_address}  (app password: {n} characters)")
    if n != 16:
        print("   !! A Gmail App Password has exactly 16 letters - this is not one. Check EMAIL_APP_PASSWORD in .env.")
    print()
    rows = ch.diagnose()
    for r in rows:
        print(f"{r['service']:<9} {r['route']:<21} {r.get('server_ip', '-'):<26} "
              f"{('TLS: ' + r['tls_issuer']) if r.get('tls_issuer') else '':<32} {r['result']}")
    issuers = {r["tls_issuer"] for r in rows if r.get("tls_issuer") and r["tls_issuer"] != "?"}
    foreign = sorted(i for i in issuers if "google" not in i.lower())
    ok = [r for r in rows if r["result"].startswith("OK")]
    smtp_ok = [r for r in ok if r["service"].startswith("SMTP")]
    print()
    if foreign:
        print(f"-> The secure connection is being opened by '{', '.join(foreign)}', not by Google. That program "
              "(usually antivirus mail / SSL scanning) is in the middle and cuts the mail. Turn off its email or "
              "SSL scanning, or add python.exe to its exceptions, then run this again.")
    elif any("AuthenticationError" in r["result"] or "AUTHENTICATIONFAILED" in r["result"] for r in rows):
        print("-> Gmail refused the login: create a new App Password in the SAME account as EMAIL_ADDRESS, "
              "paste it into .env, save and restart.")
    elif smtp_ok and all(r["route"] == "IPv4" for r in smtp_ok):
        port = smtp_ok[0]["service"].split()[1]
        print(f"-> Only IPv4 gets through on this network. Put these in .env, save and restart the app:\n"
              f"   EMAIL_FORCE_IPV4=true\n   EMAIL_SMTP_PORT={port}")
    elif smtp_ok:
        port = smtp_ok[0]["service"].split()[1]
        print(f"-> Email works. If the app still fails, set EMAIL_SMTP_PORT={port} in .env and restart the app.")
    else:
        print("-> Every route is cut. Turn off the antivirus 'Email protection' / 'Mail shield' (or allow python.exe), "
              "switch off any VPN, or try a mobile hotspot - then run this again.")


if __name__ == "__main__":
    main()
