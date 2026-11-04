# Deploy on a VPS with your own domain

The desk runs as two Docker containers on a Linux VPS: the Streamlit UI and the channel worker, which answers
Email and Telegram around the clock. Nginx in front of them provides HTTPS (a free Let's Encrypt certificate) and a
password. The example uses **support.gansgpm.tech** on a Hostinger KVM VPS with Ubuntu 22.04 / 24.04. Any VPS with
at least 2 GB RAM works the same way.

> Shared web hosting cannot run Python apps or Docker. You need a VPS: a domain alone only points to a server.

```
browser ──HTTPS──► Nginx :443 (password, certificate) ──► 127.0.0.1:8501  Streamlit UI   (container "app")
                                                                           channel worker (container "worker")
```

## 1. Point the domain to the VPS

1. In hPanel → **VPS**, select your server and copy its **IPv4 address**.
2. In hPanel → **Domains → gansgpm.tech → DNS / Nameservers → DNS records**, add a record:
   **Type** `A` · **Name** `support` · **Points to** the VPS IPv4 · **TTL** `300`.
3. Check it, usually within 5-30 minutes: `nslookup support.gansgpm.tech` should return the VPS IP.

## 2. Prepare the server (once)

Open the VPS terminal: hPanel → VPS → **Browser terminal**, or `ssh root@<VPS-IP>`.

```bash
apt update && apt -y upgrade
docker --version || curl -fsSL https://get.docker.com | sh          # Docker + compose plugin
apt -y install nginx certbot python3-certbot-nginx apache2-utils git
ss -tlnp | grep -E ':(80|443)\s'     # must show only nginx (or nothing) - see "Troubleshooting" otherwise
```

Firewall: allow SSH **first**, then the web ports.

```bash
ufw allow OpenSSH && ufw allow 80/tcp && ufw allow 443/tcp && ufw --force enable
```

The containers publish their ports on `127.0.0.1` only (see `docker-compose.yml`), so Streamlit can never be reached
without going through Nginx. Docker bypasses ufw for ports published on all interfaces.

## 3. Get the code and the settings

```bash
git clone https://github.com/<your-username>/customer-support-agents.git /opt/customer-support-agents
cd /opt/customer-support-agents
cp .env.example .env
nano .env            # OPENAI_API_KEY, COMPANY_NAME, EMAIL_*, TELEGRAM_*, SUPPORT_* ... then Ctrl+O, Enter, Ctrl+X
chmod 600 .env
```

`.env` exists only on the server. It is never in Git, and `.dockerignore` keeps it out of the image.

**Run one channel worker only.** If `channel_worker.py` also runs on your PC, stop it. Two workers would both
answer the same mailbox and Telegram bot.

## 4. Start the desk

```bash
docker compose up -d --build
docker compose ps                  # app and worker "running", app becomes "healthy" after ~30 s
curl -sI http://127.0.0.1:8501 | head -1   # HTTP/1.1 200 OK
docker compose logs -f --tail=50   # Ctrl+C to leave
```

## 5a. If Traefik already owns ports 80/443 (Hostinger templates, n8n, Coolify, Dokploy, ...)

If the `ss` check in step 2 shows `traefik`, skip Nginx: do not install it and do not stop Traefik, which serves your
other sites. Let the existing Traefik route the domain to the desk instead. It also gets the free HTTPS certificate
and asks for the password:

```bash
cd /opt/customer-support-agents
python3 deploy/traefik/connect.py support.gansgpm.tech
```

The script:
- reads Traefik's settings (HTTPS entry point, certificate resolver, Docker network);
- checks that the DNS record points to this server;
- asks for a username and password, and writes `docker-compose.override.yml` (git-ignored) with the Traefik labels;
- restarts the app and waits until `https://support.gansgpm.tech` answers with the password prompt.

Traefik itself is not changed, and the script can be run again safely, for example to change the password.

## 5. Nginx, password and HTTPS (when nothing else uses ports 80/443)

```bash
htpasswd -c /etc/nginx/.htpasswd-support judge        # pick the username, then type a password twice
cp deploy/nginx/support.gansgpm.tech.conf /etc/nginx/sites-available/
ln -s /etc/nginx/sites-available/support.gansgpm.tech.conf /etc/nginx/sites-enabled/
nginx -t && systemctl reload nginx
certbot --nginx -d support.gansgpm.tech --redirect --agree-tos -m you@example.com
```

Open **https://support.gansgpm.tech**, sign in with the username and password, and submit a test complaint.
Certbot renews the certificate automatically (`systemctl list-timers | grep certbot`).

Without the password the app is public. Every visitor's ticket then costs OpenAI tokens, and the escalation queue
can send real emails. Share the username and password only with the evaluators. Also set a monthly budget under
platform.openai.com → **Settings → Limits**.

## Update after a `git push`

```bash
cd /opt/customer-support-agents && git pull && docker compose up -d --build
```

Tickets, the learned solutions and the index live in `data/`, `logs/` and `knowledge_base/learned/` on the server.
They survive updates and restarts.

## Optional: WhatsApp webhooks

Twilio has to call a public HTTPS URL. To enable it:

1. Run `docker compose --profile webhooks up -d`.
2. Uncomment the `/webhooks/` block in `/etc/nginx/sites-available/support.gansgpm.tech.conf`.
3. Run `nginx -t && systemctl reload nginx`.
4. In the Twilio sandbox, set "When a message comes in" to `https://support.gansgpm.tech/webhooks/whatsapp`.

Email and Telegram need no webhook, because the worker polls them.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Page keeps loading / "Connection error" banner | the websocket is blocked: the `Upgrade` / `Connection` lines must be in the Nginx `location /` block; `nginx -t && systemctl reload nginx` |
| `502 Bad Gateway` | the app container is not up: `docker compose ps`, `docker compose logs app` |
| Certbot: "DNS problem" / timeout | the A record is not live yet (`nslookup support.gansgpm.tech`) or port 80 is closed (ufw, or the Hostinger VPS firewall in hPanel) |
| Port 80/443 already used by Traefik | use step 5a (`deploy/traefik/connect.py`) instead of Nginx; for another proxy (Caddy, ...) add a route for the domain → `127.0.0.1:8501` there |
| Telegram: "Conflict: terminated by other getUpdates request" | another worker (your PC) polls the same bot - stop it |
| OpenAI 401 | wrong / old key in the server's `.env`; after editing: `docker compose up -d` |
