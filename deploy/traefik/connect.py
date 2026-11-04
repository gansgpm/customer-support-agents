#!/usr/bin/env python3
"""Publish the support desk through the Traefik that already runs on the VPS (it owns ports 80 and 443).

    cd /opt/customer-support-agents
    python3 deploy/traefik/connect.py support.gansgpm.tech

The script:
  1. finds Traefik (a Docker container or a plain process) and reads its settings: the HTTPS entry point, the
     Let's Encrypt certificate resolver, the Docker network, and whether HTTP is already redirected to HTTPS;
  2. asks for a username and password for the site (stored only as a hash);
  3. writes docker-compose.override.yml with the Traefik labels for the app (git-ignored, server only);
  4. restarts the app and checks https://<domain> (expects 401 = password prompt = working).

It changes nothing in Traefik itself, so other sites on the server keep working. Safe to run again.
"""
from __future__ import annotations

import getpass
import json
import re
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
OVERRIDE = ROOT / "docker-compose.override.yml"
NAME = "support-desk"  # router / middleware / service names inside Traefik


def sh(*cmd: str, stdin: Optional[str] = None) -> Tuple[int, str]:
    try:
        p = subprocess.run(cmd, input=stdin, capture_output=True, text=True, timeout=120)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 1, str(exc)


# ------------------------------------------------------------------ reading the Traefik configuration
def flatten(obj, prefix: str = "") -> Dict[str, str]:
    """{'entryPoints': {'websecure': {'address': ':443'}}} -> {'entryPoints.websecure.address': ':443'}"""
    out: Dict[str, str] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, (dict, list)) and v:
                out.update(flatten(v, key))
            else:
                out[key] = "true" if v in (None, {}, []) else str(v)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.update(flatten(v, f"{prefix}.{i}"))
    return out


def parse_config_text(text: str) -> Dict[str, str]:
    try:
        import yaml  # present on Ubuntu (cloud-init)

        data = yaml.safe_load(text)
        if isinstance(data, dict):
            return flatten(data)
    except Exception:
        pass
    try:
        import tomllib

        return flatten(tomllib.loads(text))
    except Exception:
        return {}


def parse_args(args: List[str]) -> Dict[str, str]:
    flags: Dict[str, str] = {}
    for a in args:
        if not a.startswith("--"):
            continue
        key, _, value = a[2:].partition("=")
        flags[key] = value if _ else "true"
    return flags


def analyse(flags: Dict[str, str]) -> Dict:
    """Pick the settings the labels need out of Traefik's flattened configuration."""
    def find(pattern: str) -> List[Tuple[str, str]]:
        rx = re.compile(pattern, re.I)
        return [(m.group(1), v) for k, v in flags.items() if (m := rx.fullmatch(k))]

    entrypoints = {name: addr for name, addr in find(r"entrypoints\.([^.]+)\.address")}
    https_ep = next((n for n, a in entrypoints.items() if a.rstrip().endswith(":443")), None)
    http_ep = next((n for n, a in entrypoints.items() if a.rstrip().endswith(":80")), None)
    resolvers = list(dict.fromkeys(n for n, _ in find(r"certificatesresolvers\.([^.]+)\..+")))
    docker_keys = {k.lower(): v for k, v in flags.items() if re.match(r"providers\.docker(\.|$)", k, re.I)}
    docker = bool(docker_keys) and docker_keys.get("providers.docker", "true").lower() != "false"
    redirect = bool(http_ep) and any(re.match(rf"entrypoints\.{re.escape(http_ep)}\.http\.redirections", k, re.I)
                                     for k in flags)
    network = next((v for k, v in flags.items() if k.lower() == "providers.docker.network"), "")
    return {"https_entrypoint": https_ep, "http_entrypoint": http_ep, "resolvers": resolvers,
            "docker_provider": docker, "redirects_http": redirect, "docker_network": network}


def find_traefik() -> Optional[Dict]:
    """Traefik as a Docker container (preferred) or as a plain process; returns its flags and network."""
    code, out = sh("docker", "ps", "--format", "{{.ID}} {{.Names}} {{.Image}}")
    for line in out.splitlines() if code == 0 else []:
        cid, name, image = (line.split() + ["", "", ""])[:3]
        if "traefik" not in (name + image).lower():
            continue
        code, raw = sh("docker", "inspect", cid)
        if code != 0:
            continue
        info = json.loads(raw)[0]
        args = list(info.get("Args") or [])
        flags = parse_args(args)
        mounts = {m.get("Destination", ""): m.get("Source", "") for m in info.get("Mounts") or []}
        cfg = flags.get("configFile") or flags.get("configfile") or ""
        candidates = [cfg] if cfg else ["/etc/traefik/traefik.yml", "/etc/traefik/traefik.yaml",
                                         "/etc/traefik/traefik.toml", "/traefik.yml", "/traefik.yaml", "/traefik.toml"]
        for path in candidates:
            host = mounts.get(path) or next((src + path[len(dst):] for dst, src in mounts.items()
                                             if dst and path.startswith(dst.rstrip("/") + "/")), "")
            if host and Path(host).is_file():
                flags = {**parse_config_text(Path(host).read_text(encoding="utf-8", errors="ignore")), **flags}
                break
            if not host:  # config baked into the image: read it from the container
                code, text = sh("docker", "exec", cid, "cat", path)
                if code == 0 and text.strip():
                    flags = {**parse_config_text(text), **flags}
                    break
        mode = info.get("HostConfig", {}).get("NetworkMode", "")
        networks = list((info.get("NetworkSettings", {}).get("Networks") or {}).keys())
        return {"where": f"container {name} ({image})", "flags": flags, "network_mode": mode, "networks": networks}
    code, out = sh("pgrep", "-o", "traefik")
    if code == 0 and out.strip():
        pid = out.split()[0]
        args = Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="ignore").split("\0")
        flags = parse_args(args)
        cfg = flags.get("configFile") or flags.get("configfile") or next(
            (p for p in ("/etc/traefik/traefik.yml", "/etc/traefik/traefik.yaml", "/etc/traefik/traefik.toml")
             if Path(p).is_file()), "")
        if cfg and Path(cfg).is_file():
            flags = {**parse_config_text(Path(cfg).read_text(encoding="utf-8", errors="ignore")), **flags}
        return {"where": f"process {pid}", "flags": flags, "network_mode": "host", "networks": []}
    return None


def pick_network(t: Dict, wanted: str) -> str:
    """'' when Traefik runs on the host network (it reaches containers directly), else the network to join."""
    if t["network_mode"] == "host":
        return ""
    if wanted:
        return wanted
    usable = [n for n in t["networks"] if n not in ("bridge", "host", "none")]
    return usable[0] if usable else ""


# ------------------------------------------------------------------ writing the override
def password_hash(user: str, password: str) -> str:
    if shutil.which("htpasswd"):
        code, out = sh("htpasswd", "-niB", user, stdin=password + "\n")
        if code == 0 and out.startswith(user + ":"):
            return out.strip().splitlines()[0]
    code, out = sh("openssl", "passwd", "-apr1", "-stdin", stdin=password + "\n")
    if code == 0 and out.strip().startswith("$apr1$"):
        return f"{user}:{out.strip().splitlines()[0]}"
    sys.exit("Could not hash the password: install apache2-utils (apt -y install apache2-utils) and run again.")


def override_yaml(domain: str, cfg: Dict, resolver: str, network: str, users: str) -> str:
    users = users.replace("$", "$$")  # docker compose treats $ as a variable
    labels = [
        "traefik.enable=true",
        f"traefik.http.routers.{NAME}.rule=Host(`{domain}`)",
        f"traefik.http.routers.{NAME}.entrypoints={cfg['https_entrypoint']}",
        f"traefik.http.routers.{NAME}.tls=true",
        f"traefik.http.routers.{NAME}.tls.certresolver={resolver}",
        f"traefik.http.routers.{NAME}.middlewares={NAME}-auth",
        f"traefik.http.routers.{NAME}.service={NAME}",
        f"traefik.http.middlewares.{NAME}-auth.basicauth.users={users}",
        f"traefik.http.services.{NAME}.loadbalancer.server.port=8501",
    ]
    if cfg["http_entrypoint"] and not cfg["redirects_http"]:
        labels += [f"traefik.http.routers.{NAME}-http.rule=Host(`{domain}`)",
                   f"traefik.http.routers.{NAME}-http.entrypoints={cfg['http_entrypoint']}",
                   f"traefik.http.routers.{NAME}-http.middlewares={NAME}-https",
                   f"traefik.http.routers.{NAME}-http.service={NAME}",
                   f"traefik.http.middlewares.{NAME}-https.redirectscheme.scheme=https"]
    if network:
        labels.append(f"traefik.docker.network={network}")
    lines = ["# Written by deploy/traefik/connect.py - routes https://" + domain + " through the server's Traefik.",
             "# Server-only file (git-ignored): it holds the hashed site password.", "services:", "  app:",
             "    labels:"] + [f'      - "{label}"' for label in labels]
    if network:
        lines += ["    networks: [default, traefik_net]", "networks:", "  traefik_net:", "    external: true",
                  f"    name: {network}"]
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ checks
def public_ip() -> str:
    for url in ("https://api.ipify.org", "https://ifconfig.me"):
        code, out = sh("curl", "-4", "-s", "--max-time", "8", url)
        if code == 0 and re.fullmatch(r"\d+\.\d+\.\d+\.\d+", out.strip()):
            return out.strip()
    return ""


def wait_for_https(domain: str, seconds: int = 120) -> str:
    deadline = time.time() + seconds
    last = ""
    while time.time() < deadline:
        code, out = sh("curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", "--max-time", "10", f"https://{domain}/")
        last = out.strip()
        if last == "401":
            return "ok"
        time.sleep(5)
    code, out = sh("curl", "-sk", "-o", "/dev/null", "-w", "%{http_code}", "--max-time", "10",
                   "--resolve", f"{domain}:443:127.0.0.1", f"https://{domain}/")
    return "cert-pending" if out.strip() == "401" else f"http {last or 'no answer'}"


def main() -> None:
    domain = (sys.argv[1] if len(sys.argv) > 1 else "support.gansgpm.tech").strip().lower()
    print(f"Publishing https://{domain}\n")

    code, out = sh("docker", "compose", "ps", "--status", "running", "--services")
    if "app" not in out.split():
        sys.exit("The app is not running yet. First run:  docker compose up -d --build")

    t = find_traefik()
    if not t:
        sys.exit("Traefik was not found. Send Claude the output of:  ss -tlnp | grep -E ':(80|443)\\s'")
    cfg = analyse(t["flags"])
    network = pick_network(t, cfg["docker_network"])
    print(f"Traefik: {t['where']}, network mode '{t['network_mode'] or '-'}'")
    print(f"  HTTPS entry point: {cfg['https_entrypoint']}  HTTP entry point: {cfg['http_entrypoint']}  "
          f"HTTP->HTTPS redirect: {'yes' if cfg['redirects_http'] else 'no'}")
    print(f"  certificate resolvers: {', '.join(cfg['resolvers']) or 'none'}  Docker labels enabled: "
          f"{'yes' if cfg['docker_provider'] else 'no'}  network to join: {network or '(not needed)'}\n")

    problems = []
    if not cfg["docker_provider"]:
        problems.append("Traefik does not read Docker labels (no Docker provider)")
    if not cfg["https_entrypoint"]:
        problems.append("no Traefik entry point on port 443 was found")
    if not cfg["resolvers"]:
        problems.append("Traefik has no Let's Encrypt certificate resolver")
    if t["network_mode"] != "host" and not network:
        problems.append(f"Traefik is only on the default bridge network ({t['networks']})")
    if problems:
        print("Cannot connect automatically:\n  - " + "\n  - ".join(problems))
        sys.exit("Send Claude this output - the settings above are what is needed.")

    ip, resolved = public_ip(), ""
    try:
        resolved = socket.gethostbyname(domain)
    except OSError:
        pass
    if not resolved:
        sys.exit(f"{domain} does not resolve yet. Check the A record in Hostinger DNS, wait 5-30 min, run again.")
    if ip and resolved != ip:
        sys.exit(f"{domain} points to {resolved}, but this server is {ip}. Fix the A record value, then run again.")
    print(f"DNS OK: {domain} -> {resolved}\n")

    user = input("Username for the site [judge]: ").strip() or "judge"
    while True:
        pw = getpass.getpass("Password for the site (min 8 characters, not shown): ")
        if len(pw) >= 8 and pw == getpass.getpass("Repeat the password: "):
            break
        print("Passwords differ or are shorter than 8 characters - try again.")
    resolver = cfg["resolvers"][0]
    OVERRIDE.write_text(override_yaml(domain, cfg, resolver, network, password_hash(user, pw)), encoding="utf-8")
    OVERRIDE.chmod(0o600)
    print(f"\nWrote {OVERRIDE.name} (certificate resolver '{resolver}'). Restarting the app...")

    code, out = sh("docker", "compose", "up", "-d")
    if code != 0:
        print(out)
        sys.exit("docker compose failed - send Claude the lines above.")
    print("Waiting for Traefik and the certificate (up to 2 minutes)...")
    result = wait_for_https(domain)
    if result == "ok":
        print(f"\nDONE: https://{domain} is live (HTTPS + password). Username: {user}")
    elif result == "cert-pending":
        print(f"\nThe route works, but the HTTPS certificate is not issued yet. Wait a few minutes and open "
              f"https://{domain}. If it still warns, send Claude:  docker logs --tail 50 "
              f"{t['where'].split()[1] if t['where'].startswith('container') else 'traefik'} 2>&1 | grep -i acme")
    else:
        print(f"\nNot reachable yet ({result}). Send Claude:  docker compose ps  and  docker compose logs --tail 30 app")


if __name__ == "__main__":
    main()
