"""HTTPS entry point for phones and other devices on the local network (Windows side, stdlib only).

WSL2 is only reachable from this PC, and browsers allow the microphone only on https://, so this listens on
0.0.0.0:8443 with a self-signed certificate and forwards each connection to the server's LAN port
(127.0.0.1:8001, which Windows forwards into WSL). The server checks the access key (see lan.py).

    python windows\\lan_proxy.py [--port 8443] [--target 127.0.0.1:8001]

The certificate (certs/lan.crt, valid for this PC's LAN addresses) is created on first run with openssl
(on PATH, from Git for Windows, or inside WSL). Phones show a one-time "not private / not trusted" warning
for it: choose Advanced -> Proceed.
"""
import argparse
import asyncio
import json
import os
import shutil
import socket
import ssl
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CERTS = ROOT / "certs"
INFO = ROOT / ".lan_info.json"


def lan_ips():
    """Private IPv4 addresses of this PC, the default-route one first (no packets are sent)."""
    ips = []
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        ips.append(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
        ips.append(info[4][0])
    out = []
    for i, ip in enumerate(ips):
        a, b = (int(x) for x in ip.split(".")[:2])
        private = a == 10 or (a == 192 and b == 168) or (a == 172 and 16 <= b < 32)
        # Beyond the default-route address, skip the virtual adapters of WSL/Hyper-V (172.16-31) and VirtualBox
        virtual = a == 172 or ip.startswith("192.168.56.")
        if private and (i == 0 or not virtual) and ip not in out:
            out.append(ip)
    return out


def _openssl():
    for cand in (shutil.which("openssl"), r"C:\Program Files\Git\usr\bin\openssl.exe",
                 r"C:\Program Files\Git\mingw64\bin\openssl.exe"):
        if cand and Path(cand).exists():
            return [cand]
    if shutil.which("wsl"):
        return ["wsl", "openssl"]
    return None


def ensure_cert(ips):
    crt, key, meta = CERTS / "lan.crt", CERTS / "lan.key", CERTS / "lan.json"
    names = sorted(set(ips) | {"127.0.0.1"})
    try:
        if crt.exists() and key.exists() and json.loads(meta.read_text()).get("ips") == names:
            return crt, key
    except (FileNotFoundError, ValueError):
        pass
    CERTS.mkdir(exist_ok=True)
    cmd = _openssl()
    if cmd is None:
        sys.exit("openssl not found (install Git for Windows, or openssl in WSL)")
    host = socket.gethostname()
    san = ",".join([f"IP:{ip}" for ip in names] + [f"DNS:{host}", "DNS:localhost"])
    wsl = cmd[0] == "wsl"
    path = (lambda p: subprocess.check_output(["wsl", "wslpath", "-a", str(p)], text=True).strip()) if wsl else str
    tmp_key, tmp_crt = CERTS / "lan.key.tmp", CERTS / "lan.crt.tmp"
    subprocess.run(cmd + ["req", "-x509", "-newkey", "ec", "-pkeyopt", "ec_paramgen_curve:prime256v1", "-nodes",
                          "-days", "3650", "-subj", f"/CN=Ming Voice ({host})", "-addext", f"subjectAltName={san}",
                          "-keyout", path(tmp_key), "-out", path(tmp_crt)],
                   check=True, capture_output=True)
    os.replace(tmp_key, key)
    os.replace(tmp_crt, crt)
    meta.write_text(json.dumps({"ips": names}))
    print(f"created a certificate for {', '.join(names)}")
    return crt, key


async def pipe(reader, writer):
    try:
        while data := await reader.read(65536):
            writer.write(data)
            await writer.drain()
    except (ConnectionError, OSError, ssl.SSLError):
        pass
    finally:
        try:
            writer.close()
        except Exception:
            pass


def nodelay(writer):
    sock = writer.get_extra_info("socket")
    if sock is not None:
        try:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)  # audio frames: don't batch small writes
        except OSError:
            pass


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=int(os.environ.get("MING_PHONE_PORT", "8443")))
    ap.add_argument("--target", default=os.environ.get("MING_LAN_TARGET", "127.0.0.1:8001"))
    args = ap.parse_args()
    th, tp = args.target.rsplit(":", 1)
    ips = lan_ips()
    if not ips:
        sys.exit("no LAN address found")
    crt, key = ensure_cert(ips)
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    ctx.load_cert_chain(crt, key)

    async def handle(reader, writer):
        nodelay(writer)
        try:
            ur, uw = await asyncio.wait_for(asyncio.open_connection(th, int(tp)), 5)
        except (OSError, asyncio.TimeoutError):
            msg = (b"HTTP/1.1 503 Service Unavailable\r\nContent-Type: text/plain\r\nConnection: close\r\n\r\n"
                   b"Ming Voice is not running (start it on the PC first).\n")
            writer.write(msg)
            await writer.drain()
            writer.close()
            return
        nodelay(uw)
        await asyncio.gather(pipe(reader, uw), pipe(ur, writer))

    server = await asyncio.start_server(handle, "0.0.0.0", args.port, ssl=ctx, ssl_handshake_timeout=10)

    async def heartbeat():
        while True:
            try:
                INFO.write_text(json.dumps({"ips": ips, "port": args.port, "pid": os.getpid(), "heartbeat": time.time()}))
            except OSError:
                pass
            await asyncio.sleep(10)

    asyncio.create_task(heartbeat())
    print(f"Phone access: https://{ips[0]}:{args.port}  (open the QR code on http://localhost:8000 -> Phone)")
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
