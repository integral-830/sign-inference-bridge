"""Tiny sentence server for the ESP (standard library only, nothing to install).

  python server.py                     # listens on 0.0.0.0:8000  (reachable from your Mac AND from the ESP on the same Wi-Fi)
  python server.py --port 8000 --token mysecret

live_demo.py pushes every finished sentence here (--push http://localhost:8000); the ESP reads it with plain HTTP.

Endpoints
  GET  /health                 -> ok
  GET  /latest                 -> last sentence as plain text (English if available, else the glosses)
  GET  /latest.json            -> {"id":3,"time":"...","gloss":"ME SCHOOL GO","english":"I am going to school.","text":"..."}
  GET  /next?after=3&timeout=25 -> waits (long poll) until a sentence newer than id 3 exists, returns it as JSON;
                                  204 No Content if nothing new came within the timeout
  GET  /history?n=20           -> last n sentences as JSON
  GET  /esp?after=-1&timeout=20 -> for ESP32 displays: key=value lines (seq, live, id, text, gloss, time), long poll,
                                  204 when nothing changed. Start with after=-1, then send back the seq you received.
  GET  /live                   -> the sentence being signed RIGHT NOW (words so far, not finished), plain text; empty when idle
  GET  /live.json              -> {"seq":12,"text":"ME SCHOOL","time":"..."}
  GET  /live/next?after=12&timeout=20 -> long poll: returns when the live text changed (seq > 12), 204 on timeout
  POST /live                   -> body {"text": "ME SCHOOL"}  (live_demo does this; it clears automatically when a sentence ends)
  POST /sentence               -> body JSON {"gloss": "...", "english": "..."}  (live_demo does this for you)
Test by hand:  curl -X POST localhost:8000/sentence -H 'Content-Type: application/json' -d '{"gloss":"ME GO","english":"I go."}'

--token: if set, POST needs the header  X-Token: <token>  (GET stays open). Meant for a home network only:
do not expose this port to the internet.
"""
import argparse
import json
import socket
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse


class Store:
    def __init__(self, keep=200):
        self.items = deque(maxlen=keep)
        self.next_id = 1
        self.cond = threading.Condition()
        self.live = {"seq": 0, "text": "", "time": ""}   # sentence being signed right now (not finished yet)

    def set_live(self, text):
        with self.cond:
            if text == self.live["text"]:
                return self.live
            self.live = {"seq": self.live["seq"] + 1, "text": text, "time": time.strftime("%Y-%m-%d %H:%M:%S")}
            self.cond.notify_all()
            return self.live

    def get_live(self):
        with self.cond:
            return dict(self.live)

    def wait_live_after(self, after, timeout):
        end = time.time() + timeout
        with self.cond:
            while self.live["seq"] <= after:
                left = end - time.time()
                if left <= 0:
                    return None
                self.cond.wait(left)
            return dict(self.live)

    def add(self, gloss, english):
        with self.cond:
            item = {"id": self.next_id, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "gloss": gloss,
                    "english": english, "text": english or gloss}
            self.next_id += 1
            self.items.append(item)
            self.live = {"seq": self.live["seq"] + 1, "text": "", "time": item["time"]}  # sentence done: live line empties
            self.cond.notify_all()
            return item

    def state(self):
        """Everything a small display needs, in one go (live text + last finished sentence)."""
        with self.cond:
            it = self.items[-1] if self.items else {"id": 0, "text": "", "gloss": "", "time": ""}
            return {"seq": self.live["seq"], "live": self.live["text"], "id": it["id"], "text": it["text"],
                    "gloss": it["gloss"], "time": it["time"]}

    def wait_state(self, after, timeout):
        """Long poll: returns as soon as the state differs from `after` (also when the server restarted and its
        counter went back to 0, so a display never gets stuck). None on timeout."""
        end = time.time() + timeout
        with self.cond:
            while self.live["seq"] == after:
                left = end - time.time()
                if left <= 0:
                    return None
                self.cond.wait(left)
        return self.state()

    def latest(self):
        with self.cond:
            return self.items[-1] if self.items else None

    def wait_after(self, after, timeout):
        end = time.time() + timeout
        with self.cond:
            while True:
                for it in self.items:
                    if it["id"] > after:
                        return it
                left = end - time.time()
                if left <= 0:
                    return None
                self.cond.wait(left)

    def history(self, n):
        with self.cond:
            return list(self.items)[-n:]


def make_handler(store, token):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *a):
            print(f"[{time.strftime('%H:%M:%S')}] {self.address_string()} {fmt % a}", flush=True)

        def _send(self, code, body=b"", ctype="text/plain; charset=utf-8"):
            if isinstance(body, str):
                body = body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _json(self, obj, code=200):
            self._send(code, json.dumps(obj, ensure_ascii=False), "application/json; charset=utf-8")

        def do_GET(self):
            u = urlparse(self.path)
            q = parse_qs(u.query)
            if u.path == "/health":
                return self._send(200, "ok\n")
            if u.path == "/latest":
                it = store.latest()
                return self._send(200, (it["text"] if it else "") + "\n")
            if u.path == "/latest.json":
                return self._json(store.latest() or {"id": 0, "text": ""})
            if u.path == "/next":
                try:
                    after = int(q.get("after", ["0"])[0])
                    timeout = min(max(float(q.get("timeout", ["25"])[0]), 0.0), 55.0)
                except ValueError:
                    return self._json({"error": "after must be an integer, timeout a number"}, 400)
                it = store.wait_after(after, timeout)
                return self._json(it) if it else self._send(204)
            if u.path == "/live":
                return self._send(200, store.get_live()["text"] + "\n")
            if u.path == "/live.json":
                return self._json(store.get_live())
            if u.path == "/live/next":
                try:
                    after = int(q.get("after", ["0"])[0])
                    timeout = min(max(float(q.get("timeout", ["25"])[0]), 0.0), 55.0)
                except ValueError:
                    return self._json({"error": "after must be an integer, timeout a number"}, 400)
                lv = store.wait_live_after(after, timeout)
                return self._json(lv) if lv else self._send(204)
            if u.path == "/esp":
                # Plain text key=value lines for ESP32/Arduino (no JSON library needed). Long poll:
                #   GET /esp?after=<seq>&timeout=20  -> 200 + state when it changed, 204 when nothing new
                try:
                    after = int(q.get("after", ["-1"])[0])
                    timeout = min(max(float(q.get("timeout", ["20"])[0]), 0.0), 55.0)
                except ValueError:
                    return self._send(400, "after must be an integer, timeout a number\n")
                st = store.wait_state(after, timeout)
                if st is None:
                    return self._send(204)
                clean = lambda v: str(v).replace("\r", " ").replace("\n", " ")
                body = "".join(f"{k}={clean(st[k])}\n" for k in ("seq", "live", "id", "text", "gloss", "time"))
                return self._send(200, body)
            if u.path == "/history":
                try:
                    n = min(max(int(q.get("n", ["20"])[0]), 1), 200)
                except ValueError:
                    n = 20
                return self._json(store.history(n))
            self._send(404, "not found\n")

        def do_POST(self):
            path = urlparse(self.path).path
            if path not in ("/sentence", "/live"):
                return self._send(404, "not found\n")
            if token and self.headers.get("X-Token") != token:
                return self._json({"error": "bad or missing X-Token"}, 401)
            try:
                n = int(self.headers.get("Content-Length", "0"))
                if n <= 0 or n > 20000:
                    raise ValueError("body size")
                data = json.loads(self.rfile.read(n).decode("utf-8"))
                if path == "/live":
                    return self._json(store.set_live(str(data.get("text", "")).strip()))
                gloss = str(data.get("gloss", "")).strip()
                english = str(data.get("english", "")).strip()
                if not (gloss or english):
                    raise ValueError("empty")
            except Exception:  # noqa: BLE001
                return self._json({"error": 'send JSON like {"gloss": "ME GO", "english": "I go."}'}, 400)
            it = store.add(gloss, english)
            print(f"   stored #{it['id']}: {it['text']}", flush=True)
            self._json(it, 201)

    return H


def lan_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))  # no packet is sent; just picks the Wi-Fi interface
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return "127.0.0.1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0", help="0.0.0.0 = reachable from other devices; 127.0.0.1 = this Mac only")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--token", default=None, help="require header X-Token on POST /sentence")
    args = ap.parse_args()
    store = Store()
    srv = ThreadingHTTPServer((args.host, args.port), make_handler(store, args.token))
    srv.daemon_threads = True
    ip = lan_ip()
    print(f"Sentence server running.\n  on this Mac : http://localhost:{args.port}/latest\n"
          f"  for the ESP : http://{ip}:{args.port}/latest      (same Wi-Fi)\n"
          f"  long poll   : http://{ip}:{args.port}/next?after=0\nCtrl+C to stop.\n", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
