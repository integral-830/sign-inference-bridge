"""Sends finished sentences to server.py (used by live_demo.py --push). Standard library only.
Runs in a background thread: a slow or missing server never blocks the camera loop."""
import json
import sys
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor


class Pusher:
    def __init__(self, base_url, token=None, timeout=3.0):
        self.url = base_url.rstrip("/") + "/sentence"
        self.token, self.timeout = token, timeout
        self.pool = ThreadPoolExecutor(max_workers=1)   # one at a time keeps the sentence order
        self.live_pool = ThreadPoolExecutor(max_workers=1)
        self.live_url = base_url.rstrip("/") + "/live"
        self.live_lock, self.live_pending, self.live_sent, self.live_queued = threading.Lock(), None, "", False
        self.warned = False

    def send_live(self, text):
        """Latest in-progress sentence. Only changes are sent, and if the server is slow older values are skipped."""
        with self.live_lock:
            if text == self.live_sent and not self.live_queued:
                return
            self.live_pending = text
            if self.live_queued:
                return
            self.live_queued = True
        self.live_pool.submit(self._live_worker)

    def _live_worker(self):
        while True:
            with self.live_lock:
                text = self.live_pending
                if text is None or text == self.live_sent:
                    self.live_queued = False
                    return
            if self._post_to(self.live_url, {"text": text}):
                with self.live_lock:
                    self.live_sent = text
            else:
                with self.live_lock:       # server down: stop for now, a later change tries again
                    self.live_queued = False
                return

    def send(self, gloss, english=""):
        self.pool.submit(self._post, gloss, english)

    def _post(self, gloss, english):
        self._post_to(self.url, {"gloss": gloss, "english": english})

    def _post_to(self, url, payload):
        req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                     headers={"Content-Type": "application/json"}, method="POST")
        if self.token:
            req.add_header("X-Token", self.token)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                r.read()
            self.warned = False
            return True
        except Exception as e:  # noqa: BLE001
            if not self.warned:  # say it once, keep trying for later sentences
                print(f"[push] could not reach {url} ({type(e).__name__}: {e}). Is server.py running?",
                      file=sys.stderr, flush=True)
                self.warned = True
            return False

    def close(self):
        self.pool.shutdown(wait=True)
        self.live_pool.shutdown(wait=True)
