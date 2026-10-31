"""ONE COMMAND to run everything:  python run.py

  1. finds your trained weights by itself (model/words + model/letters, or model/sign_model.pt)
  2. starts the HTTP sentence server (port 8000, reachable from other devices on the same Wi-Fi)
  3. starts the live camera recognition and sends sentences + the sentence-in-progress to that server
  4. prints the exact curl commands for your phone / ESP / another computer

Examples
  python run.py                          # normal use (camera window on)
  python run.py --no-window              # no camera window, terminal only
  python run.py --translate llm          # force Groq English (needs GROQ_API_KEY), default 'auto' = llm if a key exists
  python run.py --token mysecret         # only senders with this secret can post (reading stays open)
  python run.py --port 9000
  python run.py --server-only            # just the server, no camera or model (to test curl from a phone first)
  python run.py --simulate               # server + fake sentences every few seconds (test the ESP without signing)
  python run.py -- --threshold 0.7 --fast     # everything after '--' goes straight to live_demo.py

Groq key: export GROQ_API_KEY="gsk_..."   or put   GROQ_API_KEY=gsk_...   in a file named .env next to this script.
(Do not share the .env file.)
"""
import argparse
import os
import sys
import threading
import time
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load_dotenv(path):
    """Tiny .env reader: KEY=VALUE lines, never overrides a variable that is already set."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip().removeprefix("export ").strip(), v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


def find_weights(root):
    """Words + letters models if present, else the single model, else any sign_model.pt under model/."""
    pair = [p for p in (root / "model/words/sign_model.pt", root / "model/letters/sign_model.pt") if p.exists()]
    if pair:
        return pair
    single = root / "model/sign_model.pt"
    if single.exists():
        return [single]
    return sorted(root.glob("model/**/sign_model.pt"))[:2]


def pick_translate(mode):
    if mode != "auto":
        return mode
    try:
        import groq  # noqa: F401
        has_lib = True
    except ImportError:
        has_lib = False
    return "llm" if (os.environ.get("GROQ_API_KEY") and has_lib) else "simple"


def health_ok(port):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
            return r.read().strip() == b"ok"
    except Exception:  # noqa: BLE001
        return False


def start_server(host, port, token):
    """Start server.py's HTTP server in a background thread. Returns (store, httpd or None if one was already running)."""
    from server import Store, make_handler
    store = Store()
    try:
        httpd = ThreadingHTTPServer((host, port), make_handler(store, token))
    except OSError:
        if health_ok(port):
            print(f"A sentence server is already running on port {port}: reusing it.")
            return None, None
        raise SystemExit(f"Port {port} is in use by something else. Try:  python run.py --port {port + 1}")
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, daemon=True, name="http").start()
    return store, httpd


def short_path(p):
    p = Path(p)
    try:
        return str(p.relative_to(HERE))
    except ValueError:      # a weights file from another folder: show it as given
        return str(p)


def banner(port, ip, token, translate, weights, simulate=False):
    base = f"http://{ip}:{port}"
    print("=" * 72)
    print(" SIGN SERVER READY")
    print("=" * 72)
    print(f" This Mac      : http://localhost:{port}")
    print(f" Other devices : {base}      (phone / ESP / laptop on the SAME Wi-Fi)")
    if weights:
        print(f" Weights       : {', '.join(short_path(w) for w in weights)}")
    if translate:
        print(f" English       : {translate}")
    print(f" Password      : {'X-Token header needed to POST' if token else 'none (reading and posting open on your LAN)'}")
    print("-" * 72)
    print(" From a phone, ESP or terminal, try these:")
    print(f"   curl {base}/health                   # ok")
    print(f"   curl {base}/live                     # sentence being signed right now (words so far)")
    print(f"   curl {base}/latest                   # last FINISHED sentence (English, else glosses)")
    print(f"   curl {base}/latest.json              # id, time, gloss, english")
    print(f"   curl \"{base}/history?n=10\"           # last 10 sentences")
    print(f"   curl \"{base}/next?after=0&timeout=20\"        # waits for the next finished sentence")
    print(f"   curl \"{base}/live/next?after=0&timeout=20\"   # waits until the live text changes")
    print(" A phone browser works too: open the 'Other devices' address + /latest")
    if simulate:
        print(" SIMULATION: fake sentences are posted every few seconds (no camera, no model).")
    print(" Ctrl+C (or q in the camera window) stops everything.")
    print("=" * 72, flush=True)


SIM = [("ME SCHOOL GO", "I am going to school."), ("YOU NAME WHAT", "What is your name?"),
       ("TODAY HOT", "It is hot today."), ("THANK YOU", "Thank you.")]


def simulate(store):
    """Fake signer: builds each sentence word by word in /live, then finishes it."""
    i = 0
    while True:
        gloss, english = SIM[i % len(SIM)]
        words = gloss.split()
        for k in range(1, len(words) + 1):
            time.sleep(1.2)
            store.set_live(" ".join(words[:k]))
        time.sleep(1.5)
        item = store.add(gloss, english)
        print(f"   [sim] stored #{item['id']}: {item['text']}", flush=True)
        time.sleep(3)
        i += 1


def main():
    argv = sys.argv[1:]
    passthrough = []
    if "--" in argv:
        k = argv.index("--")
        argv, passthrough = argv[:k], argv[k + 1:]
    ap = argparse.ArgumentParser(description="One command: weights + HTTP server + live recognition.")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="0.0.0.0", help="0.0.0.0 = other devices can connect; 127.0.0.1 = this Mac only")
    ap.add_argument("--token", default=None, help="require header X-Token on POST (reading stays open)")
    ap.add_argument("--weights", nargs="+", default=None, help="model files; default = found automatically")
    ap.add_argument("--translate", default="auto", choices=["auto", "off", "simple", "llm"])
    ap.add_argument("--translate-model", default="openai/gpt-oss-20b")
    ap.add_argument("--translate-context", default=None,
                    help='facts about the signer for the English step, e.g. "Ravi, a student; talks about school and food" '
                         "(or SIGN_CONTEXT in the environment / .env)")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--no-window", action="store_true")
    ap.add_argument("--server-only", action="store_true", help="only the HTTP server (no camera, no model)")
    ap.add_argument("--simulate", action="store_true", help="server + fake sentences, to test the ESP/phone")
    args = ap.parse_args(argv)

    os.chdir(HERE)                       # model/, transcripts/ and data/ are always relative to this folder
    sys.path.insert(0, str(HERE))
    load_dotenv(HERE / ".env")
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    from server import lan_ip
    ip = lan_ip()

    weights = translate = None
    if not (args.server_only or args.simulate):
        weights = [Path(w).resolve() for w in args.weights] if args.weights else find_weights(HERE)
        if not weights:
            raise SystemExit("No trained model found.\n  Expected model/words/sign_model.pt + model/letters/sign_model.pt, "
                             "or model/sign_model.pt.\n  Train first (RUNBOOK.md Part 8):  python train.py   "
                             "or   python train.py --subset words / --subset letters\n"
                             "  Or test the server only:  python run.py --simulate")
        missing = [w for w in weights if not w.exists()]
        if missing:
            raise SystemExit(f"Weights file not found: {missing[0]}")
        translate = pick_translate(args.translate)
        if translate == "llm" and not os.environ.get("GROQ_API_KEY"):
            print("[run] GROQ_API_KEY is not set: English will use simple formatting. "
                  "Set it (export GROQ_API_KEY=...) or add it to .env", file=sys.stderr)

    store, httpd = start_server(args.host, args.port, args.token)
    if not health_ok(args.port):
        raise SystemExit("The HTTP server did not start. Check that the port is free (python run.py --port 9000).")
    banner(args.port, ip, args.token, translate, weights, args.simulate)

    try:
        if args.server_only or args.simulate:
            if args.simulate:
                if store is None:
                    raise SystemExit("--simulate needs its own server: stop the other one first or use --port.")
                threading.Thread(target=simulate, args=(store,), daemon=True).start()
            while True:
                time.sleep(3600)
        else:
            cmd = ["live_demo.py", "--weights", *map(str, weights), "--push", f"http://127.0.0.1:{args.port}",
                   "--translate", translate, "--translate-model", args.translate_model, "--camera", str(args.camera)]
            if args.translate_context:
                cmd += ["--translate-context", args.translate_context]   # else live_demo reads SIGN_CONTEXT itself
            if args.token:
                cmd += ["--push-token", args.token]
            if args.no_window:
                cmd.append("--no-window")
            sys.argv = cmd + passthrough
            import live_demo
            live_demo.main()
    except KeyboardInterrupt:
        pass
    finally:
        if httpd is not None:
            httpd.shutdown()
            httpd.server_close()
        print("\nStopped.")


if __name__ == "__main__":
    main()
