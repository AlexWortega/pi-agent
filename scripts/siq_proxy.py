#!/usr/bin/env python3
"""
siq_proxy.py — OpenAI-compatible async proxy for the SIQ-1-35B RunPod *serverless*
endpoint (siq1.md §3.A). Exposes /v1/chat/completions on 127.0.0.1:8080 and
bridges to RunPod using async /run + poll /status/{id} (NOT /runsync, which
times out under concurrency and returns a job-status object instead of a
completion). Retries 3x, always returns a valid OpenAI object or a clean 502.

Env:
  RUNPOD_API_KEY  (required)         RunPod API key
  SIQ_EID         (default leufrm6iskrs3v)  serverless endpoint id
  SIQ_NOTHINK     (default 0)        1 -> force enable_thinking:false
  SIQ_MINTOK      (default 4096)     floor for max_tokens
  SIQ_EFFORT      (default low)      injected "Reasoning effort:" when thinking
  SIQ_PORT        (default 8080)

Run: RUNPOD_API_KEY=... SIQ_MINTOK=8192 python3 scripts/siq_proxy.py
"""
import json
import os
import time
import urllib.request
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

API_KEY = os.environ.get("RUNPOD_API_KEY", "")
EID = os.environ.get("SIQ_EID", "leufrm6iskrs3v")
NOTHINK = os.environ.get("SIQ_NOTHINK", "0") == "1"
MINTOK = int(os.environ.get("SIQ_MINTOK", "4096"))
EFFORT = os.environ.get("SIQ_EFFORT", "low")
PORT = int(os.environ.get("SIQ_PORT", "8080"))
BASE = f"https://api.runpod.ai/v2/{EID}"


def _post(url, payload):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"content-type": "application/json", "authorization": f"Bearer {API_KEY}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


def _get(url):
    req = urllib.request.Request(url, headers={"authorization": f"Bearer {API_KEY}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


def run_and_poll(openai_req, poll_timeout=600):
    """Submit via /run, poll /status until COMPLETED, return the OpenAI-shaped result."""
    # The serverless worker runs vLLM in OpenAI mode; pass the request straight through.
    job = _post(f"{BASE}/run", {"input": {"openai_route": "/v1/chat/completions", "openai_input": openai_req}})
    job_id = job.get("id")
    if not job_id:
        # Some workers answer synchronously; accept that too.
        if "output" in job:
            return job["output"]
        raise RuntimeError(f"no job id in /run response: {json.dumps(job)[:300]}")
    deadline = time.time() + poll_timeout
    while time.time() < deadline:
        st = _get(f"{BASE}/status/{job_id}")
        status = st.get("status")
        if status == "COMPLETED":
            out = st.get("output")
            # output may be the OpenAI object directly, or a list of streamed chunks.
            if isinstance(out, list) and out:
                return out[-1] if isinstance(out[-1], dict) else {"choices": [{"message": {"content": "".join(map(str, out))}}]}
            return out
        if status in ("FAILED", "CANCELLED", "TIMED_OUT"):
            raise RuntimeError(f"job {status}: {json.dumps(st)[:300]}")
        time.sleep(1.0)
    raise TimeoutError(f"poll timeout after {poll_timeout}s for job {job_id}")


def shape_request(body):
    """Apply SIQ defaults from siq1.md: max_tokens floor, thinking/effort toggles."""
    body = dict(body)
    body["max_tokens"] = max(int(body.get("max_tokens") or 0), MINTOK)
    ctk = dict(body.get("chat_template_kwargs") or {})
    if NOTHINK:
        ctk["enable_thinking"] = False
    body["chat_template_kwargs"] = ctk
    if ctk.get("enable_thinking", True) and EFFORT:
        msgs = list(body.get("messages") or [])
        if not any(m.get("role") == "system" and "Reasoning effort:" in str(m.get("content", "")) for m in msgs):
            msgs.insert(0, {"role": "system", "content": f"Reasoning effort: {EFFORT}"})
        body["messages"] = msgs
    return body


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.rstrip("/") in ("/v1/models", "/models"):
            self._send(200, {"object": "list", "data": [{"id": "siq", "object": "model"}]})
        elif self.path.rstrip("/") in ("/health", "/healthz"):
            self._send(200, {"ok": True})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path.rstrip("/") not in ("/v1/chat/completions", "/chat/completions"):
            return self._send(404, {"error": "not found"})
        length = int(self.headers.get("content-length", "0"))
        body = json.loads(self.rfile.read(length).decode() or "{}")
        req = shape_request(body)
        last = None
        for attempt in range(3):
            try:
                out = run_and_poll(req)
                if not isinstance(out, dict) or "choices" not in out:
                    raise RuntimeError(f"unexpected worker output: {json.dumps(out)[:200]}")
                return self._send(200, out)
            except Exception as e:  # noqa: BLE001
                last = e
                time.sleep(1.5 * (attempt + 1))
        self._send(502, {"error": {"message": f"siq_proxy upstream failed: {last}", "type": "upstream_error"}})

    def log_message(self, *_):
        pass


def main():
    if not API_KEY:
        raise SystemExit("RUNPOD_API_KEY is required")
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"siq_proxy on http://127.0.0.1:{PORT}/v1  (eid={EID} nothink={NOTHINK} mintok={MINTOK} effort={EFFORT})", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
