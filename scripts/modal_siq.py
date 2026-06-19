"""
modal_siq.py — serve SIQ-1-35B on Modal via llama.cpp (the only engine that runs
its hybrid linear-attention MoE arch `Qwen3_5MoeForCausalLM`; vLLM/SGLang can't).

Unlike the RunPod GGUF worker, llama-server here is a real **streaming** OpenAI
endpoint (token-by-token SSE), so the web app gets live output.

  python3 -m modal run    scripts/modal_siq.py::download   # cache the GGUF (once)
  python3 -m modal deploy scripts/modal_siq.py             # deploy the web endpoint
  python3 -m modal run    scripts/modal_siq.py::smoke      # quick generation test

Needs the `huggingface` Modal secret (HF_TOKEN) — SIQ-1-35B is a private repo.
"""
import os
import subprocess
import modal

REPO = "AlexWortega/SIQ-1-35B"
GGUF_REL = "gguf/SIQ-1-35B.Q4_K_M.gguf"
GGUF_PATH = f"/models/{GGUF_REL}"
N_CTX = 65536

app = modal.App("siq1-llama")
vol = modal.Volume.from_name("siq1-gguf", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface")

# Prebuilt llama.cpp CUDA server (tracks master → has the qwen3next / hybrid
# linear-attn arch). Avoids compiling ggml-cuda against an absent GPU driver.
image = (
    modal.Image.from_registry("ghcr.io/ggml-org/llama.cpp:server-cuda", add_python="3.11")
    .entrypoint([])  # the image's ENTRYPOINT is llama-server; clear it so Modal can run python
    .pip_install("huggingface_hub[hf_transfer]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)


def _llama_server_bin() -> str:
    """Locate the llama-server binary in the prebuilt image."""
    import shutil

    for c in ("/app/llama-server", "/llama-server", "/usr/local/bin/llama-server"):
        if os.path.exists(c):
            return c
    return shutil.which("llama-server") or "/app/llama-server"


@app.function(image=image, volumes={"/models": vol}, secrets=[hf_secret], timeout=3600)
def download():
    """Cache the GGUF into the volume (one-time, ~21 GB private download)."""
    from huggingface_hub import hf_hub_download

    if os.path.exists(GGUF_PATH):
        print("already cached:", GGUF_PATH)
        return
    hf_hub_download(REPO, GGUF_REL, local_dir="/models", token=os.environ["HF_TOKEN"])
    vol.commit()
    print("downloaded:", GGUF_PATH, os.path.getsize(GGUF_PATH))


@app.function(
    image=image,
    gpu="L40S",
    volumes={"/models": vol},
    secrets=[hf_secret],
    timeout=3600,
    scaledown_window=300,
    max_containers=1,
)
@modal.web_server(8080, startup_timeout=900)
def serve():
    """Run llama-server (streaming OpenAI API) on the Modal web endpoint."""
    vol.reload()
    cmd = [
        _llama_server_bin(),
        "-m", GGUF_PATH,
        "--alias", "siq",
        "--host", "0.0.0.0", "--port", "8080",
        "-ngl", "99",
        "-c", str(N_CTX),
        "-np", "1",
        "--jinja",
    ]
    print("launching:", " ".join(cmd))
    subprocess.Popen(cmd)


@app.function(image=image, gpu="L40S", volumes={"/models": vol}, secrets=[hf_secret], timeout=900)
def smoke():
    """Load the model in-process via a short llama-server run + curl, print the reply."""
    import time
    import urllib.request

    vol.reload()
    p = subprocess.Popen(
        [_llama_server_bin(), "-m", GGUF_PATH, "--host", "127.0.0.1", "--port", "8080",
         "-ngl", "99", "-c", "8192", "--jinja"],
    )
    try:
        for _ in range(180):
            time.sleep(2)
            try:
                urllib.request.urlopen("http://127.0.0.1:8080/v1/models", timeout=3)
                break
            except Exception:
                pass
        body = {
            "model": "siq", "max_tokens": 64,
            "messages": [{"role": "user", "content": "What is 2+2? Reply with only the number."}],
            "chat_template_kwargs": {"enable_thinking": False},
        }
        import json
        req = urllib.request.Request(
            "http://127.0.0.1:8080/v1/chat/completions",
            data=json.dumps(body).encode(), headers={"content-type": "application/json"},
        )
        out = json.load(urllib.request.urlopen(req, timeout=120))
        print("REPLY:", out["choices"][0]["message"]["content"])
    finally:
        p.terminate()
