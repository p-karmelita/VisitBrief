"""Load the pinned ROCm model and serve the VisitBrief adapter.

Run this with the AMD notebook's existing Python, not the CPU application
environment. It installs nothing and does not replace notebook packages.

    VISITBRIEF_MODEL_API_KEY='...' /opt/conda/envs/py_3.10/bin/python scripts/start_amd_adapter.py

The process stays in the foreground so the local server remains available.
It listens on 127.0.0.1 only. Stop it with the process signal; that does not
cancel a generation that has already started.
"""

import importlib.util
import json
import os
import sys
import threading
import urllib.request
from pathlib import Path

_ADAPTER_PATH = Path(__file__).with_name("amd_notebook_server.py")
_SPEC = importlib.util.spec_from_file_location("amd_notebook_server", _ADAPTER_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("The AMD notebook adapter is missing.")
_ADAPTER = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _ADAPTER
_SPEC.loader.exec_module(_ADAPTER)


def _setting(name: str, default: str) -> str:
    value = os.environ.get(name, default).strip()
    if not value:
        raise RuntimeError(f"{name} is empty.")
    return value


def load_model():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(_ADAPTER.MODEL_ID, revision=_ADAPTER.REVISION)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            _ADAPTER.MODEL_ID,
            revision=_ADAPTER.REVISION,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        )
    except (TypeError, ValueError, ImportError):
        model = AutoModelForCausalLM.from_pretrained(
            _ADAPTER.MODEL_ID,
            revision=_ADAPTER.REVISION,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
    if model.device.type != "cuda":
        model = model.to("cuda")
    model.eval()
    return model, tokenizer


def main() -> None:
    key = os.environ.get("VISITBRIEF_MODEL_API_KEY", "")
    port = int(_setting("VISITBRIEF_ADAPTER_PORT", "8001"))
    context_tokens = int(_setting("VISITBRIEF_MODEL_CONTEXT_TOKENS", "8192"))
    model, tokenizer = load_model()
    handle = _ADAPTER.start_server(
        model, tokenizer, api_key=key, port=port, context_tokens=context_tokens,
    )
    request = urllib.request.Request(
        f"http://127.0.0.1:{handle.port}/health",
        headers={"Authorization": "Bearer " + key},
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        health = json.load(response)
    if health.get("status") != "ok" or health.get("revision") != _ADAPTER.REVISION:
        handle.stop()
        raise RuntimeError("Adapter health check failed.")
    print(
        "adapter_ready "
        f"port={handle.port} model={health['model']} revision={health['revision']} "
        f"device={health['device']} context_tokens={health['context_tokens']}",
        flush=True,
    )
    threading.Event().wait()


if __name__ == "__main__":
    main()
