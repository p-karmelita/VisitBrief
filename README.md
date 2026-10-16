# VisitBrief

Source-linked documentation review for staff preparing for a visit.
**Fictional cases only. Not a clinical tool, diagnosis system, or treatment recommender.**
Every result requires verification by clinical staff; a matching quote
does not establish a correct interpretation or medical truth.

## Current implementation

- Separate FastAPI backend with an allowlisted catalog of 20 fictional cases.
- Pydantic contracts for sources, facts, event dates, review items and draft results.
- No upload, free-form prompt, real patient data, or database.
- Text PDF extraction returns every page and source block, with both raw extracted
  text and whitespace-normalized text. Offsets map blocks to both representations.
  Case details return `source_status: available` only after complete extraction.
- `POST /v1/briefs` extracts facts, keeps a sentence only when its quote is in the
  assigned block, merges duplicates, and builds a timeline and review flags.
  A paraphrased value is replaced by that quote, including when the model cites
  more than one block. An instruction to ignore the prompt or invent a diagnosis
  is not shown as a fact. The response counts kept, repaired and dropped
  candidates. Wording that is not in a block is not repeated.
  No mock model or stored answer is served. Inference is disabled by default.
- Live extractor: AMD Token Factory, model `DeepSeek-V4-Flash`, OpenAI-compatible
  chat completions. The team key stays in gitignored `.env`. The whole team shares
  the key, the 20 RPM cap and the daily spend cap.
- Optional notebook path: a private Transformers adapter for `Qwen/Qwen2.5-7B-Instruct`
  on an Instinct MI300X. That instance is down. Its 14 October measurement is kept
  in `docs/evaluation.md`.
- `/health` reports process health and checks adapter readiness when configured.
  Readiness alone does not prove successful extraction or model quality.
- One active extraction per API process, bounded retries, explicit incomplete
  results, and redacted errors for unavailable, busy or invalid model responses.
- Lightweight English Streamlit UI: case selection, source browsing, draft facts,
  timeline, review flags, and highlighted source quotations. HTTP client only;
  no upload, fallback answers, shared document cache, or direct model access.

## AMD Token Factory demo — 2026-10-15

Copy `.env.example` to `.env` and set the Token Factory block. Do not commit the key.
`VISITBRIEF_MODEL_BACKEND=token_factory` speaks the catalog at
`https://developer.amd.com.cn/radeon/api/v1`. One extraction runs at a time.
Empty fact lists are retried once. HTTP 429s follow `Retry-After`.

On 15 October the 20-case pass recalled 27/58 annotated development facts (46.6%)
and 34/43 held-out facts (79.1%). Seventeen cases were `incomplete`. Median
client-measured endpoint time was 20.7 s. Details and the older notebook numbers
are in `docs/evaluation.md`. Do not describe this as time saved or as clinical accuracy.

```bash
python main.py
```

In a second terminal:

```bash
.venv/bin/python -m streamlit run frontend/app.py \
  --server.address 127.0.0.1 --server.port 8501 --server.headless true \
  --browser.gatherUsageStats false
```

Open `http://127.0.0.1:8501`. Generate `dev-01`, then `dev-02`, then `dev-11`.
`dev-02` is the conflicting-dose case. `dev-11` contains an instruction to invent
a diagnosis; that instruction must not appear as a fact. Both services bind to
loopback. The form still needs a public repository and a judge-visible recording
if the notebook proxy is down.

`docker compose up --build` passes the same `VISITBRIEF_*` variables into the API
container. The key is read from the shell environment or the project `.env` at
compose time. It is not written into the image.

## AMD notebook demo — 2026-10-14

The extractor ran inside AMD notebook `jupyter-user-261014191442-df06a4ff`.
That host now returns HTTP 502, so the live UI is gone with the instance.
The code that had been uploaded there is saved as one local archive,
`backups/amd-notebook-instance-2026-10-14.tar.gz`, with restore steps inside
the archive. The adapter secret that existed only on the notebook could not
be copied. A Hugging Face token belongs in gitignored `.env`, never in the
archive or in Git.

While that notebook was up, the UI was:

`https://notebooks.amd.com/jupyter-user-261014191442-df06a4ff/proxy/8501/`

`/health` on the notebook API reported `inference_available: true`. A real
`POST /v1/briefs` for `dev-01` returned `incomplete`, 4 source-checked facts,
873 input tokens, 553 output tokens and 12.4 s of inference. The full 20-case
measurement is in `docs/evaluation.md`. Recall of the curated facts is below
the project's own target, and every case was marked incomplete.

The notebook file `fine_tune/llama_factory_llama3.ipynb` is AMD's Llama-3.1
fine-tuning tutorial. It was not executed. That recipe needs a gated Hugging
Face token and trains on a demo instruction set, not these visit packets.
Inference uses `Qwen/Qwen2.5-7B-Instruct` revision
`a09a35458c702b33eeacc393d103063234e8bc28`, BF16, on the Instinct MI300X VF
(191.7 GiB) already verified in this image.

Recorded stack: Transformers 4.55.0, PyTorch 2.8.0a0+gitd06a406, HIP
6.4.43484-123eb5128. On a new notebook, unpack the archive, generate a new
adapter key, and keep it only in that machine's runtime file. Do not commit it.

```bash
set -a
. /workspace/visitbrief.runtime.env
set +a
cd /workspace/visitbrief
nohup /opt/conda/envs/py_3.10/bin/python -u scripts/start_amd_adapter.py \
  >> /workspace/visitbrief-adapter.log 2>&1 &
nohup /workspace/visitbrief-cpu/bin/python main.py \
  >> /workspace/visitbrief-api.log 2>&1 &
VISITBRIEF_API_URL=http://127.0.0.1:8000 \
  nohup /workspace/visitbrief-cpu/bin/python -m streamlit run frontend/app.py \
  --server.address 127.0.0.1 --server.port 8501 --server.headless true \
  --browser.gatherUsageStats false --server.enableCORS false \
  --server.enableXsrfProtection false \
  >> /workspace/visitbrief-ui.log 2>&1 &
```

Stop the notebook instance when the demo is finished. Closing JupyterLab does
not release the GPU. Local CPU runs keep inference disabled. Submission text,
the remaining form items and the slide outline are in `docs/`.

## Local development

Use the project's virtual environment in PyCharm. Python 3.12+ is required;
the intended CPU container runtime is 3.12. Local CPU tests do not verify ROCm,
Transformers, or GPU compatibility. GPU inference uses the separate AMD notebook.

From the repository root, using the configured interpreter:

```bash
python -m pip install -e '.[dev,ui]'
python main.py
```

The API binds to `127.0.0.1:8000`; interactive API documentation is at
`http://127.0.0.1:8000/docs`. Stop with Ctrl+C.
Access logging is disabled by the entry point to avoid recording request data.
Do not expose the backend directly to the Internet.

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/v1/cases
curl http://127.0.0.1:8000/v1/cases/dev-01
python -m pytest
python -m ruff check .
docker compose up --build
```

`docker compose` builds the CPU API and UI and publishes them only on
`127.0.0.1`. Inference follows `VISITBRIEF_INFERENCE_ENABLED` from `.env`
(default off). Stop with `docker compose down`. It does not start a notebook
adapter.

Configuration is optional: copy `.env.example` to `.env` or set `VISITBRIEF_*`
environment variables. Never commit credentials. `model_backend=notebook` expects
the private adapter. `model_backend=token_factory` expects the AMD catalog.
There is no second model if the configured endpoint fails.
`VISITBRIEF_CASES_DIR` defaults to this checkout's `fixtures/cases`; custom relative
paths are relative to the working directory. An installed wheel must be configured
with a separately supplied fixtures directory; fixture files are not package data.

## Lightweight frontend

With the backend running, start a second PyCharm terminal at the project root:

```bash
.venv/bin/python -m streamlit run frontend/app.py --server.address 127.0.0.1 --browser.gatherUsageStats false
```

Open `http://127.0.0.1:8501`; stop each process with Ctrl+C. Install the `ui`
extra above first. `VISITBRIEF_API_URL` optionally sets the backend URL in the
frontend process environment (default `http://127.0.0.1:8000`); it is not the
model endpoint and takes no adapter secret. Keep both services private.

Select a fictional case to browse extracted pages even with inference disabled.
When the adapter is ready, **Generate brief** sends only the case ID to the API.
The evidence selector sits beside the facts and shows the document, page, exact
quote, highlighted source block and full normalized page. `dev-01`, `dev-02` and
`dev-11` are labeled in the case list. Review flags are written as sentences. A
model-gate count lists repaired and dropped candidates without repeating wording
that failed the quote check. Facts retain historical/unknown labels; undated
facts stay separate from dated timeline entries. Incomplete results and backend
errors are explicit. A frontend timeout does not cancel backend processing.

Results are kept only in per-session memory and cleared when changing cases or
regenerating. No stored annotations are used as responses. UI tests use controlled
HTTP responses, not live AMD inference:

```bash
.venv/bin/python -m pytest tests/test_frontend.py
```

## Data and boundaries

`fixtures/cases/catalog.json` is the allowlist, with one manifest per case.
Clients submit identifiers, never paths. Paths, expected answers, and secrets are
not returned by the API. The fixtures directory is trusted, read-only deployment
data and must not be writable by public clients. Responses use `Cache-Control: no-store`.

The 12 development cases and eight held-out cases are separated in manifests and
annotations. Held-out content must not be used for model or prompt tuning. Expected
facts are evaluation material, never an inference implementation. The small synthetic
set is not medical validation. Extraction agreement and inference time were measured
once on the notebook GPU; see `docs/evaluation.md`. No time-saving or clinical-quality
claim was measured.

`Dockerfile.api`, `Dockerfile.ui` and `compose.yaml` run the CPU application.
Inference stays disabled unless `VISITBRIEF_INFERENCE_ENABLED=true`. The live
extractor is Token Factory, not a published GPU image. A public repository and a
judge-visible demo are still required for the hackathon form. The pitch deck is
`docs/VisitBrief-pitch.pptx`. The slide outline is `docs/slides.md`.

## PDF processing contract

- Limits: three documents, 10 MiB per file, 20 pages total, 200,000 extracted characters.
  Configuration can lower these ceilings. File reads are bounded; the text limit is
  checked after extraction and is not an OS-level memory limit for hostile compressed PDFs.
- Empty pages (including an isolated blank page), scans without text, encrypted PDFs
  (even with an empty password), and malformed PDFs fail visibly. No page is dropped.
- Source IDs follow `case-id:document-id:p1:b1`. Equal quotations on different pages
  or in different cases have different IDs. Duplicate pages remain separate sources.
- Blocks contain offsets in normalized page text and raw `pypdf` output. This is text
  provenance, not visual bounding boxes or OCR. Whitespace is normalized; wording,
  punctuation, drug names and units are not rewritten. A citation must be present
  in its assigned block, not merely somewhere else in the document.
- IDs are repeatable for the same PDFs, parser version and block-size configuration.
  Do not change these within an active review session. The default is 2,000 characters
  per block; model batches do not renumber source IDs.
- `pack_blocks` uses the adapter's model tokenizer through the inference client;
  counts include the serialized JSON, system prompt and chat template, with a
  separate output budget.
  It preserves all blocks in order and rejects mixed cases, duplicate IDs and blocks
  that cannot fit. There is no guessed characters-per-token ratio or silent truncation.
  Live tokenization and context-budget verification still require AMD integration.
- Parser diagnostics are suppressed because they can include document contents;
  API errors expose stable codes only. Extracted pages are not logged or cached.
- Document text remains untrusted input. Keeping an injected instruction in its
  source text does not execute it; resistance by the model must be tested separately.