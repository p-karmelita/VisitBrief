<!--START_SECTION:badges-->
# VisitBrief

Source-linked visit preparation for staff. An allowlisted fictional chart becomes draft facts, a timeline and a review list. Every line on the screen quotes the page it came from.

**Fictional cases only. Not a clinical tool, diagnosis system, or treatment recommender.** A matching quote does not make an interpretation medically true. A clinician reviews every result before it is used.

Live extraction uses [AMD Token Factory](https://developer.amd.com.cn/) on Radeon Cloud, model `DeepSeek-V4-Flash`.

## Technologies

| Layer | Stack |
| --- | --- |
| Language | Python 3.12 |
| API | FastAPI, Uvicorn, Pydantic v2 |
| PDF text | pypdf. Pages and source blocks, no OCR |
| Model call | httpx, OpenAI-compatible chat completions |
| Model | `DeepSeek-V4-Flash` on AMD Token Factory (Radeon Cloud, AMD GPUs) |
| UI | Streamlit. Case list, facts, timeline, review flags, highlighted quotes |
| Run | Docker Compose. API and UI on loopback only |
| Checks | pytest, Ruff |

## How a brief is built

1. The client sends a case id from a catalog of 20 fictional packets. No file upload, no free-form prompt, no database.
2. Text PDFs become pages and source blocks. Offsets point at the extracted text.
3. One Token Factory call returns candidate facts.
4. A fact is kept only when its quote is inside the block it cites. A paraphrase is replaced by that quote. Wording that is not in a block is dropped and is not repeated on screen.
5. Duplicates merge. Conflicting doses stay side by side. Undated facts stay off the timeline. Missing allergy documentation is a review flag, not a "no allergies" fact.
6. The UI labels the result as a draft. Incomplete extraction stays visible.

## Run

Python 3.12+. From the repository root:

```bash
python -m pip install -e '.[dev,ui]'
cp .env.example .env
```

In `.env`, enable the Token Factory block and set `VISITBRIEF_MODEL_API_KEY`. Do not commit `.env`.

```bash
python main.py
```

Second terminal:

```bash
python -m streamlit run frontend/app.py \
  --server.address 127.0.0.1 --server.port 8501 \
  --browser.gatherUsageStats false
```

Open [http://127.0.0.1:8501](http://127.0.0.1:8501). API docs are at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs). With inference left off, the UI still browses source pages and **Generate brief** stays disabled.

The same two services, still bound to loopback, and still reading the key from the environment rather than baking it into the image:

```bash
docker compose up --build
```

```bash
python -m pytest
python -m ruff check .
```

## Boundaries

- Built-in fictional PDFs only. Clients send case ids, never paths or documents.
- No diagnosis, no dose change, no treatment advice.
- Request logs do not store document text. Responses use `Cache-Control: no-store`.
- One extraction at a time. Errors returned to the client are redacted.
- The 12 development cases and 8 held-out cases stay separate. Expected answers are evaluation material, not the model response.

Public repository: [github.com/p-karmelita/VisitBrief](https://github.com/p-karmelita/VisitBrief)
<!--END_SECTION:badges-->
