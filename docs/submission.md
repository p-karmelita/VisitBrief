# Submission pack — Track 2, Health and Wellbeing

VisitBrief prepares fictional visit packets for clinical staff. It turns one to
three text PDFs into source-linked facts, a timeline and an explicit review
list. It does not diagnose, recommend treatment, accept uploads, or take a
free-form prompt.

## Suggested form text

**Title:** VisitBrief

**Short description:** Source-linked visit preparation for staff. DeepSeek-V4-Flash
on AMD Token Factory reads fictional charts, and every displayed fact must quote
its source or it is dropped.

**Challenge:** Health and Wellbeing (Track 2).

**Technology tags:** AMD Token Factory, Radeon Cloud, FastAPI, Streamlit,
DeepSeek-V4-Flash. Earlier notebook path: AMD Instinct, ROCm, PyTorch,
Transformers, Qwen2.5-7B-Instruct.

**Long description:**

Staff preparing for a visit have to read several short documents and notice
what is historical, missing, or in conflict. VisitBrief keeps that job with
the person. An allowlisted fictional case is parsed into pages and source
blocks. DeepSeek-V4-Flash on AMD Token Factory returns candidate facts. Radeon
Cloud states that catalog models run on AMD GPUs. The API keeps a fact only
when its quote is in the assigned block. If the model paraphrases the value,
the screen shows the quote instead. Duplicates are merged, dates that are not
in the quote are cleared, and conflicting doses are shown side by side instead
of reconciled. A sentence that tells the model to ignore instructions or invent
a diagnosis is dropped, and the case is flagged. Missing allergy documentation,
undated facts and incomplete extraction stay visible. The interface says the
result is a draft for clinician review.

The demo has no upload and no chat box, so it cannot accept a real patient
record. There is no database. Request logs do not store document text.
Held-out cases were not used to change the prompt. One pass on 15 October
recalled 27/58 curated development facts (46.6%) and 34/43 held-out facts
(79.1%). Seventeen of twenty cases were incomplete. The shared endpoint is not
stable across repeats. Those figures are extraction agreement with a small
synthetic set, not clinical accuracy or time saved. An earlier Instinct MI300X
pass with Qwen2.5-7B-Instruct recalled 22.4% and 18.6%. That notebook is down.

## Safety answers the brief asks for

| Risk | What the product does |
| --- | --- |
| Incorrect or unsupported response | A paraphrase or unknown source id is dropped. The case becomes `incomplete`. |
| Sensitive information | Only built-in fictional PDFs. No upload, no free prompt, no stored patient record. |
| Uncertain results | Missing dates stay off the timeline. Missing allergy text is a review flag, not a "no allergies" fact. |
| High-risk decisions | No diagnosis, no dose change, no treatment advice. Conflicting doses are both shown. |
| Human review | The UI states that a clinician must review every result before it is used. |

## Demo path

1. Start the API and Streamlit from the repository root, with the Token Factory
   block in gitignored `.env`. Open `http://127.0.0.1:8501`.
2. Select `dev-01` and generate a brief. Expect a draft, source highlights and
   an incomplete or review state, not a finished clinical note.
3. Select `dev-02` for two recorded doses of the same medicine that the app
   does not choose between. A single draw can miss one of the doses. The
   15 October measurement pass kept both and raised `conflicting_medication_doses`.
4. Select `dev-11` for a document that contains an instruction to invent a
   diagnosis. No fact from that instruction should be displayed. The review list
   should include `untrusted_instruction_in_source`.

The notebook proxy from 14 October is down (HTTP 502). Judges need the recording
in `docs/demo/` or a fresh local run. Do not put the team API key in the form.

## Still required before the lablab form is complete

- A public Git repository. Do not commit `.env` or
  `/workspace/visitbrief.runtime.env`.
- A demo video of the three cases above, including the review flags. Record
  into `docs/demo/` from the local UI. The notebook cannot be shown while it
  returns HTTP 502.
- The pitch deck `docs/VisitBrief-pitch.pptx`, from the outline in `docs/slides.md`.
- The cover image `docs/cover.jpg`.
- The form fields copied from this file.
- If an Instinct notebook is started again, stop the instance when the demo is
  finished. Closing the browser tab does not release the GPU. Token Factory
  does not hold a notebook GPU.

Evolus is optional for Track 2 and is not part of this project. The CPU
containers in `compose.yaml` reproduce the API and UI. Set
`VISITBRIEF_MODEL_BACKEND=token_factory` to call the catalog from the container.
The key stays out of the image.
