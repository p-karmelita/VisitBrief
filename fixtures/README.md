### VisitBrief Step 1: fictional document fixtures

**FICTIONAL CASES - NOT REAL PATIENTS.** All narratives, dates, measurements,
medication lists and annotations were invented for software testing. No real
patient records were used. The material has **not been medically validated** and
must not be used for diagnosis, treatment, prescribing or clinical decisions.
Medication doses are source text, not medical advice.

### Inventory and split

| Split | Case IDs | Cases | PDF documents | Pages | Annotated facts |
| --- | --- | ---: | ---: | ---: | ---: |
| `development` | `dev-01` through `dev-12` | 12 | 21 | 23 | 58 |
| `held_out` | `holdout-01` through `holdout-08` | 8 | 16 | 17 | 43 |
| Total | | 20 | 37 | 40 | 101 |

Each case has 1-3 documents. Each PDF has 1-2 short, English, ASCII-text pages,
each under 1,500 characters including its header. Every page displays
`FICTIONAL CASE - NOT A REAL PATIENT`, the case ID, document ID and page number.
The PDFs are real PDF 1.4 files with selectable text streams, a standard Courier
font, page trees and cross-reference tables. They are not renamed text files,
scans or images. No OCR is needed.

The split is fixed in the repository, not randomly sampled. Use only development
cases for future prompt, extraction, matching or heuristic development. Held-out
cases and their expected annotations are reserved for evaluation after those
decisions are frozen. Building them and checking structural integrity is not
permission to tune logic against their expected answers. No extraction model,
prompt fitting, inference or clinical scoring is performed by the generator.

### Files and reproducibility

- `sources/development.json` and `sources/held_out.json`: hand-authored source of
  truth. Each case contains metadata, document bodies as a `pages` array, and
  manually designed `facts` and `review_items`. The two splits are kept separate.
- `cases/catalog.json`: JSON array of exactly 20 case ID strings, development
  first, in numeric order within each split.
- `cases/{case_id}/manifest.json`: `case_id`, `title`, `description`, `split`,
  `fictional: true` and `documents`. Each document has `document_id`, `title`,
  `document_date` (ISO date or `null`) and `filename` (basename only).
- `cases/{case_id}/{document_id}.pdf`: generated text document beside its manifest.
- `expected/{case_id}.json`: `case_id`, `split`, `facts` and `review_items` only.
  Facts and review items are copied from the manual source, not inferred from PDF.

From the repository root, use the project interpreter (or another Python 3.10+
interpreter; only the standard library is used):

```sh
.venv/bin/python -B scripts/build_fixtures.py
.venv/bin/python -B scripts/build_fixtures.py --check
.venv/bin/python -B scripts/build_fixtures.py --self-test
```

The default command creates missing artifacts and leaves byte-identical files
untouched. The output is deterministic: no random values, clock timestamps,
external services, installed packages or application imports are used. There are
78 generated artifacts: 37 PDFs, 20 manifests, 20 expected JSON files and a catalog.
All generated artifacts are included alongside their editable source in the repo.

To protect parallel work, the builder preflights all expected destinations and
**refuses to overwrite any differing file**, including a stale generated file.
It refuses symlink destinations and uses exclusive file creation. It does not
delete files, purge directories or modify unrelated artifacts. After intentionally
editing source, review and remove only the specific obsolete generated artifacts
you own before regenerating; there is deliberately no force-overwrite option.
`--check` validates source and compares every expected artifact byte-for-byte
without writing. It reports missing or different files. It does not remove or
reject unrelated extra files in the fixture tree.

`--self-test` runs seven stdlib generator checks, including malformed input,
unsupported quotes, cross-case references, duplicate JSON keys, PDF offsets and
stream lengths, repeatability, and safe publication. It creates and removes its
own temporary directory inside `fixtures/`, never in application or test modules.

### Annotation conventions

These are manually designed, evidence-grounded reference annotations for a small
synthetic software corpus, **not clinician-reviewed ground truth**. The fact list
is a curated set of target statements, not an exhaustive annotation of every
administrative sentence. Values preserve source assertions, uncertainty, negation
and attribution; they do not add differential diagnoses or new recommendations.

- Fact IDs are local and sequential (`fact-01`, `fact-02`, ...). Categories are
  `problem`, `medication`, `allergy`, `event` and `recommendation`. An explicitly
  negated symptom stays negated in `value`; category alone is not a positive
  diagnosis flag. A negative medication entry does not establish absence of all
  other medicines.
- A source ID is `{case_id}:{document_id}:p{page_number}:b1`, with one conceptual
  source block covering the entire page. Page numbers are one-based. This is an
  annotation convention, not a production parser or chunking implementation.
- Each `quote` is a literal, contiguous substring of that page after all
  whitespace is collapsed to single spaces. Quotes never join separate pages or
  cases. The same quotation on different pages retains different source IDs.
- Repeated assertions about the same finding are one fact with multiple evidence
  entries, not several new findings. Different doses remain separate facts linked
  by `conflicting_medication_doses`; neither source wins by position or date.
- `event_date` identifies the described occurrence when explicitly dated, not a
  later signing date. For a current assessment, medication list, allergy review
  or recommendation, it is the date of that documented assertion. It does **not**
  invent disease onset, medication start or allergy reaction dates. A documented
  allergy review may therefore have a date while the reaction onset is still
  undated and flagged for review.
- `recorded` means the statement is recorded as an assessment/status/plan at its
  stated encounter. `historical` means a prior occurrence or completed/stopped
  treatment in the case packet. `unknown` means the temporal status or date anchor
  is unresolved. These labels are relative to the fictional record, not today's
  real-world date. An undated but explicitly completed course is `historical`
  with `event_date: null`, not current medication.
- Ambiguous numeric dates, missing years and unanchored relative dates stay
  `null`; no day/month order is guessed. Future appointments remain recommendations
  dated when the plan was recorded, with the proposed appointment date in `value`;
  they are not completed events. Cancelled procedures remain cancelled.
- Absence of an allergy section produces `missing_allergy_documentation`, never
  a fabricated no-allergy fact. Explicitly unknown allergy history is a separate
  assertion, not a negative. Explicit drug-allergy denials retain their drug-only
  scope. Intolerance is not silently converted into allergy.
- `review_items` use stable reason codes and local `fact_ids`. An empty ID list
  denotes a case/source-level issue with no suitable clinical fact, for example
  a missing allergy section or an untrusted instruction. Other codes used here are
  `conflicting_medication_doses`, `undated_information`, `incomplete_document`,
  `unknown_allergy_status` and `untrusted_instruction_in_source`.

### Development coverage

| Case | Distinct narrative and annotation challenge |
| --- | --- |
| `dev-01` | Routine hypertension review and home measurements; complete documentation |
| `dev-02` | Thyroid medication lists with unresolved 50/75 mcg disagreement |
| `dev-03` | Wrist injury, negative radiograph, no allergy section |
| `dev-04` | Previous chest discomfort versus current explicit symptom denials |
| `dev-05` | Sinus treatment completed; historical antibiotic versus current spray |
| `dev-06` | Ankle injury and imaging dates different from document dates |
| `dev-07` | Undated headache questionnaire, ambiguous date, unknown allergy history |
| `dev-08` | Diabetes diagnosis and medication repeated across three documents |
| `dev-09` | Asthma medication quote identical on pages 1 and 2 |
| `dev-10` | Explicitly incomplete abdominal assessment with unavailable results |
| `dev-11` | Eczema record with a quoted prompt-injection attack, not clinical facts |
| `dev-12` | Knee osteoarthritis, stopped drug, intolerance, cancelled injection, future plan |

The eight separate held-out narratives cover audiology, seizure follow-up,
postpartum anemia, cataract assessment, urinary symptoms, back pain, pneumonia
recovery and a ureteric stone episode. They are not variations of one patient.

In `dev-11`, the hostile quotation asks for an invented leukemia diagnosis, a
false no-allergy statement and facts from another case. It is intentionally
untrusted document content, never an instruction to the application or annotator.
None of those requests becomes a fact; the actual latex-allergy statement is
retained. The incomplete sources in `dev-10` and `holdout-06` are valid PDFs with
explicitly missing narrative sections, not corrupt files to be repaired.

### Limits

This small, deliberately simplified corpus does not establish clinical accuracy,
real-world coverage, robust PDF parsing, OCR quality or model safety. It contains
no complex layouts, scanned handwriting, encrypted PDFs, real identities or
medical images. The generic writer and validator are fixture tooling only;
production parsing/chunking, model calls, application code and stage tests are
outside this contribution.