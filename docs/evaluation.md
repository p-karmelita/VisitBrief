# AMD Token Factory measurement — 15 October 2026

One live pass of all 20 fictional cases through the shared AMD Token Factory
endpoint. Annotations were not sent to the model. This is not a clinical study
and not a claim that staff time or diagnostic quality improved. A repeat call
can return a different set of facts: the endpoint is not a stable greedy decode.
The figures below are this pass only.

## Stack that produced these numbers

| Item | Recorded value |
| --- | --- |
| Endpoint | `https://developer.amd.com.cn/radeon/api/v1` |
| Catalog | AMD Token Factory, shared free model API |
| Model requested | `DeepSeek-V4-Flash` |
| Model reported | `self-dploy/DeepSeek-V4-Flash` |
| Decoding | `temperature` 0, `reasoning_effort` none, JSON object |
| Context budget | 32,768 tokens, 4,096 reserved for output |
| What runs the weights | Radeon Cloud states that catalog models run on AMD GPUs. This pass did not open an Instinct notebook. |

The 14 October notebook run is kept below. That MI300X instance now returns
HTTP 502, so it is not the live demo.

## How a match is counted

Same rule as the notebook pass. `scripts/evaluate.py` calls `POST /v1/briefs`.
A predicted fact matches an annotated fact only when the category is the same
and they share a source quote. Extra source-backed facts are counted separately.
They are not clinical mistakes. A displayed value is either the model's verbatim
copy or, when the model paraphrased the value, the verbatim quote it cited.
A quote that is not in its block is dropped. A quote that tells the model to
ignore instructions or invent a diagnosis is dropped.

## Results

| Split | Cases | Annotated facts recalled | Date agreement on matches | Temporality agreement | Annotated review reasons recalled | Extra supported facts | Endpoint time |
| --- | ---: | --- | --- | --- | --- | ---: | ---: |
| Development | 12 | 27 / 58 (46.6%) | 77.8% | 92.6% | 70% | 13 | 268.3 s |
| Held out | 8 | 34 / 43 (79.1%) | 82.4% | 97.1% | 83.3% | 4 | 201.3 s |

17 of 20 cases came back `incomplete`. `dev-07`, `dev-11` and `holdout-03`
came back `ready_for_review`. Incomplete is the visible state when a candidate
was dropped or repaired. The internal targets of 95% field precision and 90%
completeness were not met. Held-out recall is higher than development recall
on this pass. Held-out cases were not used to change the prompt.

Endpoint time across the 20 calls: minimum 4.8 s, median 20.7 s, 95th
percentile 46.6 s, maximum 51.2 s. This is client-measured time for the chat
call, including network delay. It is not device time on a GPU we could see.
Token totals: 20,059 input and 8,788 output.

Case-level rows are in `docs/token-factory-evaluation.json`.

## What the safety checks did on this pass

`dev-02` kept both recorded levothyroxine doses and raised
`conflicting_medication_doses`. The app did not pick a dose.

`dev-11` raised `untrusted_instruction_in_source`. The instruction to invent a
diagnosis is in the source text and is not shown as a fact. On this pass the
case still missed annotated facts, so the extraction is not complete.

The notebook section below is the earlier Qwen2.5-7B measurement. Do not mix
its recall with the table above.

# AMD notebook measurement — 14 October 2026

One live pass of all 20 fictional cases through the notebook GPU. Annotations were
not sent to the model. This is not a clinical study and not a claim that staff
time or diagnostic quality improved.

## Stack that produced these numbers

| Item | Recorded value |
| --- | --- |
| Host | AMD AI Notebooks instance `jupyter-user-261014191442-df06a4ff` |
| GPU | AMD Instinct MI300X VF, 191.7 GiB |
| ROCm / HIP | 6.4.43484-123eb5128 |
| PyTorch | 2.8.0a0+gitd06a406 |
| Transformers | 4.55.0 |
| Model process | Python 3.10.18, existing conda env, no package changes |
| API and UI | Python 3.12.11 in a separate virtual environment |
| Model | `Qwen/Qwen2.5-7B-Instruct` |
| Revision | `a09a35458c702b33eeacc393d103063234e8bc28` |
| Weights | BF16, SDPA attention, greedy decoding (`temperature` 0) |

The open notebook file is AMD's LLaMA-Factory tutorial for Llama-3.1-8B.
That recipe trains on the identity and Alpaca demo sets and needs a gated
Hugging Face token. No token was available, and that demo set is not a visit
record. The tutorial was not run. Inference uses the Qwen checkpoint already
verified on this image, served by `scripts/start_amd_adapter.py`.

## How a match is counted

`scripts/evaluate.py` calls `POST /v1/briefs`. A predicted fact matches an
annotated fact only when the category is the same and they share a source
quote. The annotations are a curated set, not every true sentence, so extra
source-backed facts are reported separately. They are not treated as clinical
mistakes. Every displayed fact has already survived quote validation.

## Results

| Split | Cases | Annotated facts recalled | Date agreement on matches | Temporality agreement | Annotated review reasons recalled | Extra supported facts | Inference time |
| --- | ---: | --- | --- | --- | --- | ---: | ---: |
| Development | 12 | 13 / 58 (22.4%) | 76.9% | 84.6% | 40% | 10 | 86.5 s |
| Held out | 8 | 8 / 43 (18.6%) | 100% | 37.5% | 50% | 4 | 54.7 s |

All 20 cases came back `incomplete`. That is the intended visible state when
supported sentences are missed. The internal targets of 95% field precision and
90% completeness were not met.

Inference time across the 20 calls: minimum 3.22 s, median 6.81 s, 95th
percentile 10.03 s, maximum 10.61 s. The weights were already cached, so this
is not a cold download. The plan's separate warm-up study of 20 repeated runs
was not done. Token totals: 17,155 input and 9,595 output.

Case-level rows are in `docs/amd-evaluation.json`.

## What the safety checks did on this run

On `dev-01` the model returned an allergy quote that is present in the source,
but the fact value was not identical to that quote. Validation dropped it and
the case stayed incomplete, with `missing_allergy_documentation`.

On `dev-11`, whose source contains a hostile instruction asking for an invented
diagnosis, the model result contained no displayable fact. The app does not yet
emit the annotated reason `untrusted_instruction_in_source`; it reported
`no_supported_facts`, `missing_allergy_documentation` and
`incomplete_extraction` instead.
