# VisitBrief — slide outline

Twelve short slides. Say the numbers from `docs/evaluation.md` only. Do not
add a claim that the tool saves clinician time.

1. **Title.** VisitBrief. Source-linked preparation for a fictional visit.
   Track 2, Health and Wellbeing. AMD Token Factory on Radeon Cloud.
2. **User.** A staff member about to open a short packet of notes, not a
   patient looking for advice.
3. **Problem.** The useful lines are scattered across a few documents. A
   summary that cannot be checked is not usable in this setting.
4. **Product.** Pick a case. See documented problems, medicines, allergies,
   events and recorded plans. Each line opens the page and the quote.
5. **Uncertainty.** Undated facts stay off the timeline. Missing allergy text
   is a flag, not a negative finding. Conflicting doses are both kept.
6. **Boundary.** Draft for review. No diagnosis, no treatment change, no
   upload, no free-form prompt, fictional cases only.
7. **AMD.** The live call is `DeepSeek-V4-Flash` on AMD Token Factory.
   Radeon Cloud states that these catalog models run on AMD GPUs. The 14 October
   Instinct MI300X notebook, Qwen2.5-7B-Instruct, BF16, is down (HTTP 502). Its
   Llama-Factory tutorial was not the extractor.
8. **Pipeline.** PDF text to source blocks, one model call, then quote checks.
   A sentence that is not in its block never reaches the screen. A rewritten
   value is replaced by the quote. An instruction to invent a diagnosis is dropped.
9. **Measurement.** One Token Factory pass on 15 October. Development recall of
   curated facts 27/58 (46.6%). Held-out recall 34/43 (79.1%). 17 of 20 cases
   incomplete. Median endpoint time 20.7 s. The endpoint is not run-to-run stable.
   Not a clinical trial. The older notebook pass was 13/58 and 8/43.
10. **Demo.** `dev-01` routine packet, `dev-02` disagreeing doses, `dev-11`
    hostile text that must not become a fact.
11. **Limits.** Small English corpus, no OCR, no EHR, one request at a time,
    shared team rate limit, recall below the project's own target, no public
    judge URL while the notebook is down.
12. **Close.** The person still decides. AMD Token Factory runs the extraction.
    The interface shows what was found, what is missing, and where it came from.
