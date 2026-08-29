# Graham development-set failure audit — 2026-08-29

Scope: the 85-case Graham development/calibration regression in
`eval/questions_graham_85.jsonl`. This is a diagnostic record, not a blind-test
report. No retrieval rule, query alias, or threshold was changed from this
audit.

| Case | Failure | Observation | Disposition |
|---|---|---|---|
| `graham-062` | Source miss | The question about “competition” retrieved `219_superlinear-returns.md`, `051_ideas-for-startups.md`, and `129_the-future-of-startup-funding.md`, while the gold source named only `218_how-to-do-great-work.md`. | Gold/source ambiguity; review the annotation before treating it as a retrieval defect. |
| `graham-063` | Source miss | The early-startup competition question retrieved other directly startup-related essays rather than the two annotated files. | Gold/source ambiguity; do not add a question-specific rewrite. |
| `graham-058` | False refusal | Relevant passages about preserving innocent boldness were retrieved, but the top rerank score (0.2739) left the evidence status at `insufficient`. | Conservative false refusal. A global threshold reduction would weaken unrelated refusals, so leave it visible. |
| `graham-067` | False acceptance | A current-status question about managing Y Combinator was accepted from historical and autobiographical documents. | Static-corpus freshness boundary error. |
| `graham-080` | False acceptance | A question asking which articles were published “this year” was accepted as `review` from a static corpus. | Static-corpus freshness boundary error. |

The two freshness failures support a general future design question—how a
static corpus should declare its effective date—but this round deliberately
does not introduce temporal regexes or per-question exceptions.
