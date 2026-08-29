# Resume Fit AI frozen v1 — first and only tuning-excluded run

- Executed: 2026-08-29 11:41:32 +08:00
- Dataset: `eval/questions_resume_fit_ai_frozen_v1.jsonl`
- Locked SHA-256: `1437f57c405ab50c01f91e5cbf4ac0fa8bfe02baa9a421319db26651b85e4f6e`
- Profile: Hybrid + Cross-Encoder rerank, Top-3, six-candidate budget
- Policy: the cases and results below were not used for any subsequent retrieval,
  threshold, rewrite, alias, or prompt tuning.

## Result

| Measure | Result |
|---|---:|
| In-domain source hit | 20 / 20 |
| MRR | 1.0000 |
| Out-of-scope rejection | 8 / 10 |
| Overall evidence decision | 24 / 30 |
| Retrieval P50 / P95 | 1086.3 / 3635.9 ms |

## Six decision failures

Four in-domain questions (`resume-frozen-006`, `007`, `010`, `018`) retrieved
the annotated source at rank 1 but were conservatively marked `insufficient`
with rerank scores from 0.2467 to 0.2949. These are evidence-gating false
refusals rather than source-retrieval misses.

Two out-of-scope questions were accepted as `review`:

- `resume-frozen-023` asks the knowledge-base assistant to perform a resume
  rewrite. Relevant product documentation was found, but documentation about a
  capability is not authorization or sufficient input to execute that action.
- `resume-frozen-029` asks for the current paid-user count. Product and
  operations documents were found, but a static technical corpus cannot supply
  live business data.

The compact machine-readable report is
`eval/reports/resume_fit_ai_frozen_v1.json`. No post-run tuning was performed.
