# Resume Fit AI answer audit v1 — first successful run

- Executed: 2026-08-29 14:04:54 +08:00
- Dataset: `eval/questions_resume_fit_ai_answers_v1.jsonl`
- Profile: Hybrid + Cross-Encoder rerank, Top-3, DeepSeek Chat
- Cases: 9 answerable, 3 expected refusals
- Model answer calls: 9
- API errors: 0

## Machine-scored result

| Measure | Result |
|---|---:|
| In-domain source hit | 9 / 9 |
| Retrieval evidence decision | 12 / 12 |
| Correct refusal | 3 / 3 |
| Four-section answer contract | 9 / 9 |
| Citation provenance validity | 12 / 12 |
| Missing citations | 0 |
| Unsupported citations | 0 |
| Initial literal concept-group coverage | 23 / 32 (71.88%) |

Citation provenance means every cited filename came from the supplied retrieval
context. It does not prove that every sentence is entailed by the cited source.

## Post-generation rubric audit

Manual review found that six apparent concept misses were scoring aliases rather
than answer omissions:

- `原始简历` was equivalent to the rubric's `原简历`.
- `覆盖率惩罚` was equivalent to `覆盖惩罚`.
- `初稿` was equivalent to `原稿` / `第一版`.
- `重复的 AI 任务` expressed the required duplicate-task concept.
- `进程本地` was equivalent to `进程内`.
- `共享状态` expressed the required shared-state/queue boundary.

These expressions were added to the post-generation sidecar rubric. A
deterministic rescore of the same saved answers yields 29 / 32 concept groups
(90.63%). No answer was regenerated and no prompt or retrieval setting was
changed after inspecting the outputs.

Three genuine omissions remain visible:

- `resume-answer-007` did not mention output allowlisting.
- `resume-answer-009` did not explicitly mention shared rate-limit state.
- `resume-answer-009` did not explicitly mention distributed cancellation.

The committed compact JSON report preserves the original first-run machine
summary; this audit records the later rubric review rather than overwriting the
historical result.
