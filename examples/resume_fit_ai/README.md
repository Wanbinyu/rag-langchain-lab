# Resume Fit AI transfer evaluation

This example is a small, first-party knowledge base extracted from the architecture and operating boundaries of Resume Fit AI. It exists to check whether the generic retrieval pipeline transfers beyond the Paul Graham essay corpus.

The example deliberately contains no user resumes, job descriptions, API keys, private source code, production hostnames, or generated model output.

## Run it

From the repository root:

```powershell
# Build an isolated index and run the full hybrid + reranker profile.
.\.venv\Scripts\python.exe examples\02_resume_fit_ai_eval.py

# Compare dense retrieval, hybrid retrieval, and hybrid + reranker.
.\.venv\Scripts\python.exe examples\02_resume_fit_ai_eval.py --mode all

# Regenerate the tracked, compact experiment summary.
.\.venv\Scripts\python.exe examples\02_resume_fit_ai_eval.py `
  --mode all --publish-summary

# Run the small answer/structure/citation/refusal audit. A summary is published
# only if every required Chat API call succeeds.
.\.venv\Scripts\python.exe examples\02_resume_fit_ai_eval.py `
  --dataset answers --mode rerank --with-answers --publish-summary
```

If a configured Hugging Face mirror cannot serve the ONNX files, retry the process with the official endpoint:

```powershell
$env:HF_ENDPOINT = "https://huggingface.co"
.\.venv\Scripts\python.exe examples\02_resume_fit_ai_eval.py --mode all
```

The isolated Chroma database is written under `work/resume-fit-ai-transfer/` and normal detailed reports are written under `eval/results/`; both locations are ignored by Git. The compact summary produced by `--publish-summary` is written to `eval/reports/resume_fit_ai_transfer_v1.json`.

This is a transfer/acceptance dataset, not a blind test. Its questions and expected sources are public and should not be used to claim unseen generalization.

The repository also contains a 30-case prospective frozen set. Its SHA-256 and
one-run policy are recorded under `eval/frozen/`. The first run achieved 20/20
in-domain Top-3 hits, MRR 1.0000, 8/10 out-of-scope rejections, and 24/30
overall evidence decisions. Four in-domain cases were conservative false
refusals even though the expected source ranked first; two boundary cases were
false acceptances. The test was not used for post-run tuning.

## Recorded result

On the tracked 22-case dataset (18 in-domain and 4 out-of-scope), the local Top-3 run produced:

| Mode | Recall@3 | MRR | Evidence decision accuracy |
|---|---:|---:|---:|
| Dense | 100% | 0.9259 | 18.18% (rejects every case) |
| Dense + BM25/RRF | 100% | 0.9444 | 18.18% (rejects every case) |
| Dense + BM25/RRF + Cross-Encoder | 100% | 1.0000 | 100% (22/22) |

The evidence decision is correct only when an in-domain question is accepted or an out-of-scope question is rejected. This prevents an “always refuse” policy from looking good. All three modes use the same six-candidate budget for a fair comparison; the five-document corpus does not need the general-purpose maximum of 18. The tracked JSON summary includes latency and configuration details. Cross-Encoder latency includes cold-start/model-loading cost and varies by hardware and cache state.
