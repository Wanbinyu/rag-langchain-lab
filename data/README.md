# Local knowledge base

Place `.txt` or `.md` documents that you are authorized to use in this directory. Subdirectories are supported.

Knowledge-base files are intentionally ignored by Git because they may contain copyrighted, private, or sensitive material. This README is the only tracked file under `data/`.

Optional corpus-side retrieval glossary: put `retrieval_glossary.json` in this directory (or set `QUERY_GLOSSARY_PATH`). The file is not indexed as a document. A Graham essays glossary lives at [`eval/glossaries/graham.json`](../eval/glossaries/graham.json).

For a tracked, non-sensitive example corpus and isolated transfer evaluation, see [`examples/resume_fit_ai/`](../examples/resume_fit_ai/). It is indexed from its own directory and never mixed into this local knowledge base.
