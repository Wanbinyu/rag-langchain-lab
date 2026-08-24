"""Repair legacy expected_keywords labels in the 150-question JSONL set.

Some imported rows contain literal ``?`` replacement characters in the old
one-of keyword list.  The curated required groups are the reviewed source of
truth; this script only refreshes the legacy compatibility field and keeps all
questions, sources, answers, and abstention labels unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path

from rebuild_answer_keywords import DATASET_PATH, OUTPUT_PATH, parse_rubric


def main() -> None:
    rubric = parse_rubric()
    rows = [
        json.loads(line)
        for line in DATASET_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    repaired = 0
    for row in rows:
        case_id = str(row["id"])
        if bool(row.get("expect_abstention", False)):
            keywords: list[str] = []
        else:
            keywords = [group[0] for group in rubric[case_id]]
        old_keywords = row.get("expected_keywords", [])
        if old_keywords != keywords:
            repaired += 1
        row["expected_keywords"] = keywords
        if case_id == "graham-084":
            row["annotation_note"] = (
                "该题是电商运营问题，与 Paul Graham 知识库无关；保留用于暴露题集标注异常。"
            )

    DATASET_PATH.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) for row in rows)
        + "\n",
        encoding="utf-8",
    )
    print(f"修复 {DATASET_PATH}：{repaired}/{len(rows)} 行 expected_keywords")


if __name__ == "__main__":
    main()
