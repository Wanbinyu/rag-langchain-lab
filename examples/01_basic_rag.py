"""兼容入口：实际实现位于 src/ 下的可拆分模块。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cli import main


if __name__ == "__main__":
    main()
