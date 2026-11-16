"""Bangun laporan PDF.

Jalankan dari direktori proyek:

    python -m scripts.build_report
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from scripts.rpt.engine import build_pdf  # noqa: E402


def main() -> None:
    started = time.time()
    print("[report] menyusun laporan ...")
    output = build_pdf(config.REPORT_PDF)
    size = output.stat().st_size / 1024
    print(f"[report] selesai dalam {time.time() - started:.1f} detik")
    print(f"[report] {output}  ({size:.0f} KB)")


if __name__ == "__main__":
    main()