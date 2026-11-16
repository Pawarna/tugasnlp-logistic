"""Perakitan seluruh alur laporan."""
from __future__ import annotations

from scripts.rpt import bab3, bab4, bab5, back, bab12, front
from scripts.rpt.engine import page_break


def assemble(ctx) -> None:
    """Susun seluruh flowable laporan sesuai urutan bab."""
    front.build(ctx)                 # cover s.d. daftar gambar (sudah diakhiri PageBreak)

    bab12.bab1(ctx)
    ctx.add(page_break())
    bab12.bab2(ctx)
    ctx.add(page_break())

    bab3.build(ctx)
    ctx.add(page_break())

    bab4.build(ctx)
    ctx.add(page_break())

    bab5.build(ctx)
    back.build(ctx)                  # diawali PageBreak sendiri