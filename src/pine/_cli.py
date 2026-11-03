"""Shared input handling for the capture checkers (``pine.reference``, ``pine.semantics``,
``pine.capture``): plain-language errors pointing at docs/CAPTURE.md."""

from __future__ import annotations

from pathlib import Path

from pine.io import TradingViewExport, load_tradingview_csv
from pine.manifest import Manifest, load_manifest

__all__ = ["InputError", "load_export", "manifest_hint"]


class InputError(ValueError):
    """Bad or missing input files; message is meant for the user as-is."""


def manifest_hint(csv: str | Path) -> str:
    return (
        "create it with:\n"
        f"  uv run python -m pine.capture manifest {csv} --symbol EXCHANGE:SYMBOL "
        "--timeframe TF --timezone AREA/CITY\n"
        "(step-by-step guide: docs/CAPTURE.md), or pass --timeframe instead of --manifest"
    )


def load_export(
    csv: str | Path,
    manifest: str | Path | None = None,
    timeframe: str | None = None,
    tz: str | None = None,
) -> tuple[TradingViewExport, Manifest | None]:
    """Load an export using its manifest (timeframe and time zone) or explicit values."""
    csv = Path(csv)
    if not csv.exists():
        raise InputError(
            f"export not found: {csv}\n"
            "Export it from TradingView first (docs/CAPTURE.md, step 4) and pass its real path."
        )
    m = None
    if manifest is not None:
        mp = Path(manifest)
        if not mp.exists():
            raise InputError(f"manifest not found: {mp}\n{manifest_hint(csv)}")
        m = load_manifest(mp)
        timeframe, tz = m.timeframe, m.timezone
    elif not timeframe:
        raise InputError(f"no --manifest and no --timeframe given.\n{manifest_hint(csv)}")
    return load_tradingview_csv(csv, timeframe=timeframe, session_tz=tz or "UTC"), m
