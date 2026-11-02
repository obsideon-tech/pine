"""Fixture manifests (PINE_PLAN.md §7.1).

Every fixture ``name.csv`` has a sibling ``name.yaml``. Without it a parity result cannot
be interpreted: a different session, adjustment or chart type gives different bars.

Required fields::

    source: synthetic | tradingview_export
    symbol_class: synthetic | public
    symbol: str                 # required when symbol_class == public
    timeframe: str              # Pine spelling, e.g. "60", "1D"
    session: regular | extended
    timezone: str               # IANA zone of the exchange session
    adjustment: none | splits | splits_dividends
    chart_type: candles | bars | line    # Heikin Ashi, Renko, Kagi, ... are rejected
    pine_version: 5 | 6
    export_date: YYYY-MM-DD
    timestamp_convention: bar_open | bar_close
    displayed_precision: int >= 0, or null for synthetic data
    indicator:                  # null when the file holds bars only
      name: str
      inputs: {str: scalar}
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from pine.bars import BarsError, validate_timeframe, validate_timezone

__all__ = ["Manifest", "ManifestError", "load_manifest", "validate_manifest", "check_fixture_dir"]

_ENUMS: dict[str, tuple[Any, ...]] = {
    "source": ("synthetic", "tradingview_export"),
    "symbol_class": ("synthetic", "public"),
    "session": ("regular", "extended"),
    "adjustment": ("none", "splits", "splits_dividends"),
    "chart_type": ("candles", "bars", "line"),
    "pine_version": (5, 6),
    "timestamp_convention": ("bar_open", "bar_close"),
}
_REQUIRED = (
    "source",
    "symbol_class",
    "timeframe",
    "session",
    "timezone",
    "adjustment",
    "chart_type",
    "pine_version",
    "export_date",
    "timestamp_convention",
    "displayed_precision",
    "indicator",
)
_OPTIONAL = ("symbol", "notes")


class ManifestError(ValueError):
    """Raised with every problem found, one per line."""

    def __init__(self, path: str, problems: list[str]):
        self.problems = problems
        super().__init__(f"{path}:\n  " + "\n  ".join(problems))


@dataclass(frozen=True)
class Manifest:
    source: str
    symbol_class: str
    timeframe: str
    session: str
    timezone: str
    adjustment: str
    chart_type: str
    pine_version: int
    export_date: dt.date
    timestamp_convention: str
    displayed_precision: int | None
    indicator: dict[str, Any] | None
    symbol: str | None = None
    notes: str | None = None


def validate_manifest(data: Any, where: str = "<manifest>") -> Manifest:
    problems: list[str] = []
    if not isinstance(data, dict):
        raise ManifestError(where, ["top level must be a mapping"])

    for key in _REQUIRED:
        if key not in data:
            problems.append(f"missing required field {key!r}")
    for key in data:
        if key not in _REQUIRED and key not in _OPTIONAL:
            problems.append(f"unknown field {key!r}")

    for key, allowed in _ENUMS.items():
        if key in data and data[key] not in allowed:
            problems.append(f"{key} = {data[key]!r}; allowed: {', '.join(map(str, allowed))}")

    if "timeframe" in data:
        try:
            validate_timeframe(str(data["timeframe"]))
        except BarsError as exc:
            problems.append(str(exc))
    if "timezone" in data:
        try:
            validate_timezone(data["timezone"])
        except BarsError as exc:
            problems.append(str(exc).replace("session_tz", "timezone"))

    ed = data.get("export_date")
    if "export_date" in data and not isinstance(ed, dt.date):
        try:
            data = {**data, "export_date": dt.date.fromisoformat(str(ed))}
        except ValueError:
            problems.append(f"export_date = {ed!r}; expected YYYY-MM-DD")

    source, sclass = data.get("source"), data.get("symbol_class")
    if sclass == "public" and not data.get("symbol"):
        problems.append("symbol is required when symbol_class is 'public'")
    if source == "synthetic" and sclass != "synthetic":
        problems.append("synthetic source requires symbol_class 'synthetic'")

    if "displayed_precision" in data:
        p = data["displayed_precision"]
        if p is None:
            if source == "tradingview_export":
                problems.append("displayed_precision is required for a TradingView export")
        elif not (isinstance(p, int) and not isinstance(p, bool) and p >= 0):
            problems.append(f"displayed_precision = {p!r}; expected a non-negative integer")

    ind = data.get("indicator")
    if "indicator" in data and ind is not None:
        if not isinstance(ind, dict) or not isinstance(ind.get("name"), str):
            problems.append("indicator must be null or a mapping with a string 'name'")
        elif not isinstance(ind.get("inputs", {}), dict):
            problems.append("indicator.inputs must be a mapping")

    if problems:
        raise ManifestError(where, problems)
    return Manifest(
        **{k: data[k] for k in _REQUIRED},
        **{k: data.get(k) for k in _OPTIONAL},
    )


def load_manifest(path: str | Path) -> Manifest:
    path = Path(path)
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    # PyYAML parses unquoted numbers; timeframe "60" must stay a string.
    if isinstance(data, dict) and isinstance(data.get("timeframe"), int):
        data["timeframe"] = str(data["timeframe"])
    return validate_manifest(data, str(path))


def check_fixture_dir(root: str | Path) -> dict[Path, Manifest]:
    """Validate every ``*.csv`` under ``root`` has a valid sibling ``.yaml``.

    Raises `ManifestError` listing every failing fixture.
    """
    root = Path(root)
    found: dict[Path, Manifest] = {}
    problems: list[str] = []
    for csv_path in sorted(root.rglob("*.csv")):
        man = csv_path.with_suffix(".yaml")
        if not man.exists():
            problems.append(f"{csv_path}: no manifest {man.name}")
            continue
        try:
            found[csv_path] = load_manifest(man)
        except ManifestError as exc:
            problems.extend(f"{man}: {p}" for p in exc.problems)
    if problems:
        raise ManifestError(str(root), problems)
    return found
