"""Read the Phase 1 tables lazily, so later phases load only the columns and rows they need."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from backend.utils.config import PROCESSED_DIR


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python -m backend.preprocessing.ingest")
    return path


def scan_neurons(processed_dir: Path = PROCESSED_DIR) -> pl.LazyFrame:
    return pl.scan_parquet(_require(processed_dir / "neurons.parquet"))


def scan_edges(processed_dir: Path = PROCESSED_DIR) -> pl.LazyFrame:
    return pl.scan_parquet(_require(processed_dir / "edges.parquet"))


def load_neurons(columns: list[str] | None = None, processed_dir: Path = PROCESSED_DIR) -> pl.DataFrame:
    frame = scan_neurons(processed_dir)
    return (frame.select(columns) if columns else frame).collect()


def load_edges(
    min_synapses: int = 1,
    columns: list[str] | None = None,
    processed_dir: Path = PROCESSED_DIR,
) -> pl.DataFrame:
    frame = scan_edges(processed_dir).filter(pl.col("synapse_count") >= min_synapses)
    return (frame.select(columns) if columns else frame).collect()
