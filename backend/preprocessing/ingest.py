"""Phase 1 - data ingestion for the FlyWire FAFB v783 connectome.

Reads raw FlyWire files, maps their columns onto one canonical schema, handles
missing values and writes two normalized Parquet tables:

    neurons.parquet   one row per neuron              (future graph nodes)
    edges.parquet     one row per directed neuron pair (future graph edges)
    ingest_report.json  data-quality summary

Two raw sources are supported and produce the same output schema:

    codex   CSV exports from https://codex.flywire.ai (free login required):
            neurons, classification, consolidated_cell_types,
            connections_princeton, coordinates  (.csv or .csv.gz)
    public  login-free copies of the same v783 release:
            Supplemental_file1_neuron_annotations.tsv  (Schlegel et al. 2024)
            Connectivity_783.parquet                   (Shiu et al. 2024)

Run from the project root:

    python -m backend.preprocessing.ingest --inspect   # look at the raw columns
    python -m backend.preprocessing.ingest             # build processed tables
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import time
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from backend.utils.config import PROCESSED_DIR, RAW_CODEX_DIR, RAW_PUBLIC_DIR, UNKNOWN, VOXEL_SIZE_NM

ID_COLUMNS = ("neuron_id", "source_id", "target_id")
POSITION_COLUMNS = ("x", "y", "z", "soma_x", "soma_y", "soma_z")
CATEGORY_COLUMNS = (
    "flow", "super_class", "cell_class", "cell_sub_class",
    "cell_type", "hemibrain_type", "side", "neurotransmitter",
)

NEURON_COLUMNS = [
    "node_index", "neuron_id", *POSITION_COLUMNS, *CATEGORY_COLUMNS, "nt_confidence",
    "has_annotation", "has_position", "has_soma", "has_connections", "is_outlier",
]
EDGE_COLUMNS = [
    "source_index", "target_index", "source_id", "target_id",
    "synapse_count", "weight", "sign", "main_neuropil",
]

# Codex abbreviates transmitter names; the annotation table spells them out.
NT_NAMES = {
    "ach": "acetylcholine", "gaba": "gaba", "glut": "glutamate",
    "da": "dopamine", "ser": "serotonin", "oct": "octopamine",
}

# Sign convention of Shiu et al. 2024: acetylcholine excites, GABA and glutamate inhibit.
# Modulatory transmitters get 0 (sign unknown). This is a modelling assumption, not a measurement.
NT_SIGN = {"acetylcholine": 1, "gaba": -1, "glutamate": -1}

PUBLIC_ANNOTATIONS = "Supplemental_file1_neuron_annotations.tsv"
PUBLIC_CONNECTIVITY = "Connectivity_783.parquet"

# raw column -> canonical column
PUBLIC_ANNOTATION_COLUMNS = {
    "root_id": "neuron_id",
    "pos_x": "x", "pos_y": "y", "pos_z": "z",
    "soma_x": "soma_x", "soma_y": "soma_y", "soma_z": "soma_z",
    "flow": "flow",
    "super_class": "super_class",
    "cell_class": "cell_class",
    "cell_sub_class": "cell_sub_class",
    "cell_type": "cell_type",
    "hemibrain_type": "hemibrain_type",
    "side": "side",
    "top_nt": "neurotransmitter",
    "top_nt_conf": "nt_confidence",
    "status": "status",
}
# The file's own "Excitatory" column is ignored: it marks dopamine, serotonin and octopamine as
# excitatory. Both sources derive the sign from NT_SIGN instead, so they stay consistent.
PUBLIC_CONNECTIVITY_COLUMNS = {
    "Presynaptic_ID": "source_id",
    "Postsynaptic_ID": "target_id",
    "Connectivity": "synapse_count",
}

# canonical column -> raw names accepted for it (Codex has renamed columns between releases)
CODEX_COLUMNS = {
    "neurons": {
        "neuron_id": ["root_id"],
        "neurotransmitter": ["nt_type"],
        "nt_confidence": ["nt_type_score"],
    },
    "classification": {
        "neuron_id": ["root_id"],
        "flow": ["flow"],
        "super_class": ["super_class"],
        "cell_class": ["class", "cell_class"],
        "cell_sub_class": ["sub_class", "cell_sub_class"],
        "cell_type": ["cell_type"],
        "hemibrain_type": ["hemibrain_type"],
        "side": ["side"],
    },
    "consolidated_cell_types": {
        "neuron_id": ["root_id"],
        "consolidated_type": ["primary_type", "cell_type"],
    },
    "connections_princeton": {
        "source_id": ["pre_root_id", "pre_pt_root_id"],
        "target_id": ["post_root_id", "post_pt_root_id"],
        "synapse_count": ["syn_count", "synapse_count"],
        "neuropil": ["neuropil"],
    },
    "coordinates": {
        "neuron_id": ["root_id"],
        "position": ["position"],
    },
}
CODEX_REQUIRED = {"neuron_id", "source_id", "target_id", "synapse_count", "position"}


@dataclass
class RawTables:
    """Raw data renamed to canonical columns, before cleaning."""

    source: str
    neurons: pl.LazyFrame  # neuron_id, x, y, z in nm, plus optional attributes
    edges: pl.LazyFrame  # source_id, target_id, synapse_count [, sign] [, neuropil]
    column_mapping: dict
    coordinate_note: str


# ---------------------------------------------------------------------------
# Reading raw sources
# ---------------------------------------------------------------------------

def _voxels_to_nm(prefix: str = "") -> list[pl.Expr]:
    return [
        (pl.col(f"{prefix}{axis}") * size).cast(pl.Float32).alias(f"{prefix}{axis}")
        for axis, size in zip("xyz", VOXEL_SIZE_NM)
    ]


def _require_files(paths: list[Path], hint: str) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing raw files: {missing}. {hint}")


def read_public(raw_dir: Path = RAW_PUBLIC_DIR) -> RawTables:
    annotations_path = raw_dir / PUBLIC_ANNOTATIONS
    connectivity_path = raw_dir / PUBLIC_CONNECTIVITY
    _require_files([annotations_path, connectivity_path], "Run: python -m backend.preprocessing.download")

    neurons = (
        pl.scan_csv(
            annotations_path, separator="\t", infer_schema_length=None,
            schema_overrides={"root_id": pl.Int64},
        )
        .select(list(PUBLIC_ANNOTATION_COLUMNS))
        .rename(PUBLIC_ANNOTATION_COLUMNS)
        .with_columns(*_voxels_to_nm(), *_voxels_to_nm("soma_"))
    )
    edges = (
        pl.scan_parquet(connectivity_path)
        .select(list(PUBLIC_CONNECTIVITY_COLUMNS))
        .rename(PUBLIC_CONNECTIVITY_COLUMNS)
    )
    return RawTables(
        source="public",
        neurons=neurons,
        edges=edges,
        column_mapping={PUBLIC_ANNOTATIONS: PUBLIC_ANNOTATION_COLUMNS, PUBLIC_CONNECTIVITY: PUBLIC_CONNECTIVITY_COLUMNS},
        coordinate_note="pos_* and soma_* are FAFB voxels (4x4x40 nm), converted to nm",
    )


def _find_csv(directory: Path, stem: str) -> Path | None:
    for suffix in (".csv", ".csv.gz"):
        path = directory / f"{stem}{suffix}"
        if path.exists():
            return path
    return None


def _csv_header(path: Path) -> list[str]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig", newline="") as handle:
        return next(csv.reader(handle))


def _scan_any_csv(path: Path, **kwargs) -> pl.LazyFrame:
    # polars can only stream plain CSV; gzip files are decompressed in memory.
    if path.suffix == ".gz":
        return pl.read_csv(path, **kwargs).lazy()
    return pl.scan_csv(path, **kwargs)


def _read_codex_table(path: Path, aliases: dict[str, list[str]]) -> tuple[pl.LazyFrame, dict[str, str]]:
    header = _csv_header(path)
    rename, absent = {}, []
    for canonical, candidates in aliases.items():
        found = next((c for c in candidates if c in header), None)
        if found:
            rename[found] = canonical
        elif canonical in CODEX_REQUIRED:
            raise ValueError(f"{path.name}: no column for '{canonical}' (tried {candidates}); columns present: {header}")
        else:
            absent.append(canonical)

    overrides = {raw: pl.Int64 for raw, canonical in rename.items() if canonical in ID_COLUMNS}
    if path.suffix == ".gz":
        frame = pl.read_csv(path, columns=list(rename), schema_overrides=overrides).lazy()
    else:
        frame = pl.scan_csv(path, schema_overrides=overrides).select(list(rename))
    frame = frame.rename(rename).with_columns(pl.lit(None).alias(column) for column in absent)
    return frame, rename


def codex_files_present(raw_dir: Path = RAW_CODEX_DIR) -> bool:
    return all(_find_csv(raw_dir, name) for name in CODEX_COLUMNS)


def read_codex(raw_dir: Path = RAW_CODEX_DIR) -> RawTables:
    paths = {name: _find_csv(raw_dir, name) for name in CODEX_COLUMNS}
    missing = [f"{name}.csv" for name, path in paths.items() if path is None]
    if missing:
        raise FileNotFoundError(f"Missing Codex files in {raw_dir}: {missing}")

    tables, mapping = {}, {}
    for name, path in paths.items():
        tables[name], mapping[path.name] = _read_codex_table(path, CODEX_COLUMNS[name])

    # "position" looks like "[x y z]"; a neuron may have several rows, keep the first.
    coordinates = (
        tables["coordinates"].collect()
        .with_columns(pl.col("position").cast(pl.String).str.extract_all(r"-?\d+(?:\.\d+)?").alias("xyz"))
        .filter(pl.col("xyz").list.len() == 3)
        .with_columns(pl.col("xyz").list.get(i).cast(pl.Float64).alias(axis) for i, axis in enumerate("xyz"))
        .group_by("neuron_id", maintain_order=True)
        .first()
        .select("neuron_id", "x", "y", "z")
    )
    # Positions have been published both in voxels and in nm. Brain depth tells them apart:
    # FAFB is ~7,000 sections deep in voxels, ~280,000 nm deep in nanometres.
    in_voxels = (coordinates["z"].max() or 0) < 20_000
    if in_voxels:
        coordinates = coordinates.with_columns(_voxels_to_nm())
    else:
        coordinates = coordinates.with_columns(pl.col(axis).cast(pl.Float32) for axis in "xyz")

    neurons = (
        coordinates.lazy()
        .join(
            tables["neurons"].with_columns(
                pl.col("neurotransmitter").cast(pl.String).str.to_lowercase().replace(NT_NAMES)
            ),
            on="neuron_id", how="full", coalesce=True,
        )
        .join(tables["classification"], on="neuron_id", how="full", coalesce=True)
        .join(tables["consolidated_cell_types"], on="neuron_id", how="full", coalesce=True)
        .with_columns(pl.coalesce(pl.col("consolidated_type").cast(pl.String), pl.col("cell_type").cast(pl.String)).alias("cell_type"))
        .drop("consolidated_type")
    )
    return RawTables(
        source="codex",
        neurons=neurons,
        edges=tables["connections_princeton"],
        column_mapping=mapping,
        coordinate_note=f"coordinates.csv detected in {'voxels, converted to nm' if in_voxels else 'nm'}",
    )


def resolve_source(source: str, raw_codex: Path = RAW_CODEX_DIR) -> str:
    if source == "auto":
        return "codex" if codex_files_present(raw_codex) else "public"
    return source


# ---------------------------------------------------------------------------
# Cleaning and normalization
# ---------------------------------------------------------------------------

def normalize(raw: RawTables) -> tuple[pl.DataFrame, pl.DataFrame, dict]:
    """Turn canonical raw tables into the final neuron and edge tables plus a quality report."""
    report: dict = {"source": raw.source, "column_mapping": raw.column_mapping, "coordinates": raw.coordinate_note}

    # --- edges: keep valid rows, merge rows describing the same neuron pair ---
    edge_columns = raw.edges.collect_schema().names()
    raw_edges = raw.edges.with_columns(
        pl.col("source_id", "target_id").cast(pl.Int64),
        pl.col("synapse_count").cast(pl.Int64),
    )
    valid = pl.col("source_id").is_not_null() & pl.col("target_id").is_not_null() & (pl.col("synapse_count") > 0)
    row_counts = raw_edges.select(pl.len().alias("rows"), valid.sum().alias("valid")).collect()
    raw_rows, valid_rows = int(row_counts["rows"][0]), int(row_counts["valid"][0])

    aggregations = [pl.col("synapse_count").sum()]
    if "neuropil" in edge_columns:
        # Codex splits a pair by brain region; remember the region holding most synapses.
        aggregations.append(pl.col("neuropil").sort_by("synapse_count", descending=True).first().alias("main_neuropil"))
    if "sign" in edge_columns:
        aggregations.append(pl.col("sign").first())
    edges = raw_edges.filter(valid).group_by("source_id", "target_id").agg(aggregations).collect(engine="streaming")

    # --- neurons: every annotated neuron plus every neuron that appears in an edge ---
    annotations = raw.neurons.collect()
    for column in (*POSITION_COLUMNS, "status", "nt_confidence", *CATEGORY_COLUMNS):
        if column not in annotations.columns:
            annotations = annotations.with_columns(pl.lit(None).alias(column))
    duplicate_neuron_rows = annotations.height - annotations["neuron_id"].n_unique()
    annotations = (
        annotations.filter(pl.col("neuron_id").is_not_null())
        .unique("neuron_id", keep="first", maintain_order=True)
        .with_columns(pl.lit(True).alias("has_annotation"))
    )
    connected = (
        pl.concat([edges.select(pl.col("source_id").alias("neuron_id")), edges.select(pl.col("target_id").alias("neuron_id"))])
        .unique()
        .with_columns(pl.lit(True).alias("has_connections"))
    )
    neurons = (
        pl.concat([annotations.select("neuron_id"), connected.select("neuron_id")])
        .unique()
        .join(annotations, on="neuron_id", how="left")
        .join(connected, on="neuron_id", how="left")
        .sort("neuron_id")
    )
    missing_before_fill = {
        column: int(neurons[column].null_count())
        for column in (*POSITION_COLUMNS, *CATEGORY_COLUMNS, "nt_confidence")
    }
    neurons = neurons.with_columns(
        pl.int_range(pl.len(), dtype=pl.UInt32).alias("node_index"),
        pl.col("has_annotation", "has_connections").fill_null(False),
        pl.all_horizontal(pl.col("x", "y", "z").is_not_null()).alias("has_position"),
        pl.col("soma_x").is_not_null().alias("has_soma"),
        pl.col("status").is_not_null().alias("is_outlier"),
        *(pl.col(column).cast(pl.String).fill_null(UNKNOWN) for column in CATEGORY_COLUMNS),
        pl.col(*POSITION_COLUMNS, "nt_confidence").cast(pl.Float32),
    ).select(NEURON_COLUMNS)

    # --- edges: attach node indices, sign and normalized weight ---
    if "sign" not in edges.columns:
        presynaptic_sign = neurons.select(
            pl.col("neuron_id").alias("source_id"),
            pl.col("neurotransmitter").replace_strict(NT_SIGN, default=0, return_dtype=pl.Int8).alias("sign"),
        )
        edges = edges.join(presynaptic_sign, on="source_id", how="left")
    if "main_neuropil" not in edges.columns:
        edges = edges.with_columns(pl.lit(None, dtype=pl.String).alias("main_neuropil"))

    index = neurons.select("neuron_id", "node_index")
    edges = (
        edges.join(index.rename({"neuron_id": "source_id", "node_index": "source_index"}), on="source_id")
        .join(index.rename({"neuron_id": "target_id", "node_index": "target_index"}), on="target_id")
        .with_columns(
            pl.col("synapse_count").cast(pl.Int32),
            # weight = share of the target neuron's total input carried by this connection
            (pl.col("synapse_count") / pl.col("synapse_count").sum().over("target_id")).cast(pl.Float32).alias("weight"),
            pl.col("sign").fill_null(0).cast(pl.Int8),
            pl.col("main_neuropil").cast(pl.String).fill_null(UNKNOWN),
        )
        .sort("source_index", "target_index")
        .select(EDGE_COLUMNS)
    )

    counts = edges["synapse_count"]
    report["neurons"] = {
        "total": neurons.height,
        "with_annotation": int(neurons["has_annotation"].sum()),
        "only_in_connectivity": int((~neurons["has_annotation"]).sum()),
        "with_position": int(neurons["has_position"].sum()),
        "with_soma": int(neurons["has_soma"].sum()),
        "with_connections": int(neurons["has_connections"].sum()),
        "without_connections": int((~neurons["has_connections"]).sum()),
        "flagged_outliers": int(neurons["is_outlier"].sum()),
        "duplicate_rows_dropped": duplicate_neuron_rows,
        "missing_values_before_fill": missing_before_fill,
        "super_class_counts": dict(neurons["super_class"].value_counts(sort=True).iter_rows()),
        "neurotransmitter_counts": dict(neurons["neurotransmitter"].value_counts(sort=True).iter_rows()),
    }
    report["edges"] = {
        "raw_rows": raw_rows,
        "invalid_rows_dropped": raw_rows - valid_rows,
        "rows_merged_into_same_pair": valid_rows - edges.height,
        "total": edges.height,
        "total_synapses": int(counts.sum()),
        "self_loops": int((edges["source_id"] == edges["target_id"]).sum()),
        "synapse_count_quantiles": {
            **{f"p{int(q * 100)}": float(counts.quantile(q)) for q in (0.25, 0.5, 0.75, 0.9, 0.99)},
            "max": int(counts.max()),
        },
        "edges_with_min_synapses": {str(t): int((counts >= t).sum()) for t in (1, 2, 3, 5, 10, 20)},
        "sign_counts": {
            "excitatory": int((edges["sign"] > 0).sum()),
            "inhibitory": int((edges["sign"] < 0).sum()),
            "unknown": int((edges["sign"] == 0).sum()),
        },
    }
    return neurons, edges, report


def build(
    source: str = "auto",
    raw_public: Path = RAW_PUBLIC_DIR,
    raw_codex: Path = RAW_CODEX_DIR,
    out_dir: Path = PROCESSED_DIR,
) -> dict:
    started = time.perf_counter()
    source = resolve_source(source, raw_codex)
    raw = read_codex(raw_codex) if source == "codex" else read_public(raw_public)
    neurons, edges, report = normalize(raw)

    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {"neurons": out_dir / "neurons.parquet", "edges": out_dir / "edges.parquet"}
    neurons.write_parquet(outputs["neurons"], compression="zstd")
    edges.write_parquet(outputs["edges"], compression="zstd")

    report["output_mb"] = {name: round(path.stat().st_size / 1e6, 1) for name, path in outputs.items()}
    report["seconds"] = round(time.perf_counter() - started, 1)
    (out_dir / "ingest_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


# ---------------------------------------------------------------------------
# Inspection
# ---------------------------------------------------------------------------

def inspect_raw(source: str = "auto", raw_public: Path = RAW_PUBLIC_DIR, raw_codex: Path = RAW_CODEX_DIR) -> None:
    """Print row count, column names, dtypes, null counts and sample rows of every raw file."""
    source = resolve_source(source, raw_codex)
    if source == "codex":
        frames = {path.name: _scan_any_csv(path) for name in CODEX_COLUMNS if (path := _find_csv(raw_codex, name))}
    else:
        frames = {
            PUBLIC_ANNOTATIONS: pl.scan_csv(raw_public / PUBLIC_ANNOTATIONS, separator="\t", infer_schema_length=None),
            PUBLIC_CONNECTIVITY: pl.scan_parquet(raw_public / PUBLIC_CONNECTIVITY),
        }

    with pl.Config(tbl_formatting="ASCII_MARKDOWN", tbl_cols=10, fmt_str_lengths=25, tbl_width_chars=200):
        for name, frame in frames.items():
            schema = frame.collect_schema()
            stats = frame.select(pl.len().alias("__rows__"), pl.all().null_count()).collect()
            print(f"\n=== {name}: {stats['__rows__'][0]:,} rows, {len(schema)} columns")
            for column, dtype in schema.items():
                print(f"  {column:<28} {str(dtype):<8} nulls={stats[column][0]:,}")
            print(frame.head(3).collect())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=["auto", "public", "codex"], default="auto")
    parser.add_argument("--inspect", action="store_true", help="only print raw columns, dtypes and null counts")
    args = parser.parse_args()

    if args.inspect:
        inspect_raw(args.source)
        return
    report = build(args.source)
    summary = {key: report[key] for key in ("source", "coordinates", "neurons", "edges", "output_mb", "seconds")}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
