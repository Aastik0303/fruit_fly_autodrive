"""In-memory connectome index used by the API.

Edges are held as NumPy arrays in CSR (compressed sparse row) form. Because edges.parquet is
sorted by source_index, the outgoing edges of neuron i are rows out_offsets[i] .. out_offsets[i+1].
Incoming edges use the same trick on a second ordering sorted by target_index.
Looking up any neuron's partners is then a slice instead of a scan over 15 million rows.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import polars as pl

from backend.preprocessing.loaders import load_edges, load_neurons
from backend.utils.config import PROCESSED_DIR

METADATA_FIELDS = (
    "cell_type", "super_class", "cell_class", "cell_sub_class",
    "hemibrain_type", "flow", "side", "neurotransmitter",
)


def _offsets(indices: np.ndarray, size: int) -> np.ndarray:
    return np.concatenate(([0], np.cumsum(np.bincount(indices, minlength=size))))


class ConnectomeStore:
    def __init__(self, processed_dir: Path = PROCESSED_DIR):
        self.processed_dir = processed_dir
        self.loaded = False

    def load(self) -> None:
        neurons = load_neurons(processed_dir=self.processed_dir)
        self.neurons = neurons
        self.ids = neurons["neuron_id"].to_numpy()  # sorted; position in this array == node_index
        self.cell_type = neurons["cell_type"].to_list()
        self.super_class = neurons["super_class"].to_list()
        size = len(self.ids)

        edges = load_edges(
            columns=["source_index", "target_index", "synapse_count", "weight", "sign"],
            processed_dir=self.processed_dir,
        )
        self.src = edges["source_index"].to_numpy()
        self.dst = edges["target_index"].to_numpy()
        self.syn = edges["synapse_count"].to_numpy()
        self.weight = edges["weight"].to_numpy()
        self.sign = edges["sign"].to_numpy()
        del edges

        self.out_offsets = _offsets(self.src, size)
        self.in_order = np.argsort(self.dst, kind="stable").astype(np.int64)
        self.in_offsets = _offsets(self.dst, size)
        self.in_synapses = np.bincount(self.dst, weights=self.syn, minlength=size).astype(np.int64)
        self.out_synapses = np.bincount(self.src, weights=self.syn, minlength=size).astype(np.int64)

        self.report = json.loads((self.processed_dir / "ingest_report.json").read_text(encoding="utf-8"))
        self._point_cloud_json = self._build_point_cloud()
        self._point_cloud_gzip = None
        self.loaded = True

    # --- lookups ------------------------------------------------------------

    def index_of(self, neuron_id: int) -> int:
        index = int(np.searchsorted(self.ids, neuron_id))
        if index >= len(self.ids) or self.ids[index] != neuron_id:
            raise KeyError(neuron_id)
        return index

    def _edge_rows(self, index: int, direction: str) -> np.ndarray:
        if direction == "out":
            return np.arange(self.out_offsets[index], self.out_offsets[index + 1])
        return self.in_order[self.in_offsets[index]:self.in_offsets[index + 1]]

    def neuron(self, index: int) -> dict:
        row = self.neurons.row(index, named=True)

        def micrometres(prefix: str) -> list[float] | None:
            values = [row[f"{prefix}{axis}"] for axis in "xyz"]
            return None if any(v is None for v in values) else [round(v / 1000, 2) for v in values]

        return {
            "neuron_id": str(row["neuron_id"]),
            "node_index": index,
            **{field: row[field] for field in METADATA_FIELDS},
            "nt_confidence": None if row["nt_confidence"] is None else round(row["nt_confidence"], 3),
            "position_um": micrometres(""),
            "soma_um": micrometres("soma_"),
            "has_annotation": row["has_annotation"],
            "is_outlier": row["is_outlier"],
            "in_degree": int(self.in_offsets[index + 1] - self.in_offsets[index]),
            "out_degree": int(self.out_offsets[index + 1] - self.out_offsets[index]),
            "in_synapses": int(self.in_synapses[index]),
            "out_synapses": int(self.out_synapses[index]),
        }

    def connections(self, index: int, direction: str, min_synapses: int, limit: int) -> dict:
        rows = self._edge_rows(index, direction)
        rows = rows[self.syn[rows] >= min_synapses]
        partners = self.dst[rows] if direction == "out" else self.src[rows]
        top = np.argsort(-self.syn[rows], kind="stable")[:limit]
        items = [
            {
                "neuron_id": str(self.ids[partner]),
                "node_index": int(partner),
                "cell_type": self.cell_type[partner],
                "super_class": self.super_class[partner],
                "synapse_count": int(self.syn[row]),
                "weight": round(float(self.weight[row]), 4),
                "sign": int(self.sign[row]),
            }
            for row, partner in zip(rows[top], partners[top])
        ]
        return {"total": int(len(rows)), "shown": len(items), "items": items}

    def neighbors(self, index: int, min_synapses: int) -> dict:
        incoming = self._edge_rows(index, "in")
        outgoing = self._edge_rows(index, "out")
        upstream = self.src[incoming[self.syn[incoming] >= min_synapses]]
        downstream = self.dst[outgoing[self.syn[outgoing] >= min_synapses]]
        return {
            "upstream": [str(self.ids[i]) for i in upstream],
            "downstream": [str(self.ids[i]) for i in downstream],
        }

    def search(self, query: str, limit: int) -> list[dict]:
        text = query.strip()
        frame = self.neurons.select("node_index", "neuron_id", "cell_type", "super_class", "side")
        if text.isdigit():
            hits = frame.filter(pl.col("neuron_id").cast(pl.String).str.starts_with(text))
        else:
            lowered = pl.col("cell_type").str.to_lowercase()
            hits = (
                frame.filter(lowered.str.contains(text.lower(), literal=True))
                .with_columns((lowered == text.lower()).alias("exact"))
                .sort(["exact", "cell_type"], descending=[True, False])
                .drop("exact")
            )
        return [{**row, "neuron_id": str(row["neuron_id"])} for row in hits.head(limit).iter_rows(named=True)]

    def stats(self) -> dict:
        neurons, edges = self.report["neurons"], self.report["edges"]
        return {
            "dataset": "FlyWire FAFB v783",
            "source": self.report["source"],
            "neurons": neurons["total"],
            "edges": edges["total"],
            "synapses": edges["total_synapses"],
            "super_class_counts": neurons["super_class_counts"],
            "edges_with_min_synapses": edges["edges_with_min_synapses"],
        }

    # --- visualization --------------------------------------------------------

    def point_cloud_json(self) -> str:
        return self._point_cloud_json

    def point_cloud_gzip(self) -> bytes:
        # ~6 MB of JSON compresses to ~1.7 MB; doing it once avoids seconds of work per request.
        if self._point_cloud_gzip is None:
            self._point_cloud_gzip = gzip.compress(self._point_cloud_json.encode(), compresslevel=6)
        return self._point_cloud_gzip

    def _build_point_cloud(self) -> str:
        """Positions of every neuron with a known location, serialized once at startup."""
        placed = self.neurons.filter(pl.col("has_position")).select("node_index", "neuron_id", "x", "y", "z", "super_class")
        classes = placed["super_class"].value_counts(sort=True)["super_class"].to_list()
        code = {name: i for i, name in enumerate(classes)}

        xyz = placed.select("x", "y", "z").to_numpy().astype(np.float64) / 1000.0  # nm -> um
        xyz -= (xyz.min(axis=0) + xyz.max(axis=0)) / 2  # centre the brain on the origin
        # EM image y grows downwards. Rotating 180 degrees about x (flip y and z) puts dorsal up
        # without mirroring left and right.
        xyz[:, 1:] *= -1

        payload = {
            "count": placed.height,
            "units": "micrometres, centred on the brain",
            "classes": classes,
            "node_index": placed["node_index"].to_list(),
            # IDs are sent as strings: JavaScript numbers cannot hold 18-digit integers exactly.
            "neuron_ids": placed["neuron_id"].cast(pl.String).to_list(),
            "class_code": [code[name] for name in placed["super_class"]],
            "positions": np.round(xyz, 1).ravel().tolist(),
        }
        return json.dumps(payload, separators=(",", ":"))
