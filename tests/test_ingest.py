"""Phase 1 ingestion tests on tiny synthetic files in both raw formats."""

import gzip

import polars as pl
import pytest

from backend.preprocessing.ingest import build

# Real FlyWire IDs are this large (> 2**53), so a float64 conversion would silently corrupt them.
BASE = 720575940600000000
A, B, C, D = BASE + 1, BASE + 2, BASE + 3, BASE + 4


def write_csv(path, frame: pl.DataFrame) -> None:
    if path.suffix == ".gz":
        with gzip.open(path, "wb") as handle:
            frame.write_csv(handle)
    else:
        frame.write_csv(path)


def write_public(raw_dir) -> None:
    raw_dir.mkdir(parents=True)
    pl.DataFrame(
        {
            "supervoxel_id": [11, 12, 13],
            "root_id": [A, B, C],
            "pos_x": [10.0, 20.0, 30.0], "pos_y": [1.0, 2.0, 3.0], "pos_z": [5.0, 6.0, 7.0],
            "soma_x": [10, None, 30], "soma_y": [1, None, 3], "soma_z": [5, None, 7],
            "flow": ["intrinsic", "intrinsic", "afferent"],
            "super_class": ["central", "optic", "sensory"],
            "cell_class": [None, "ME", "olfactory"],
            "cell_sub_class": [None, None, None],
            "cell_type": ["T1", None, "ORN"],
            "hemibrain_type": [None, None, None],
            "side": ["left", "right", "left"],
            "top_nt": ["acetylcholine", "gaba", None],
            "top_nt_conf": [0.9, 0.8, None],
            "status": [None, "outlier_seg", None],
        },
        schema_overrides={"cell_sub_class": pl.String, "hemibrain_type": pl.String},
    ).write_csv(raw_dir / "Supplemental_file1_neuron_annotations.tsv", separator="\t")
    pl.DataFrame(
        {
            "Presynaptic_ID": [A, C, B],
            "Postsynaptic_ID": [B, B, D],  # D has no annotation row
            "Presynaptic_Index": [0, 2, 1],
            "Postsynaptic_Index": [1, 1, 3],
            "Connectivity": [10, 30, 5],
            "Excitatory": [1, -1, -1],
        }
    ).write_parquet(raw_dir / "Connectivity_783.parquet")


def write_codex(raw_dir, synapse_column: str = "syn_count") -> None:
    raw_dir.mkdir(parents=True)
    write_csv(raw_dir / "neurons.csv", pl.DataFrame(
        {"root_id": [A, B, C], "group": ["g1", "g2", "g3"], "nt_type": ["ACH", "GABA", "GLUT"], "nt_type_score": [0.9, 0.7, 0.6]}
    ))
    write_csv(raw_dir / "classification.csv", pl.DataFrame(
        {
            "root_id": [A, B, C], "flow": ["intrinsic"] * 3, "super_class": ["central", "central", "optic"],
            "class": ["CX", None, "ME"], "sub_class": [None] * 3, "cell_type": ["old_type", None, None],
            "hemibrain_type": [None] * 3, "hemilineage": [None] * 3, "side": ["left", "right", "left"], "nerve": [None] * 3,
        },
        schema_overrides={c: pl.String for c in ("sub_class", "hemibrain_type", "hemilineage", "nerve")},
    ))
    write_csv(raw_dir / "consolidated_cell_types.csv", pl.DataFrame(
        {"root_id": [A], "primary_type": ["EPG"], "additional_type(s)": [None]},
        schema_overrides={"additional_type(s)": pl.String},
    ))
    write_csv(raw_dir / "connections_princeton.csv.gz", pl.DataFrame(
        {"pre_root_id": [A, A, B], "post_root_id": [B, B, C], "neuropil": ["EB", "PB", "ME_L"],
         synapse_column: [4, 6, 3], "nt_type": ["ACH", "ACH", "GABA"]}
    ))
    write_csv(raw_dir / "coordinates.csv", pl.DataFrame(
        {"root_id": [A, A, B], "position": ["[400 800 100]", "[1 1 1]", "[1200 1600 300]"], "supervoxel_id": [1, 2, 3]}
    ))


def read_outputs(out_dir):
    return pl.read_parquet(out_dir / "neurons.parquet"), pl.read_parquet(out_dir / "edges.parquet")


def test_public_source(tmp_path):
    write_public(tmp_path / "public")
    report = build("public", raw_public=tmp_path / "public", raw_codex=tmp_path / "codex", out_dir=tmp_path / "out")
    neurons, edges = read_outputs(tmp_path / "out")

    assert neurons["neuron_id"].dtype == pl.Int64
    assert neurons["neuron_id"].to_list() == [A, B, C, D]
    assert neurons["node_index"].to_list() == [0, 1, 2, 3]

    a = neurons.row(0, named=True)
    assert (a["x"], a["y"], a["z"]) == (40.0, 4.0, 200.0)  # voxels -> nm
    assert a["cell_class"] == "unknown" and a["has_soma"]

    d = neurons.row(3, named=True)
    assert not d["has_annotation"] and not d["has_position"] and d["super_class"] == "unknown"
    assert neurons.filter(pl.col("neuron_id") == B)["is_outlier"].item()
    assert report["neurons"]["missing_values_before_fill"]["cell_type"] == 2

    weights = dict(zip(zip(edges["source_id"], edges["target_id"]), edges["weight"]))
    assert weights[(A, B)] == pytest.approx(0.25)
    assert weights[(C, B)] == pytest.approx(0.75)
    assert weights[(B, D)] == pytest.approx(1.0)

    signs = dict(zip(zip(edges["source_id"], edges["target_id"]), edges["sign"]))
    assert signs == {(A, B): 1, (C, B): 0, (B, D): -1}  # from presynaptic transmitter, not the Excitatory column

    id_of = dict(zip(neurons["node_index"], neurons["neuron_id"]))
    assert all(id_of[i] == n for i, n in zip(edges["source_index"], edges["source_id"]))
    assert all(id_of[i] == n for i, n in zip(edges["target_index"], edges["target_id"]))
    assert report["edges"]["total_synapses"] == 45


def test_codex_source_is_picked_automatically(tmp_path):
    write_codex(tmp_path / "codex")
    report = build("auto", raw_public=tmp_path / "public", raw_codex=tmp_path / "codex", out_dir=tmp_path / "out")
    neurons, edges = read_outputs(tmp_path / "out")

    assert report["source"] == "codex"
    assert neurons["neuron_id"].to_list() == [A, B, C]

    a = neurons.row(0, named=True)
    assert a["cell_type"] == "EPG"  # consolidated type wins over classification
    assert a["cell_class"] == "CX" and a["neurotransmitter"] == "acetylcholine"
    assert (a["x"], a["y"], a["z"]) == (1600.0, 3200.0, 4000.0)  # first position row, voxels -> nm
    assert not neurons.row(2, named=True)["has_position"]

    ab = edges.filter((pl.col("source_id") == A) & (pl.col("target_id") == B)).row(0, named=True)
    assert ab["synapse_count"] == 10  # two neuropil rows merged
    assert ab["main_neuropil"] == "PB"
    assert ab["sign"] == 1 and ab["weight"] == pytest.approx(1.0)
    assert edges.filter(pl.col("source_id") == B)["sign"].item() == -1  # GABA
    assert report["edges"]["rows_merged_into_same_pair"] == 1


def test_codex_missing_required_column(tmp_path):
    write_codex(tmp_path / "codex", synapse_column="weight_unknown")
    with pytest.raises(ValueError, match="synapse_count"):
        build("codex", raw_codex=tmp_path / "codex", out_dir=tmp_path / "out")
