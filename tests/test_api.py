"""API tests on a tiny processed dataset built from the synthetic public files."""

import pytest
from fastapi.testclient import TestClient

from backend.api import main
from backend.preprocessing.ingest import build
from tests.test_ingest import BASE, A, B, C, D, write_public


@pytest.fixture
def client(tmp_path, monkeypatch):
    write_public(tmp_path / "public")
    build("public", raw_public=tmp_path / "public", raw_codex=tmp_path / "codex", out_dir=tmp_path / "out")
    monkeypatch.setattr(main.store, "processed_dir", tmp_path / "out")
    with TestClient(main.app) as test_client:
        yield test_client


def test_neuron_lookup_keeps_ids_as_strings(client):
    body = client.get(f"/neurons/{B}").json()
    assert body["neuron_id"] == str(B)
    assert (body["in_degree"], body["out_degree"]) == (2, 1)
    assert (body["in_synapses"], body["out_synapses"]) == (40, 5)


def test_unknown_neuron_returns_404(client):
    assert client.get("/neurons/123").status_code == 404


def test_connections_sorted_and_filtered(client):
    body = client.get(f"/neurons/{B}/connections").json()
    incoming = body["incoming"]["items"]
    assert [item["neuron_id"] for item in incoming] == [str(C), str(A)]  # 30 synapses before 10
    assert incoming[0]["weight"] == pytest.approx(0.75)
    assert body["outgoing"]["items"][0]["neuron_id"] == str(D)

    strong = client.get(f"/neurons/{B}/connections", params={"min_synapses": 20, "direction": "in"}).json()
    assert strong["incoming"]["total"] == 1 and "outgoing" not in strong


def test_neighbors(client):
    body = client.get(f"/neurons/{B}/neighbors").json()
    assert set(body["upstream"]) == {str(A), str(C)}
    assert body["downstream"] == [str(D)]


def test_search_by_cell_type_and_id_prefix(client):
    assert client.get("/neurons/search", params={"q": "orn"}).json()["results"][0]["neuron_id"] == str(C)
    assert len(client.get("/neurons/search", params={"q": str(BASE)[:10]}).json()["results"]) == 4


def test_point_cloud_skips_neurons_without_position(client):
    body = client.get("/visualization/neurons").json()
    assert body["count"] == 3 and len(body["positions"]) == 9
    assert str(D) not in body["neuron_ids"]
