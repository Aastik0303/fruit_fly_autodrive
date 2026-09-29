"""FastAPI backend.

Current scope (Phase 1 data): neuron lookup, search, connections and 3D positions.
Path, circuit, GNN and RL endpoints are added in their own phases.

    python -m uvicorn backend.api.main:app --port 8000
    interactive docs: http://127.0.0.1:8000/docs
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from backend.api.game import router as game_router
from backend.api.store import ConnectomeStore

store = ConnectomeStore()


@asynccontextmanager
async def lifespan(app: FastAPI):
    store.load()
    yield


app = FastAPI(title="NeuroPath AI API", version="0.1.0", lifespan=lifespan)
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET"],
    allow_headers=["*"],
)
app.include_router(game_router)  # WebSocket /ws/game: the fly plays the dodge game


def _index(neuron_id: int) -> int:
    try:
        return store.index_of(neuron_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Neuron {neuron_id} is not in FlyWire v783") from None


@app.get("/health")
def health():
    return {"status": "ok", "loaded": store.loaded}


@app.get("/stats")
def stats():
    return store.stats()


# declared before /neurons/{neuron_id} so "search" is not parsed as an ID
@app.get("/neurons/search")
def search_neurons(
    q: str = Query(..., min_length=1, description="neuron ID prefix or part of a cell type"),
    limit: int = Query(20, ge=1, le=100),
):
    return {"query": q, "results": store.search(q, limit)}


@app.get("/neurons/{neuron_id}")
def get_neuron(neuron_id: int):
    return store.neuron(_index(neuron_id))


@app.get("/neurons/{neuron_id}/neighbors")
def get_neighbors(neuron_id: int, min_synapses: int = Query(1, ge=1)):
    return {"neuron_id": str(neuron_id), "min_synapses": min_synapses, **store.neighbors(_index(neuron_id), min_synapses)}


@app.get("/neurons/{neuron_id}/connections")
def get_connections(
    neuron_id: int,
    direction: Literal["in", "out", "both"] = "both",
    min_synapses: int = Query(1, ge=1),
    limit: int = Query(50, ge=1, le=1000),
):
    index = _index(neuron_id)
    body = {"neuron_id": str(neuron_id), "min_synapses": min_synapses}
    if direction in ("in", "both"):
        body["incoming"] = store.connections(index, "in", min_synapses, limit)
    if direction in ("out", "both"):
        body["outgoing"] = store.connections(index, "out", min_synapses, limit)
    return body


@app.get("/visualization/neurons")
def visualization_neurons(request: Request):
    if "gzip" in request.headers.get("accept-encoding", ""):
        return Response(
            store.point_cloud_gzip(),
            media_type="application/json",
            headers={"Content-Encoding": "gzip", "Vary": "Accept-Encoding"},
        )
    return Response(store.point_cloud_json(), media_type="application/json")
