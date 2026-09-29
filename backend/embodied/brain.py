"""Connectome "brain": linear signed propagation of photoreceptor input through FlyWire v783.

    a_0 = P x                         photoreceptor drive (x = flattened ego-view grid)
    a_h = W a_(h-1)                   one synaptic hop, h = 1..HOPS
    W[target, source] = weight(source -> target) * sign(source)
    activity = sum_h a_h / g_h        g_h = mean L1 size of hop h, so every hop is on the same scale

The model is linear in x, so the response of every neuron to every grid cell is computed once:
R = sum_h W^h P / g_h  (neurons x cells). At run time, activity = R x.

This is a teaching model of how visual input spreads through the wiring diagram. It has no spikes,
no time constants and no plasticity, and its "activity" is not a prediction of real neural firing.

    python -m backend.embodied.brain      # build the cached response matrix (~1 min, ~135 MB)
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import polars as pl
import scipy.sparse as sp

from backend.embodied.eye import GRID_COLS, GRID_ROWS, N_CELLS, input_matrix, photoreceptor_layout
from backend.preprocessing.loaders import load_edges, load_neurons
from backend.utils.config import PROCESSED_DIR

HOPS = 4  # 3-4 hops from the eye already reach almost every descending neuron


class ConnectomeBrain:
    def __init__(self, hops: int = HOPS, processed_dir: Path = PROCESSED_DIR):
        self.hops = hops
        self.processed_dir = processed_dir
        self.cache_path = processed_dir / "embodied" / f"eye_response_hops{hops}_grid{GRID_ROWS}x{GRID_COLS}.npy"

    def load(self) -> "ConnectomeBrain":
        neurons = load_neurons(
            columns=["node_index", "neuron_id", "x", "y", "z", "side", "cell_type", "super_class", "has_position"],
            processed_dir=self.processed_dir,
        )
        self.n_neurons = neurons.height
        self.layout = photoreceptor_layout(neurons)
        self.descending = neurons.filter(pl.col("super_class") == "descending").select("node_index", "neuron_id", "cell_type", "side")
        self.descending_index = self.descending["node_index"].to_numpy()

        if self.cache_path.exists():
            self.response = np.load(self.cache_path)
        else:
            self.response = self._propagate()
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(self.cache_path, self.response)

        self.descending_response = np.ascontiguousarray(self.response[self.descending_index])
        self.neuron_scale = np.abs(self.response).max(axis=1)  # strongest possible response per neuron
        return self

    def _propagate(self) -> np.ndarray:
        started = time.perf_counter()
        edges = load_edges(columns=["source_index", "target_index", "weight", "sign"], processed_dir=self.processed_dir)
        signed = (edges["weight"].to_numpy() * edges["sign"].to_numpy()).astype(np.float32)
        wiring = sp.csr_matrix(
            (signed, (edges["target_index"].to_numpy(), edges["source_index"].to_numpy())),
            shape=(self.n_neurons, self.n_neurons),
        )
        del edges, signed

        drive = input_matrix(self.layout, self.n_neurons).toarray()  # hop 0: photoreceptors
        response = drive.copy()
        self.hop_gains = [1.0]
        for hop in range(1, self.hops + 1):
            drive = wiring @ drive
            gain = float(np.abs(drive).sum(axis=0).mean())
            response += drive / gain
            self.hop_gains.append(gain)
            active = int((np.abs(drive).max(axis=1) > 0).sum())
            print(f"hop {hop}: {active:,} neurons reached, gain {gain:.4f}, {time.perf_counter() - started:.0f} s")
        return response

    def activity(self, view: np.ndarray) -> np.ndarray:
        """Response of all neurons to one ego-view grid."""
        return self.response @ view.reshape(N_CELLS)

    def descending_activity(self, view: np.ndarray) -> np.ndarray:
        return self.descending_response @ view.reshape(N_CELLS)


if __name__ == "__main__":
    brain = ConnectomeBrain().load()
    counts = np.bincount(brain.layout["row"].to_numpy() * GRID_COLS + brain.layout["col"].to_numpy(), minlength=N_CELLS)
    print(f"photoreceptors mapped: {brain.layout.height:,}; per grid cell min {counts.min()}, median {int(np.median(counts))}, max {counts.max()}")
    print(f"response matrix {brain.response.shape}, {brain.response.nbytes / 1e6:.0f} MB -> {brain.cache_path}")
    reached = (brain.neuron_scale[brain.descending_index] > 0).sum()
    print(f"descending neurons with non-zero response: {reached} / {len(brain.descending_index)}")
