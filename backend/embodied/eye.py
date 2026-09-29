"""From game pixels to fly photoreceptors.

1. Ego-centric view: the screen is re-centred on the fly, so the fly always looks from the middle.
   Space beyond the screen edge is shown as a faint wall.
2. The view is cut into a GRID_ROWS x GRID_COLS grid. Each cell holds how much of it is covered by
   blocks (0..1).
3. The left half of the grid feeds the left eye, the right half the right eye. Each eye's R7/R8
   photoreceptors lie roughly on a 2D sheet; PCA of their 3D positions gives the two sheet
   directions. The direction closest to the body's dorsal-ventral axis is used as grid rows, the
   other as grid columns (oriented so that the middle of the view maps to the medial edge of the
   eye). Photoreceptors are split into equal-count bins (rows, then columns within each row), so
   every cell drives its own group of roughly the same size.

This is a simplified, documented encoding, not an optical model of the compound eye.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import scipy.sparse as sp

from backend.embodied.dodge_env import ENEMY_SIZE, PLAYER_SIZE, PLAYER_Y, WIDTH

GRID_ROWS = 12
GRID_COLS = 20  # 10 per eye
N_CELLS = GRID_ROWS * GRID_COLS
VIEW_HALF_WIDTH = 500  # pixels visible on each side of the fly
WALL_VALUE = 0.3
# Only the inner photoreceptors R7 and R8 are used. They synapse directly in the medulla.
# In this dataset, R1-6 input from several eye regions stays inside the lamina for 4+ hops and never
# reaches a descending neuron, which left blind spots right next to the fly.
PHOTORECEPTOR_TYPES = ("R7", "R8")

_ROW_EDGES = np.linspace(0, PLAYER_Y + PLAYER_SIZE, GRID_ROWS + 1)


def ego_view(player_x: float, enemies: list) -> np.ndarray:
    """Block coverage of each grid cell, shape (GRID_ROWS, GRID_COLS), values 0..1."""
    centre = player_x + PLAYER_SIZE / 2
    col_edges = np.linspace(centre - VIEW_HALF_WIDTH, centre + VIEW_HALF_WIDTH, GRID_COLS + 1)
    col_width = col_edges[1] - col_edges[0]
    row_height = _ROW_EDGES[1] - _ROW_EDGES[0]

    if enemies:
        blocks = np.asarray(enemies, dtype=np.float64)
        left, top = blocks[:, :1], blocks[:, 1:2]
        cover_x = np.clip(np.minimum(left + ENEMY_SIZE, col_edges[1:]) - np.maximum(left, col_edges[:-1]), 0, None) / col_width
        cover_y = np.clip(np.minimum(top + ENEMY_SIZE, _ROW_EDGES[1:]) - np.maximum(top, _ROW_EDGES[:-1]), 0, None) / row_height
        view = np.minimum(cover_y.T @ cover_x, 1.0)
    else:
        view = np.zeros((GRID_ROWS, GRID_COLS))

    beyond_wall = (np.clip(-col_edges[:-1], 0, col_width) + np.clip(col_edges[1:] - WIDTH, 0, col_width)) / col_width
    view = np.maximum(view, WALL_VALUE * np.minimum(beyond_wall, 1.0)[None, :])
    return view.astype(np.float32)


def _equal_bins(values: np.ndarray, bins: int) -> np.ndarray:
    ranks = np.argsort(np.argsort(values))
    return (ranks * bins // len(values)).astype(np.int32)


def photoreceptor_layout(neurons: pl.DataFrame) -> pl.DataFrame:
    """Assign every photoreceptor to one grid cell. Returns node_index, side, cell_type, row, col."""
    midline_x = float(neurons.filter(pl.col("has_position"))["x"].mean())
    half = GRID_COLS // 2
    parts = []
    for side in ("left", "right"):
        eye = neurons.filter(
            pl.col("cell_type").is_in(PHOTORECEPTOR_TYPES) & (pl.col("side") == side) & pl.col("has_position")
        )
        xyz = eye.select("x", "y", "z").to_numpy().astype(np.float64)
        centred = xyz - xyz.mean(axis=0)
        sheet = np.linalg.svd(centred, full_matrices=False)[2][:2]

        vertical = int(np.argmax(np.abs(sheet[:, 1])))
        down = sheet[vertical] * np.sign(sheet[vertical, 1])  # EM y grows ventrally -> row 0 is dorsal
        across = sheet[1 - vertical]
        toward_midline = np.sign(midline_x - xyz[:, 0].mean())
        across = across * (np.sign(across[0]) * toward_midline or 1.0)

        # rows first, then columns inside each row, so no grid cell is left without photoreceptors
        row = _equal_bins(centred @ down, GRID_ROWS)
        medial = np.empty_like(row)  # half - 1 = most medial
        position_across = centred @ across
        for r in range(GRID_ROWS):
            in_row = row == r
            medial[in_row] = _equal_bins(position_across[in_row], half)
        col = medial if side == "left" else GRID_COLS - 1 - medial
        parts.append(eye.select("node_index", "side", "cell_type").with_columns(row=pl.Series(row), col=pl.Series(col)))
    return pl.concat(parts)


def input_matrix(layout: pl.DataFrame, n_neurons: int) -> sp.csr_matrix:
    """P[neuron, cell] = 1 / (photoreceptors in that cell): a fully covered cell delivers a total drive of 1."""
    cells = layout["row"].to_numpy() * GRID_COLS + layout["col"].to_numpy()
    counts = np.bincount(cells, minlength=N_CELLS)
    data = (1.0 / counts[cells]).astype(np.float32)
    return sp.csr_matrix((data, (layout["node_index"].to_numpy(), cells)), shape=(n_neurons, N_CELLS))
