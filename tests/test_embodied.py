"""Tests for the dodge-game port, the eye encoding and the Q-learning update."""

import numpy as np
import polars as pl

from backend.embodied.agent import LinearQAgent
from backend.embodied.dodge_env import (
    ATTEMPTS, ENEMY_SIZE, ENEMY_SPEED, HEIGHT, LEFT, PLAYER_SIZE, PLAYER_SPEED, PLAYER_Y, RIGHT, STAY, WIDTH,
    DodgeEnv, detect_collision, milestone_message,
)
from backend.embodied.eye import GRID_COLS, GRID_ROWS, N_CELLS, WALL_VALUE, ego_view, input_matrix, photoreceptor_layout


def test_collision_matches_original_rule():
    assert detect_collision((100, 500), (90, 480))
    assert not detect_collision((100, 500), (200, 480))
    assert not detect_collision((100, 500), (130, 480))  # strict inequalities: touching edges is not a hit


def test_player_moves_and_stops_at_walls():
    env = DodgeEnv(seed=0)
    env.player_x = 0
    env.frame_step(LEFT)
    assert env.player_x == 0
    env.frame_step(RIGHT)
    assert env.player_x == PLAYER_SPEED
    env.player_x = WIDTH - PLAYER_SIZE
    env.frame_step(RIGHT)
    assert env.player_x == WIDTH - PLAYER_SIZE


def test_block_leaving_screen_scores_and_hit_costs_an_attempt():
    env = DodgeEnv(seed=0)
    env.rng.random = lambda: 1.0  # never spawn
    env.enemies = [[0, HEIGHT - ENEMY_SPEED + 1]]
    assert env.frame_step(STAY).score_gained == 1 and env.score == 1

    env.enemies = [[env.player_x - 10, PLAYER_Y - ENEMY_SIZE + 10 - ENEMY_SPEED]]
    result = env.frame_step(STAY)
    assert result.hit and env.attempts == ATTEMPTS - 1 and env.enemies == [] and env.player_x == WIDTH // 2


def test_game_ends_after_three_hits():
    env = DodgeEnv(seed=0)
    env.rng.random = lambda: 1.0
    for _ in range(ATTEMPTS):
        env.enemies = [[env.player_x - 10, PLAYER_Y - 20]]
        env.frame_step(STAY)
    assert env.done and env.attempts == 0


def test_milestones():
    assert milestone_message(55) == "Half Century!"
    assert milestone_message(61) is None


def test_ego_view_is_centred_on_the_player():
    player_x = WIDTH // 2 - PLAYER_SIZE // 2  # view spans exactly the screen, so no wall is visible
    centre = player_x + PLAYER_SIZE / 2
    view = ego_view(player_x, [[int(centre) - ENEMY_SIZE, 0]])  # block just left of the fly, top row
    assert view.shape == (GRID_ROWS, GRID_COLS)
    assert view[0, GRID_COLS // 2 - 1] > 0 and view[0, GRID_COLS // 2] == 0
    assert view[GRID_ROWS - 1].max() == 0


def test_ego_view_shows_walls_near_the_edge():
    view = ego_view(0, [])
    assert view[:, 0].max() == WALL_VALUE and view[:, -1].max() == 0


def test_photoreceptor_layout_fills_every_cell_and_keeps_sides_apart():
    rng = np.random.default_rng(0)
    rows = []
    for side, x_offset in (("left", 0.0), ("right", 1000.0)):
        for i in range(2400):
            rows.append({"node_index": len(rows), "cell_type": "R7", "side": side, "has_position": True,
                         "x": x_offset + rng.normal(0, 5), "y": rng.uniform(0, 100), "z": rng.uniform(0, 60)})
    neurons = pl.DataFrame(rows)
    layout = photoreceptor_layout(neurons)
    cells = layout["row"].to_numpy() * GRID_COLS + layout["col"].to_numpy()
    assert np.bincount(cells, minlength=N_CELLS).min() > 0
    assert (layout.filter(pl.col("side") == "left")["col"] < GRID_COLS // 2).all()
    assert (layout.filter(pl.col("side") == "right")["col"] >= GRID_COLS // 2).all()

    drive = input_matrix(layout, len(rows))
    assert np.allclose(np.asarray(drive.sum(axis=0)).ravel(), 1.0)  # each cell delivers a total drive of 1


def test_q_learning_update_moves_value_toward_target():
    agent = LinearQAgent(n_features=2, learning_rate=0.5, gamma=0.9)
    phi = np.array([1.0, 0.0], dtype=np.float32)
    before = agent.q_values(phi)[LEFT]
    agent.update(phi, LEFT, reward=-10.0, next_phi=phi, terminal=True)
    assert agent.q_values(phi)[LEFT] < before
    assert agent.act(phi) != LEFT
