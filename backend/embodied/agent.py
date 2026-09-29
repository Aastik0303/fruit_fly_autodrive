"""Policies for the dodge game.

LinearQAgent learns Q(s, a) = w_a . phi(s) + b_a with one-step Q-learning:

    target = r + gamma * max_a' Q(s', a')      (just r when the game is over)
    error  = target - Q(s, a)
    w_a   += step * error * phi(s),   step = learning_rate / (|phi|^2 + 1)

Dividing by |phi|^2 ("normalized LMS") keeps updates stable whatever the number of features.
The connectome itself is never changed: only the readout weights w and b are learned.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from backend.embodied.dodge_env import ENEMY_SIZE, ENEMY_SPEED, LEFT, PLAYER_SIZE, PLAYER_SPEED, PLAYER_Y, RIGHT, STAY, WIDTH

N_ACTIONS = 3
STAY_ON_TIE = np.array([0.0, 1e-6, 0.0])  # an untrained agent (all Q equal) stays still


class LinearQAgent:
    def __init__(self, n_features: int, learning_rate: float = 0.1, gamma: float = 0.97, seed: int = 0):
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.rng = np.random.default_rng(seed)
        self.weights = np.zeros((N_ACTIONS, n_features), dtype=np.float32)
        self.bias = np.zeros(N_ACTIONS, dtype=np.float32)
        self.feature_mean = np.zeros(n_features, dtype=np.float32)
        self.feature_std = np.ones(n_features, dtype=np.float32)

    def fit_normalization(self, samples: np.ndarray) -> None:
        std = samples.std(axis=0)
        floor = 0.01 * float(np.median(std[std > 0])) if np.any(std > 0) else 1.0
        self.feature_mean = samples.mean(axis=0).astype(np.float32)
        self.feature_std = np.maximum(std, floor).astype(np.float32)

    def features(self, raw: np.ndarray) -> np.ndarray:
        return (raw - self.feature_mean) / self.feature_std

    def q_values(self, phi: np.ndarray) -> np.ndarray:
        return self.weights @ phi + self.bias

    def act(self, phi: np.ndarray, epsilon: float = 0.0) -> int:
        if epsilon > 0 and self.rng.random() < epsilon:
            return int(self.rng.integers(N_ACTIONS))
        return int(np.argmax(self.q_values(phi) + STAY_ON_TIE))

    def update(self, phi: np.ndarray, action: int, reward: float, next_phi: np.ndarray, terminal: bool) -> float:
        target = reward if terminal else reward + self.gamma * float(np.max(self.q_values(next_phi)))
        error = target - float(self.q_values(phi)[action])
        step = self.learning_rate / (float(phi @ phi) + 1.0)
        self.weights[action] += step * error * phi
        self.bias[action] += step * error
        return error

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path, weights=self.weights, bias=self.bias, feature_mean=self.feature_mean, feature_std=self.feature_std,
            learning_rate=self.learning_rate, gamma=self.gamma,
        )

    @classmethod
    def load(cls, path: Path) -> "LinearQAgent":
        data = np.load(path)
        agent = cls(data["weights"].shape[1], float(data["learning_rate"]), float(data["gamma"]))
        agent.weights, agent.bias = data["weights"], data["bias"]
        agent.feature_mean, agent.feature_std = data["feature_mean"], data["feature_std"]
        return agent


def heuristic_action(player_x: float, enemies: list, look_ahead: int = 12) -> int:
    """Hand-written reference policy: move to where the fewest blocks will arrive soon. Not learned."""
    best_action, best_danger = STAY, float("inf")
    for action, direction in ((STAY, 0), (LEFT, -1), (RIGHT, 1)):
        x = min(max(player_x + direction * PLAYER_SPEED * look_ahead, 0), WIDTH - PLAYER_SIZE)
        danger = 0.0
        for enemy_x, enemy_y in enemies:
            frames_until_contact = (PLAYER_Y - (enemy_y + ENEMY_SIZE)) / ENEMY_SPEED
            overlaps = enemy_x - PLAYER_SIZE - 8 < x < enemy_x + ENEMY_SIZE + 8
            if overlaps and -ENEMY_SIZE / ENEMY_SPEED <= frames_until_contact <= 45:
                danger += 1.0 / (1.0 + max(frames_until_contact, 0.0))
        if danger < best_danger - 1e-9:
            best_action, best_danger = action, danger
    return best_action
