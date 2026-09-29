"""Headless, frame-exact port of the "Dodge the Falling Blocks" pygame game.

The rules are copied from the original script: 1000x600 screen, a 30 px player moving 9 px per
frame, 50 px blocks falling 5 px per frame, at most 10 blocks, a new block with probability 0.1 per
frame, +1 score when a block leaves the screen, 3 attempts, and the same collision test.
No pygame is needed, so thousands of games can be simulated quickly for reinforcement learning.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

WIDTH, HEIGHT = 1000, 600
FPS = 80
PLAYER_SIZE = 30
PLAYER_Y = HEIGHT - PLAYER_SIZE - 10
PLAYER_SPEED = 9
ENEMY_SIZE = 50
ENEMY_SPEED = 5
MAX_ENEMIES = 10
SPAWN_PROBABILITY = 0.1
ATTEMPTS = 3

LEFT, STAY, RIGHT = 0, 1, 2
ACTION_NAMES = ("left", "stay", "right")

MILESTONES = (
    (50, 60, "Half Century!"),
    (100, 110, "Century!"),
    (200, 210, "Double Century!"),
    (300, 310, "Triple Century!"),
    (400, 410, "Quadruple Century!"),
    (500, 510, "Quintuple Century!"),
    (1000, 1020, "Millennium!"),
)


def detect_collision(player_pos, enemy_pos) -> bool:
    """Same strict-inequality test as the original game."""
    p_x, p_y = player_pos
    e_x, e_y = enemy_pos
    return (e_x < p_x < e_x + ENEMY_SIZE or e_x < p_x + PLAYER_SIZE < e_x + ENEMY_SIZE) and (
        e_y < p_y < e_y + ENEMY_SIZE or e_y < p_y + PLAYER_SIZE < e_y + ENEMY_SIZE
    )


def milestone_message(score: int) -> str | None:
    return next((message for low, high, message in MILESTONES if low <= score <= high), None)


@dataclass
class StepResult:
    score_gained: int
    hit: bool
    done: bool


class DodgeEnv:
    def __init__(self, seed: int | None = None, max_frames: int | None = None):
        self.rng = random.Random(seed)
        self.max_frames = max_frames
        self.reset()

    def reset(self) -> None:
        self.player_x = WIDTH // 2
        self.enemies: list[list[int]] = []
        self.score = 0
        self.attempts = ATTEMPTS
        self.hits = 0
        self.frame = 0
        self.done = False

    def frame_step(self, action: int) -> StepResult:
        """Advance one frame in the same order as the original main loop."""
        if action == LEFT and self.player_x > 0:
            self.player_x -= PLAYER_SPEED
        if action == RIGHT and self.player_x < WIDTH - PLAYER_SIZE:
            self.player_x += PLAYER_SPEED

        # drop_enemies
        delay = self.rng.random()
        if len(self.enemies) < MAX_ENEMIES and delay < SPAWN_PROBABILITY:
            self.enemies.append([self.rng.randint(0, WIDTH - ENEMY_SIZE), 0])

        # update_enemy_positions
        gained = 0
        for enemy in self.enemies[:]:
            enemy[1] += ENEMY_SPEED
            if enemy[1] > HEIGHT:
                self.enemies.remove(enemy)
                gained += 1
        self.score += gained

        hit = False
        for enemy in self.enemies:
            if detect_collision((self.player_x, PLAYER_Y), enemy):
                hit = True
                self.hits += 1
                self.attempts -= 1
                self.enemies.clear()
                self.player_x = WIDTH // 2
                break

        self.frame += 1
        self.done = self.attempts == 0 or (self.max_frames is not None and self.frame >= self.max_frames)
        return StepResult(gained, hit, self.done)

    def step(self, action: int, frames: int = 1) -> StepResult:
        """Hold one action for several frames (the agent decides less often than the game renders)."""
        gained, hit = 0, False
        for _ in range(frames):
            result = self.frame_step(action)
            gained += result.score_gained
            hit = hit or result.hit
            if result.done:
                break
        return StepResult(gained, hit, self.done)

    def snapshot(self) -> dict:
        return {
            "frame": self.frame,
            "score": self.score,
            "attempts": self.attempts,
            "player_x": self.player_x,
            "player_y": PLAYER_Y,
            "enemies": [list(enemy) for enemy in self.enemies],
            "milestone": milestone_message(self.score),
            "done": self.done,
        }
