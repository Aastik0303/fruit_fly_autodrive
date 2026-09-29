"""Train the dodge-game agents and compare them with simple baselines.

Agents that learn (same Q-learning, different inputs):
    connectome      eye grid -> R7/R8 photoreceptors -> 4 hops through FlyWire -> 1,303 descending neurons -> Q
    pixels          eye grid -> Q directly (no connectome)
    random_mixing   eye grid -> random 1,303 x 240 matrix -> Q (control for "does the real wiring matter?")

Fixed reference policies: random, always stay, hand-written heuristic.

    python -m backend.embodied.train
    python -m backend.embodied.train --episodes 40 --eval-episodes 5     # quick check
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable

import numpy as np

from backend.embodied.agent import N_ACTIONS, LinearQAgent, heuristic_action
from backend.embodied.brain import ConnectomeBrain
from backend.embodied.dodge_env import FPS, STAY, DodgeEnv
from backend.embodied.eye import ego_view
from backend.utils.config import MODELS_DIR

FRAME_SKIP = 4  # one decision every 4 frames = 20 decisions per second at 80 FPS
MAX_FRAMES = 8000  # 100 s of game time per episode
REWARD_PER_BLOCK = 0.1
HIT_PENALTY = -10.0
EPSILON_START, EPSILON_END = 1.0, 0.02

RL_DIR = MODELS_DIR / "rl"


def reward_for(result) -> float:
    return REWARD_PER_BLOCK * result.score_gained + (HIT_PENALTY if result.hit else 0.0)


def random_policy_samples(featurize: Callable, count: int = 4000, seed: int = 7) -> np.ndarray:
    env, rng, samples = DodgeEnv(seed=seed), np.random.default_rng(seed), []
    while len(samples) < count:
        if env.done:
            env.reset()
        env.step(int(rng.integers(N_ACTIONS)), FRAME_SKIP)
        samples.append(featurize(ego_view(env.player_x, env.enemies)))
    return np.asarray(samples)


def train_agent(name: str, featurize: Callable, n_features: int, episodes: int, seed: int = 0) -> tuple[LinearQAgent, list]:
    agent = LinearQAgent(n_features, seed=seed)
    agent.fit_normalization(random_policy_samples(featurize))
    curve, started = [], time.perf_counter()
    explore_episodes = max(1, int(0.7 * episodes))

    for episode in range(episodes):
        epsilon = max(EPSILON_END, EPSILON_START - (EPSILON_START - EPSILON_END) * episode / explore_episodes)
        env = DodgeEnv(seed=1_000 + episode, max_frames=MAX_FRAMES)
        phi = agent.features(featurize(ego_view(env.player_x, env.enemies)))
        while not env.done:
            action = agent.act(phi, epsilon)
            result = env.step(action, FRAME_SKIP)
            next_phi = agent.features(featurize(ego_view(env.player_x, env.enemies)))
            agent.update(phi, action, reward_for(result), next_phi, terminal=env.attempts == 0)
            phi = next_phi
        curve.append({"episode": episode, "score": env.score, "seconds": round(env.frame / FPS, 1), "hits": env.hits, "epsilon": round(epsilon, 3)})
        if (episode + 1) % 20 == 0:
            recent = curve[-20:]
            print(
                f"[{name}] episode {episode + 1}/{episodes}  eps {epsilon:.2f}  "
                f"mean score {np.mean([c['score'] for c in recent]):.0f}  "
                f"mean survival {np.mean([c['seconds'] for c in recent]):.0f} s  "
                f"({time.perf_counter() - started:.0f} s)"
            )
    return agent, curve


def evaluate(policy: Callable, episodes: int) -> dict:
    scores, seconds, hits, full_runs = [], [], [], 0
    for episode in range(episodes):
        env = DodgeEnv(seed=50_000 + episode, max_frames=MAX_FRAMES)
        while not env.done:
            env.step(policy(env), FRAME_SKIP)
        scores.append(env.score)
        seconds.append(env.frame / FPS)
        hits.append(env.hits)
        full_runs += env.attempts > 0
    minutes = sum(seconds) / 60
    return {
        "mean_score": round(float(np.mean(scores)), 1),
        "std_score": round(float(np.std(scores)), 1),
        "mean_survival_s": round(float(np.mean(seconds)), 1),
        "hits_per_minute": round(sum(hits) / minutes, 2),
        "survived_full_100s": f"{full_runs}/{episodes}",
    }


def q_policy(agent: LinearQAgent, featurize: Callable) -> Callable:
    return lambda env: agent.act(agent.features(featurize(ego_view(env.player_x, env.enemies))))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", type=int, default=400)
    parser.add_argument("--eval-episodes", type=int, default=30)
    args = parser.parse_args()

    brain = ConnectomeBrain().load()
    n_descending = len(brain.descending_index)
    view_size = ego_view(500, []).size
    # Control: the same eye grid mixed by a random matrix of the connectome readout's size.
    # If it scores like the connectome, the score alone says nothing about real fly wiring.
    mixing = np.random.default_rng(12345).normal(size=(n_descending, view_size)).astype(np.float32)
    featurizers = {
        "connectome": (brain.descending_activity, n_descending),
        "pixels": (lambda view: view.ravel(), view_size),
        "random_mixing": (lambda view: mixing @ view.ravel(), n_descending),
    }

    report = {"settings": {
        "episodes": args.episodes, "eval_episodes": args.eval_episodes, "frame_skip": FRAME_SKIP,
        "max_frames": MAX_FRAMES, "reward_per_block": REWARD_PER_BLOCK, "hit_penalty": HIT_PENALTY,
        "hops": brain.hops, "descending_neurons": len(brain.descending_index), "grid_cells": view_size,
    }, "training": {}, "evaluation": {}}

    agents = {}
    for name, (featurize, size) in featurizers.items():
        agent, curve = train_agent(name, featurize, size, args.episodes)
        agent.save(RL_DIR / f"dodge_{name}_q.npz")
        agents[name] = agent
        report["training"][name] = curve

    rng = np.random.default_rng(0)
    policies = {
        "random": lambda env: int(rng.integers(N_ACTIONS)),
        "always_stay": lambda env: STAY,
        "heuristic (hand-written)": lambda env: heuristic_action(env.player_x, env.enemies),
        "pixels Q-learning": q_policy(agents["pixels"], featurizers["pixels"][0]),
        "random mixing Q-learning (control)": q_policy(agents["random_mixing"], featurizers["random_mixing"][0]),
        "connectome Q-learning": q_policy(agents["connectome"], featurizers["connectome"][0]),
    }
    for name, policy in policies.items():
        report["evaluation"][name] = evaluate(policy, args.eval_episodes)
        print(f"{name:<26} {report['evaluation'][name]}")

    RL_DIR.mkdir(parents=True, exist_ok=True)
    (RL_DIR / "dodge_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"saved agents and report to {RL_DIR}")


if __name__ == "__main__":
    main()
