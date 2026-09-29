"""Live session: the connectome-driven fly plays the dodge game over a WebSocket.

The game advances at 80 frames per second (times the chosen speed). Every FRAME_SKIP frames the fly
looks (ego-view grid), the linear connectome model turns that into descending-neuron activity, and
the trained Q-learning readout picks left / stay / right. Each decision message also carries a
compact summary of whole-brain activity for the 3D view.

Client commands (JSON): {"cmd": "start" | "pause" | "reset"} and {"cmd": "config", "speed": 1|2|4|8, "policy": "connectome"|"random"}
"""

from __future__ import annotations

import asyncio
import json
import time

import numpy as np
import polars as pl
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.embodied.agent import STAY_ON_TIE, LinearQAgent
from backend.embodied.brain import ConnectomeBrain
from backend.embodied.dodge_env import ACTION_NAMES, ENEMY_SIZE, FPS, HEIGHT, LEFT, PLAYER_SIZE, RIGHT, STAY, WIDTH, DodgeEnv
from backend.embodied.eye import GRID_COLS, GRID_ROWS, ego_view
from backend.embodied.train import FRAME_SKIP, RL_DIR
from backend.preprocessing.loaders import load_neurons

router = APIRouter()

# At most this many of the most strongly driven neurons per class are sent to the 3D view per decision.
DISPLAY_QUOTA = {
    "sensory": 400, "optic": 900, "visual_projection": 400, "visual_centrifugal": 80, "central": 600,
    "descending": 200, "ascending": 60, "sensory_ascending": 30, "motor": 20, "endocrine": 10,
}
DISPLAY_THRESHOLD = 0.03  # relative activity below this is not shown
SPEEDS = (1, 2, 4, 8)
POLICIES = [
    {"id": "connectome", "label": "Trained connectome readout"},
    {"id": "random", "label": "Random moves (baseline)"},
]
REPORT_NOTE = (
    "'random mixing' sends the same eye grid through a random matrix instead of the connectome. "
    "If it scores as well, the game score alone is not evidence about how real fly circuits work."
)


class GameRuntime:
    """Brain model and trained readout, loaded once and shared by all sessions."""

    def __init__(self):
        self.ready = False

    def load(self) -> None:
        if self.ready:
            return
        model_path = RL_DIR / "dodge_connectome_q.npz"
        if not model_path.exists():
            raise FileNotFoundError("No trained fly yet. Run: python -m backend.embodied.train")
        self.brain = ConnectomeBrain().load()
        self.agent = LinearQAgent.load(model_path)

        classes = load_neurons(columns=["super_class"])["super_class"].to_numpy()
        self.class_groups = []
        for name, quota in DISPLAY_QUOTA.items():
            index = np.flatnonzero(classes == name)
            scale = self.brain.neuron_scale[index]
            reference = float(np.percentile(scale[scale > 0], 99)) if np.any(scale > 0) else 1.0
            self.class_groups.append((index, min(quota, len(index)), reference))

        descending = self.brain.descending
        self.dn_ids = descending["neuron_id"].cast(pl.String).to_list()
        self.dn_types = descending["cell_type"].to_list()
        self.dn_sides = descending["side"].to_list()

        report_path = RL_DIR / "dodge_report.json"
        self.report = None
        if report_path.exists():
            data = json.loads(report_path.read_text(encoding="utf-8"))
            self.report = {"episodes": data["settings"]["eval_episodes"], "results": data["evaluation"], "note": REPORT_NOTE}
        self.ready = True

    def hello(self) -> dict:
        return {
            "type": "hello",
            "policies": POLICIES,
            "speeds": SPEEDS,
            "fps": FPS,
            "frame_skip": FRAME_SKIP,
            "grid": [GRID_ROWS, GRID_COLS],
            "photoreceptors": self.brain.layout.height,
            "descending": len(self.brain.descending_index),
            "hops": self.brain.hops,
            "arena": {"width": WIDTH, "height": HEIGHT, "player_size": PLAYER_SIZE, "enemy_size": ENEMY_SIZE},
            "report": self.report,
        }

    def brain_summary(self, activity: np.ndarray) -> dict:
        nodes, values = [], []
        for index, quota, reference in self.class_groups:
            relative = activity[index] / reference
            top = np.argpartition(-np.abs(relative), quota)[:quota] if quota < len(index) else np.arange(len(index))
            keep = top[np.abs(relative[top]) >= DISPLAY_THRESHOLD]
            nodes.extend(index[keep].tolist())
            values.extend(np.clip(np.round(relative[keep] * 100), -100, 100).astype(int).tolist())
        return {"nodes": nodes, "values": values}


class GameSession:
    def __init__(self, runtime: GameRuntime):
        self.runtime = runtime
        self.rng = np.random.default_rng()
        self.running = False
        self.speed = 1
        self.policy = "connectome"
        self.needs_send = True
        self.reset()

    def reset(self) -> None:
        self.env = DodgeEnv()
        self.action = STAY
        self.decision = None
        self.decision_id = 0
        self.sent_decision_id = 0
        self.hit_since_send = False
        self.needs_send = True

    def decide(self) -> None:
        runtime, brain, agent = self.runtime, self.runtime.brain, self.runtime.agent
        view = ego_view(self.env.player_x, self.env.enemies)
        phi = agent.features(brain.descending_activity(view))
        q = agent.q_values(phi)
        if self.policy == "connectome":
            self.action = int(np.argmax(q + STAY_ON_TIE))
        else:
            self.action = int(self.rng.integers(len(ACTION_NAMES)))

        # contribution of each descending neuron to preferring left over right (> 0 pushes left)
        push = (agent.weights[LEFT] - agent.weights[RIGHT]) * phi
        top = np.argsort(-np.abs(push))[:8]
        self.decision_id += 1
        self.decision = {
            "id": self.decision_id,
            "action": ACTION_NAMES[self.action],
            "policy": self.policy,
            "q": [round(float(v), 3) for v in q],
            "view": np.round(view, 2).tolist(),
            "brain": runtime.brain_summary(brain.activity(view)),
            "drivers": [
                {
                    "neuron_id": runtime.dn_ids[i],
                    "node_index": int(brain.descending_index[i]),
                    "cell_type": runtime.dn_types[i],
                    "side": runtime.dn_sides[i],
                    "push": round(float(push[i]), 3),
                }
                for i in top
            ],
        }

    def advance(self) -> None:
        if self.env.frame % FRAME_SKIP == 0:
            self.decide()
        self.hit_since_send |= self.env.frame_step(self.action).hit

    def message(self) -> dict:
        state = {"type": "frame", **self.env.snapshot(), "action": ACTION_NAMES[self.action], "hit": self.hit_since_send, "running": self.running}
        self.hit_since_send = False
        self.needs_send = False
        if self.decision and self.decision_id != self.sent_decision_id:
            state["decision"] = self.decision
            self.sent_decision_id = self.decision_id
        return state

    def handle(self, command: dict) -> None:
        name = command.get("cmd")
        if name == "start":
            if self.env.done:
                self.reset()
            self.running = True
        elif name == "pause":
            self.running = False
        elif name == "reset":
            self.reset()
            self.running = False
        elif name == "config":
            if command.get("speed") in SPEEDS:
                self.speed = command["speed"]
            if command.get("policy") in {p["id"] for p in POLICIES}:
                self.policy = command["policy"]
        self.needs_send = True


runtime = GameRuntime()


@router.websocket("/ws/game")
async def game_socket(websocket: WebSocket):
    await websocket.accept()
    try:
        await asyncio.to_thread(runtime.load)
    except FileNotFoundError as error:
        await websocket.send_json({"type": "error", "message": str(error)})
        await websocket.close()
        return

    session = GameSession(runtime)
    await websocket.send_json(runtime.hello())

    async def receive_commands():
        while True:
            session.handle(json.loads(await websocket.receive_text()))

    receiver = asyncio.create_task(receive_commands())
    try:
        while not receiver.done():
            tick = time.perf_counter()
            if session.running:
                for _ in range(session.speed):
                    session.advance()
                    if session.env.done:
                        session.running = False
                        break
                await websocket.send_json(session.message())
            elif session.needs_send:
                await websocket.send_json(session.message())
            await asyncio.sleep(max(0.0, 1 / FPS - (time.perf_counter() - tick)))
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        receiver.cancel()
