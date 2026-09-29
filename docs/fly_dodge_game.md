# Fly plays "Dodge the Falling Blocks"

A side project ahead of the main phase order: the FlyWire connectome is put inside a simple body.
A 3D fly plays the pygame game "Dodge the Falling Blocks" by itself. Its eyes feed real FlyWire
photoreceptors, the signal flows through the real wiring to the descending neurons, and a
reinforcement-learning readout turns that activity into left / stay / right. The web page shows the
game and the brain activity side by side.

Open it at `http://localhost:5173/#game` (backend and frontend running).

```
falling blocks ─► eye grid ─► R7/R8 photoreceptors ─► 4 hops through FlyWire ─► 1,303 descending neurons ─► Q-learning readout ─► move
```

Code: [backend/embodied/](../backend/embodied/) · live session: [backend/api/game.py](../backend/api/game.py) ·
page: [frontend/src/pages/FlyGamePage.jsx](../frontend/src/pages/FlyGamePage.jsx)

---

## 1. The game, without pygame

[dodge_env.py](../backend/embodied/dodge_env.py) is a frame-exact copy of the original rules:

- the screen is 1000 × 600 and runs at 80 FPS;
- the player is 30 px and moves 9 px per frame;
- blocks are 50 px and fall 5 px per frame, with at most 10 on screen;
- each frame a new block appears with probability 0.1;
- a block leaving the screen gives +1 score;
- the player has 3 attempts, and a hit clears the blocks and re-centres the player;
- the collision test is identical to the original.

Without a window, thousands of games can be simulated for training.
The fly decides every 4 frames (20 decisions per second) and holds that move in between.

## 2. The eyes

[eye.py](../backend/embodied/eye.py)

1. **Ego-centric view.** The screen is re-centred on the fly and cut into a 12 × 20 grid.
   Each cell stores how much of it is covered by blocks (0–1). Space beyond the screen edge shows up as a faint wall.
2. **Left half → left eye, right half → right eye.** Each eye's photoreceptors lie on a curved sheet.
   PCA of their 3D positions gives the sheet's two directions:
   - the direction closest to the dorsal–ventral axis becomes the grid rows;
   - the other direction becomes the columns, with the middle of the view on the medial side of the eye.
3. Photoreceptors are split into equal-count bins (≈11 per cell). A fully covered cell delivers a total drive of 1.

**Why only R7/R8.** The first version also used the outer photoreceptors R1-6. The whole bottom-left
part of the view (right next to the fly) then produced *exactly zero* response in every descending
neuron. The hop-by-hop trace showed why: in this dataset, R1-6 input from those eye regions stays
inside the lamina (R1-6 ↔ lamina neurons) for 4 hops and never reaches the medulla. R7 and R8 synapse
directly in the medulla. With R7/R8 only, every grid cell reaches the descending neurons and the
response matrix has full rank (240).

This is a simplified, documented encoding, not an optical model of the compound eye.

## 3. The brain model

[brain.py](../backend/embodied/brain.py)

$$W_{t,s} = \text{weight}(s \to t)\cdot \text{sign}(s), \qquad a_0 = P\,x, \qquad a_h = W\,a_{h-1}$$

$$\text{activity} = a_0 + \sum_{h=1}^{4} \frac{a_h}{g_h}$$

- $x$ is the eye grid (240 numbers) and $P$ maps each cell to its photoreceptors.
- `weight` is the input fraction from Phase 1; `sign` is +1 for acetylcholine, −1 for GABA or glutamate, and 0 otherwise.
- $g_h$ is the average size of hop $h$, so later hops are not drowned out.

Everything is linear in $x$, so the response of all 139,262 neurons to every grid cell is computed once:
$R = P + \sum_h W^h P / g_h$. That is a 139,262 × 240 matrix: 134 MB, built in about 15 seconds.
While playing, whole-brain activity is just $R\,x$, which takes about 8 ms.

Three hops from the eye already reach 1,174 descending neurons, and four hops reach 1,298 of the 1,303.

**Honest limits:** there are no spikes, no time constants and no nonlinearity. "Activity" means
"how strongly this neuron is connected to what the eye currently sees, under this linear model",
not a prediction of real firing.

## 4. Reinforcement learning

[agent.py](../backend/embodied/agent.py), [train.py](../backend/embodied/train.py)

**State.** $\phi$ = activity of the 1,303 descending neurons, standardized with the mean and standard deviation from random play.

**Actions.** left, stay, right.

**Q-function.** Linear: $Q(s,a) = w_a\cdot\phi(s) + b_a$.

**Q-learning update** (one step, semi-gradient):

$$\text{target} = r + \gamma \max_{a'} Q(s', a'), \quad \delta = \text{target} - Q(s,a), \quad w_a \leftarrow w_a + \frac{\alpha}{\lVert\phi\rVert^2 + 1}\,\delta\,\phi$$

Dividing by $\lVert\phi\rVert^2$ (normalized LMS) keeps the step size stable whatever the number of features.

| Setting | Value |
|---|---|
| reward | +0.1 per block that passes, −10 per hit |
| γ | 0.97 per decision (≈ 33 decisions ≈ 1.6 s look-ahead) |
| α | 0.1 |
| exploration | ε-greedy, ε from 1.0 to 0.02 over the first 70% of episodes |
| training | 400 games, each capped at 100 s |

**The connectome is never changed.** Only the readout weights $w$ and $b$ are learned.
In the terms of the main project plan, the connectome model is the fixed "graph representation"
and RL learns which action to take from it.

## 5. Results

Each policy played 30 new games of up to 100 s:

| Policy | Mean score | Hits / min | Survived the full 100 s |
|---|---|---|---|
| random moves | 15.7 | 25.3 | 0 / 30 |
| always stay | 20.6 | 22.6 | 0 / 30 |
| hand-written heuristic | 275.2 | 3.2 | 3 / 30 |
| Q-learning on the raw eye grid | 537.1 | 0.43 | 27 / 30 |
| **Q-learning on connectome descending neurons** | **543.0** | 0.53 | 28 / 30 |
| Q-learning on a random 1,303 × 240 mixing (control) | 541.5 | 0.47 | 28 / 30 |

**How to read this:**

- The connectome-driven fly learns to play well: about 35× the random score, and it survives almost every 100 s game.
- The random-mixing control does just as well. In this *linear* model, any mixing that keeps the visual
  information lets a learned readout recover it. So the score shows that the FlyWire wiring **relays**
  enough visual information to the descending neurons. It does **not** show that the fly uses these
  particular neurons to dodge, or that the real wiring is better than random wiring for this task.
- Before the R7/R8 fix, the connectome readout scored only ~30–60 because of the blind spot described in section 2.
  That was a real data property, found by comparing against the control.

## 6. What the page shows

- **Left panel:** the game in 3D. The fly is built from simple shapes: red compound eyes, beating wings, a banded abdomen and six legs. It banks as it moves, and the background changes colour as attempts run out, like the original game.
- **Middle panel:** the brain. Every decision, up to ~2,500 of the most strongly driven neurons are lit per class.
  Orange means net excitatory drive and blue means net inhibitory drive. Big dots are the 5 descending neurons pushing the choice hardest.
- **Right panel:**
  - the pipeline;
  - the Q-values for left / stay / right;
  - what the eyes receive;
  - which descending neurons tipped the decision (push = learned weight × activity);
  - the evaluation table.
- **Controls:** Start / Pause / Restart, speed 1–8×, and "trained connectome" vs "random moves".

## 7. Limitations and next steps

- The eye mapping and the transmitter signs are simplifications; see sections 2–3.
- A **nonlinear** model (for example, rectified activity at each hop) or a *degree-preserving shuffled
  connectome* control would be needed before saying anything about whether real wiring matters.
- The sound effects and music of the original game are not used.
- This RL agent plays the game. The RL path-finding agent of the main plan (Phase 8) is still separate and still to come.
