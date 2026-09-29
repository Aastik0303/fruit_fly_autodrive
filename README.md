# NeuroPath AI — FlyWire 3D Connectome Circuit Analysis

A learning-first research project on the FlyWire FAFB **v783** whole-brain connectome of *Drosophila melanogaster*
(about 139,000 neurons and 15 million neuron-to-neuron connections).
The project is built one phase at a time: classical graph analysis first, then graph neural networks,
then reinforcement learning, with an interactive 3D viewer and an API on top.

## Status

| # | Phase | Status |
|---|-------|--------|
| 1 | Data ingestion | ✅ done — [docs/01_data_ingestion.md](docs/01_data_ingestion.md) |
| 2 | Graph construction (NetworkX) | next |
| 3 | Basic graph analysis | |
| 4 | Path analysis | |
| 5 | Circuit analysis | |
| 6 | 3D visualization (React Three Fiber) | |
| 7 | GNN (PyTorch Geometric, CPU) | |
| 8 | GNN-based ranking | |
| 9 | Reinforcement learning (Q-learning) | |
| 10 | GNN + RL integration | |
| 11 | FastAPI backend | |
| 12 | AI explanation layer | |

## Setup (Windows, PowerShell)

```powershell
py -3.13 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt

# 1. get the login-free v783 files (~130 MB)
.venv\Scripts\python.exe -m backend.preprocessing.download

# 2. look at the raw columns, then build the normalized tables (~10 s)
.venv\Scripts\python.exe -m backend.preprocessing.ingest --inspect
.venv\Scripts\python.exe -m backend.preprocessing.ingest

# 3. tests
.venv\Scripts\python.exe -m pytest -q
```

### Run the backend and the 3D viewer

```powershell
cd frontend; npm install; cd ..                                   # first time only
powershell -ExecutionPolicy Bypass -File .\start-dev.ps1          # opens two windows
```

or manually, in two terminals:

```powershell
.venv\Scripts\python.exe -m uvicorn backend.api.main:app --port 8000   # API, docs at http://127.0.0.1:8000/docs
cd frontend; npm run dev                                               # viewer at http://localhost:5173
```

The backend needs ~10 s and about 600 MB RAM to load the connectome. Available endpoints so far:
`GET /stats`, `GET /neurons/search?q=`, `GET /neurons/{id}`, `GET /neurons/{id}/neighbors`,
`GET /neurons/{id}/connections`, `GET /visualization/neurons`. Path, circuit, GNN and RL endpoints
arrive with their phases.

### Fly plays "Dodge the Falling Blocks"

A side project that puts the connectome in a body: a 3D fly plays the dodge game while its brain
activity is shown next to it. Details and results: [docs/fly_dodge_game.md](docs/fly_dodge_game.md).

```powershell
.venv\Scripts\python.exe -m backend.embodied.brain    # eye -> connectome response matrix (~15 s, 134 MB)
.venv\Scripts\python.exe -m backend.embodied.train    # Q-learning agents + evaluation (a few minutes)
# then start the backend and frontend and open http://localhost:5173/#game
```

Python 3.13 is used because PyTorch Geometric (Phase 7) does not yet support every newer Python release.
Everything runs on CPU; no GPU is required.

## Data

Two interchangeable sources produce the same tables:

| Source | Files | Login |
|--------|-------|-------|
| `public` (default) | `Supplemental_file1_neuron_annotations.tsv` from [flyconnectome/flywire_annotations](https://github.com/flyconnectome/flywire_annotations), `Connectivity_783.parquet` from [philshiu/Drosophila_brain_model](https://github.com/philshiu/Drosophila_brain_model) | no |
| `codex` | `neurons`, `classification`, `consolidated_cell_types`, `connections_princeton`, `coordinates` (`.csv` or `.csv.gz`) from [codex.flywire.ai](https://codex.flywire.ai) | free FlyWire account |

To use Codex files, sign in at codex.flywire.ai, download the five FAFB v783 files and put them in
`backend/data/raw/codex/`. The pipeline then picks them up automatically (`--source auto`).
The Codex reader has so far only been tested on synthetic files that follow the documented column names;
if a real export uses a different column name, the error message lists the columns it found.

### Output tables (`backend/data/processed/`)

`neurons.parquet`

| column | meaning |
|--------|---------|
| `node_index` | 0 … N-1, contiguous index used by graph and GNN code |
| `neuron_id` | FlyWire root ID (Int64, never float) |
| `x`, `y`, `z` | representative point on the neuron, nanometres |
| `soma_x`, `soma_y`, `soma_z` | cell body position, nanometres (missing for ~15% of neurons) |
| `flow`, `super_class`, `cell_class`, `cell_sub_class`, `cell_type`, `hemibrain_type`, `side` | hierarchical annotations, `unknown` when missing |
| `neurotransmitter`, `nt_confidence` | predicted transmitter and its confidence |
| `has_annotation`, `has_position`, `has_soma`, `has_connections`, `is_outlier` | data-quality flags |

`edges.parquet`

| column | meaning |
|--------|---------|
| `source_index`, `target_index` | node indices of the presynaptic and postsynaptic neuron |
| `source_id`, `target_id` | FlyWire root IDs |
| `synapse_count` | number of synapses from source to target |
| `weight` | `synapse_count / total synapses received by target` (input fraction, 0–1) |
| `sign` | +1 excitatory, −1 inhibitory, 0 unknown (modelling convention, see docs) |
| `main_neuropil` | brain region with most of the synapses (Codex source only) |

## Project layout

```
backend/
    data/            raw/ and processed/ data (not in git)
    preprocessing/   download.py, ingest.py, loaders.py
    utils/           config.py
    graph/ analysis/ circuits/ gnn/ rl/ api/   (later phases)
frontend/            React + Three.js viewer (Phase 6)
models/              trained GNN and RL models
notebooks/           01_data_exploration.ipynb, ...
docs/                one explanation document per phase
tests/
```

## Credits

FlyWire data: Dorkenwald et al. 2024 and Schlegel et al. 2024 (*Nature*), FlyWire Consortium.
Connectivity table and neurotransmitter sign convention: Shiu et al. 2024 (*Nature*).
Neurotransmitter predictions: Eckstein, Bates et al. 2024 (*Cell*).
Please follow the citation and licence terms of FlyWire and of the source repositories when using the data.
