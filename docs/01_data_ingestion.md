# Phase 1 — Data ingestion

**Goal:** turn raw FlyWire FAFB v783 files into two clean tables that every later phase can trust:
a neuron table (future graph **nodes**) and a connection table (future graph **edges**).

```
raw files ──► read only needed columns ──► rename to one schema ──► clean & fill ──► neurons.parquet
                                                                              └──► edges.parquet
                                                                              └──► ingest_report.json
```

Code: [backend/preprocessing/ingest.py](../backend/preprocessing/ingest.py) ·
reading helpers: [loaders.py](../backend/preprocessing/loaders.py) ·
walkthrough: [notebooks/01_data_exploration.ipynb](../notebooks/01_data_exploration.ipynb)

---

## 1. Where the data comes from

The FlyWire Codex website offers the five CSV files named in the project brief, but downloading them needs a
FlyWire login. The same v783 release is also published without a login:

| What | Login-free file (used now) | Codex file (optional) |
|------|----------------------------|-----------------------|
| neuron annotations, cell types, neurotransmitter, position | `Supplemental_file1_neuron_annotations.tsv` | `neurons`, `classification`, `consolidated_cell_types`, `coordinates` |
| neuron → neuron synapse counts | `Connectivity_783.parquet` | `connections_princeton` |

Both paths end in exactly the same output tables, so nothing downstream cares which one was used.
If the Codex files are placed in `backend/data/raw/codex/`, they are used automatically.

## 2. Identifying the important columns

This is the "which column means what" step (tasks 3–8 of the brief).

| Concept | Public file column | Codex column | Canonical name |
|---------|--------------------|--------------|----------------|
| neuron ID | `root_id` | `root_id` | `neuron_id` |
| presynaptic (sending) neuron | `Presynaptic_ID` | `pre_root_id` | `source_id` |
| postsynaptic (receiving) neuron | `Postsynaptic_ID` | `post_root_id` | `target_id` |
| connection strength | `Connectivity` | `syn_count` | `synapse_count` |
| position | `pos_x/y/z` | `position` ("[x y z]") | `x`, `y`, `z` (nm) |
| cell body position | `soma_x/y/z` | — | `soma_x/y/z` (nm) |
| classification | `flow`, `super_class`, `cell_class`, `cell_sub_class` | `flow`, `super_class`, `class`, `sub_class` | same names |
| cell type | `cell_type` | `primary_type` / `cell_type` | `cell_type` |
| neurotransmitter | `top_nt`, `top_nt_conf` | `nt_type`, `nt_type_score` | `neurotransmitter`, `nt_confidence` |
| brain region | — | `neuropil` | `main_neuropil` |

Run `python -m backend.preprocessing.ingest --inspect` to see every raw column with its data type and null count.

## 3. What the raw data looks like

| | count |
|---|---|
| annotated neurons | 139,248 |
| neurons that appear in the connectivity table | 138,639 |
| neurons in connectivity but **not** annotated | 14 |
| annotated neurons with **no** connection | 623 |
| directed neuron pairs (edges) | 15,091,983 |
| synapses in those edges | 54,492,922 |
| duplicate pairs / self-loops / invalid rows | 0 / 0 / 0 |

Final neuron table: **139,262** neurons (the union of both lists, so no connection loses its endpoint).

## 4. Missing values — what was done and why

| Column | Missing | Decision | Why |
|--------|---------|----------|-----|
| `x, y, z` | 14 | keep null, `has_position = false` | inventing a position would put a fake dot in the 3D view |
| `soma_x/y/z` | 21,158 (15%) | keep null, `has_soma = false` | 19,252 of these are afferent (sensory / ascending) neurons, whose cell bodies lie outside the imaged brain |
| `cell_class` | 31,744 (23%) | `"unknown"` | a category is still useful as a model input; "unknown" is honest |
| `cell_sub_class` | 113,434 (81%) | `"unknown"` | same |
| `cell_type` | 1,542 (1%) | `"unknown"` | same |
| `neurotransmitter` | 616 | `"unknown"`, confidence null | same |
| `status` (outlier flags) | 658 flagged | turned into `is_outlier` | lets later phases exclude badly segmented neurons |

Rule used everywhere: **never fill a number with a made-up number.** Categories get `"unknown"`,
numbers stay null and get a boolean flag.

## 5. The mathematics

### 5.1 Voxels → nanometres
The electron-microscopy volume is cut into sections 40 nm thick and each image pixel is 4 nm wide.
The annotation file stores positions in voxels, so:

$$x_{nm} = 4\,x_{vox}, \quad y_{nm} = 4\,y_{vox}, \quad z_{nm} = 40\,z_{vox}$$

Without this, the brain would look 10× too thin along z in the 3D viewer.

### 5.2 Edge weight = input fraction
A raw synapse count is hard to compare: 10 synapses matter a lot for a small neuron and very little for one
that receives 10,000. So each edge also gets a normalized weight:

$$w(A \to B) = \frac{s(A \to B)}{\sum_{k} s(k \to B)}$$

where $s$ is the synapse count and the sum runs over every neuron $k$ that connects to $B$.
For every target neuron the incoming weights add up to exactly 1 (checked in the notebook: min = max = 1.0).

### 5.3 Edge sign
A predicted neurotransmitter gives the sign of the *sending* neuron:
acetylcholine → **+1** (excitatory), GABA and glutamate → **−1** (inhibitory),
dopamine / serotonin / octopamine / unknown → **0** (modulatory or not known).
Result: 8,187,699 excitatory, 6,071,458 inhibitory, 832,826 unknown edges.
This is a common modelling convention (Shiu et al. 2024), **not** a measurement of each synapse.

### 5.4 Why IDs must stay 64-bit integers
FlyWire IDs such as `720575940596125868` are larger than $2^{53} = 9{,}007{,}199{,}254{,}740{,}992$,
the largest integer a `float64` stores exactly. Converted to float, that ID becomes `720575940596125824`,
a *different* neuron ID. pandas converts an integer column to `float64` as soon as one value is missing,
so the pipeline keeps IDs as `Int64` from the moment the file is read.

### 5.5 Most connections are weak
Synapse counts per edge: median 2, 90th percentile 7, 99th percentile 32, maximum 2,405.

| minimum synapses | edges kept |
|---|---|
| 1 | 15,091,983 |
| 2 | 7,595,967 |
| 5 | 2,700,513 |
| 10 | 1,066,822 |

One- and two-synapse edges are the most likely to be detection errors. Phase 2 will choose a threshold
(usually ≥ 5) when building the graph; Phase 1 keeps everything so that choice stays open.

## 6. Small example — neuron DNa02 (right side)

```
neuron_id 720575940604737708   cell_type DNa02   super_class descending   neurotransmitter acetylcholine
position  (612,064  170,248  83,440) nm
incoming connections 751, outgoing connections 213

strongest inputs                         strongest outputs
source   synapses  weight   sign         target    synapses  weight   sign
PS049       378    0.0287    -1          PS137        48     0.0213    +1
DNa03       264    0.0201    +1          PS100        39     0.0049    +1
DNae005     240    0.0182    +1          PS137        39     0.0169    +1
AOTU019     216    0.0164    -1          DNge026      39     0.0157    +1
LAL018      185    0.0141    +1          PS274        35     0.0069    +1
```

Reading it: PS049 → DNa02 has 378 synapses, which is 2.9% of everything DNa02 receives
(so DNa02 receives about 378 / 0.0287 ≈ 13,000 synapses in total). PS049 is predicted to release an
inhibitory transmitter, so the sign is −1. On the output side every sign is +1 because DNa02 itself
is predicted to be cholinergic.

## 7. Memory and disk decisions

* Only the needed columns are read (17 of 31 annotation columns, 3 of 8 connectivity columns).
* The connectivity file is scanned lazily with polars and aggregated with its streaming engine.
* Output is Parquet with zstd compression: `neurons.parquet` 3.7 MB, `edges.parquet` 163 MB.
* Later phases use `load_edges(min_synapses=5, columns=[...])`, which reads just what they need.
* The full build takes about 9 seconds on a Ryzen 7 5800HS.

## 8. Why this matters for the next phases

* `node_index` (0 … N−1) is what NetworkX, SciPy and PyTorch Geometric will use as node numbers.
* `synapse_count` and `weight` become edge weights for PageRank, shortest and strongest paths.
* `super_class`, `cell_type`, `neurotransmitter` and `x, y, z` become node features for the GNN.
* `has_position` and `is_outlier` let the 3D viewer and the models skip unreliable neurons.

## 9. Known limitations

* The sign rule is an assumption (see 5.3).
* Positions are one representative point per neuron, not the full shape. Shapes (meshes/skeletons) are large
  and will be fetched only for a few selected neurons in Phase 6.
* The Codex reader is tested on synthetic files using the documented column names; a real Codex export
  with renamed columns will stop with an error listing the columns it found.
