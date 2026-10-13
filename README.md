# EQLG

Research implementation of **Entropy-normalized Quasi-Laplacian Gravity
Centrality (EQLG)** for identifying influential nodes in static, undirected,
unweighted networks.

The repository accompanies the manuscript *Identifying influential nodes in
complex networks with entropy-normalized quasi-Laplacian gravity centrality*.
It contains the unchanged EQLG implementation and the experiment code used in
the revision, including the new analyses in Sections 4.4.1-4.4.3.

## Repository layout

```text
EQLG/
|- src/
|  |- eqlg.py                 # EQLG implementation
|  |- sir.py                  # SIR influence simulation
|  `- metrics.py              # Kendall, Jaccard, monotonicity
|- experiments/
|  |- main_experiment.py
|  |- radius_sensitivity.py   # Section 4.4.1
|  |- runtime_test.py         # Section 4.4.2
|  `- scalability_test.py     # Section 4.4.3
|- config/parameters.yaml
|- data/README.md
`- requirements.txt
```

## Installation

Python 3.10 or later is recommended.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

Run commands from the repository root.  Experiment outputs are written below
`results/` by default; this directory is intentionally not versioned.

## Core method

```python
import networkx as nx
from src.eqlg import eqlg_centrality

G = nx.karate_club_graph()
scores = eqlg_centrality(G, radius=3)
```

The default truncation radius in the main experiments is `r = 3`.

## Real-network input

Raw datasets are not redistributed.  Sources and preprocessing conventions are
listed in `data/README.md`.  The revision scripts accept undirected edge lists
through a small JSON manifest:

```json
{
  "Dolphins": {
    "edge_list": "data/dolphins.txt",
    "node_type": "int",
    "delimiter": null,
    "comments": "#",
    "sir_scores": "results/sir/dolphins.json"
  }
}
```

Paths are resolved relative to the manifest.  `node_type` may be `str`, `int`,
or `float`.  The `sir_scores` field is needed only by the radius-sensitivity
experiment.  Its JSON file stores the five cached SIR score dictionaries:

```json
{
  "0.8": {"0": 1.23, "1": 1.41},
  "0.9": {"0": 1.36, "1": 1.57},
  "1.0": {"0": 1.52, "1": 1.83},
  "1.1": {"0": 1.79, "1": 2.12},
  "1.2": {"0": 2.05, "1": 2.44}
}
```

Each file must contain every graph node.  Generate these values with
`src/sir.py` using `gamma = 1`, 1000 simulations per source, and infection
probabilities `0.8, 0.9, 1.0, 1.1, 1.2` times the network epidemic threshold.

## Section 4.4.1: radius sensitivity

The script evaluates `r = 1, 2, 3, 4` using Kendall correlation, Jaccard
similarity for the top 20% of nodes, and monotonicity.  It writes both the
per-SIR-setting values and the averages over the five SIR settings.

```bash
python experiments/radius_sensitivity.py --manifest experiment_manifest.json
```

Default outputs:

- `results/radius_sensitivity_detail.csv`
- `results/radius_sensitivity_summary.csv`

## Section 4.4.2: runtime and radius-dependent cost

Wall-clock time is measured with `time.perf_counter`.  Graph loading,
structural statistics, diameter calculation, and radius-coverage estimation
are outside the timed region.  The default protocol uses one untimed warm-up
and three measured repetitions for each `r = 1, 2, 3, 4`.

```bash
python experiments/runtime_test.py --manifest experiment_manifest.json
```

The output `results/radius_runtime.csv` reports network size, density,
diameter, mean radius-limited coverage, and the mean/standard deviation of
runtime.  Coverage uses every source node by default; pass
`--coverage-sample-size K` for deterministic sampling on larger graphs.

For the method-to-method timing table, import `compare_methods_runtime` and
provide the exact callable used for each baseline.  This keeps all methods
under the same timing wrapper without silently substituting different
implementations:

```python
from experiments.runtime_test import compare_methods_runtime

records = compare_methods_runtime(
    G,
    {
        "DC": degree_method,
        "PR": pagerank_method,
        "EQLG": lambda graph: eqlg_centrality(graph, radius=3),
    },
)
```

## Section 4.4.3: scalability and memory

The controlled scalability series uses Barabasi-Albert networks with
`N = 1,000, 5,000, 10,000, 20,000, 50,000`, attachment parameter `m = 3`,
and EQLG radius `r = 3`.  Network generation is excluded from measurement.
Peak memory is measured with Python's `tracemalloc` and reported in MiB.  After
one untimed warm-up, three runs are measured.  The script also estimates mean
three-hop coverage from a deterministic sample of up to 1000 sources and fits
log-log runtime and memory exponents.

```bash
python experiments/scalability_test.py
```

Default outputs:

- `results/scalability.csv` - Table 8 source data
- `results/scalability_fit.json` - empirical scaling exponents
- `results/scalability.png` - log-log runtime and memory figure

Runtime and memory values depend on the Python version, NetworkX version,
operating system, and hardware.  Record that environment when comparing a new
run with the manuscript values.  In the reported environment, the largest
network had 50,000 nodes and 149,991 edges; the corresponding results were
approximately 166.3 s and 78.9 MB, with fitted runtime and memory exponents of
1.34 and 1.00, respectively.

## Experimental parameters

The complete revision protocol is recorded in `config/parameters.yaml`:

| Analysis | Setting |
|---|---|
| Default EQLG radius | `3` |
| Radius sensitivity/cost | `1, 2, 3, 4` |
| Jaccard top fraction | `20%` |
| Timing repetitions | `3` after `1` warm-up |
| Scalability graph | BA, `m = 3` |
| Scalability sizes | `1k, 5k, 10k, 20k, 50k` |
| Scalability radius | `3` |
| Random seed | `42` |

## Baselines

The study compares EQLG with DC, KC, BC, PageRank, ILC, GLC, MSIS, KSGC, and
WDKS.  Classical NetworkX methods may be passed directly to the timing API.
For recent methods, use the formulations and implementations cited in the
manuscript; the repository does not relabel approximate reimplementations as
the original baselines.

## Citation

If you use this code, please cite:

Siqi Zhu, Jie Zhan, Yan Hui, and Shibo Wei. *Identifying influential nodes in
complex networks with entropy-normalized quasi-Laplacian gravity centrality*.

## License

MIT License.

