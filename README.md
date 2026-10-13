# EQLG

Research implementation of **Entropy-normalized Quasi-Laplacian Gravity Centrality (EQLG)** for identifying influential nodes in static, undirected, and unweighted complex networks.

This repository accompanies the manuscript *Identifying influential nodes in complex networks with entropy-normalized quasi-Laplacian gravity centrality*.

The repository intentionally provides the **core method, evaluation functions, experimental parameters, and experiment logic** required to understand and reproduce the reported analyses. It does **not** redistribute the complete private research project, local server paths, intermediate files, or third-party baseline source code. Users are expected to provide their own dataset paths, computing environment, and baseline implementations where applicable.

## Repository layout

```text
EQLG/
|- src/
|  |- eqlg.py                 # EQLG implementation
|  |- sir.py                  # SIR spreading simulation
|  `- metrics.py              # Kendall, Jaccard, monotonicity
|- experiments/
|  |- main_experiment.py      # Main evaluation logic
|  |- radius_sensitivity.py   # Section 4.4.1
|  |- runtime_test.py         # Section 4.4.2
|  `- scalability_test.py     # Section 4.4.3
|- config/
|  `- parameters.yaml         # Final experimental settings
|- data/
|  `- README.md               # Dataset sources and preprocessing
|- requirements.txt
`- LICENSE
```

## Environment

The wall-clock experiments reported in the revised manuscript were conducted using **Python 3.9.23** and **NetworkX 3.2.1** on a Linux server with 32 logical CPU cores and approximately 98 GB of memory.

The public research code is intentionally lightweight. Install the required Python packages listed in `requirements.txt` in an environment appropriate for your system.

## Core EQLG method

```python
import networkx as nx
from src.eqlg import eqlg_centrality

G = nx.karate_club_graph()
scores = eqlg_centrality(G, radius=3)
```

The default truncation radius used in the main experiments is `r = 3`.

## Data and preprocessing

Raw datasets are not redistributed in this repository. Dataset sources, node/edge counts, and preprocessing conventions are described in `data/README.md`.

The manuscript uses eight real-world networks: Dolphins, Email, NS, Power, Router, Euroroad, Wikivote, and Yeast. All are treated as static, undirected, and unweighted graphs.

Users should obtain the original datasets from the cited public sources, construct NetworkX `Graph` objects, and provide them directly to the experiment functions.

## Main experimental evaluation

`experiments/main_experiment.py` contains the main experimental logic used to compare structural rankings with SIR-based spreading influence.

The experimental setting uses:

- recovery probability `gamma = 1.0`;
- five infection probabilities: `0.8, 0.9, 1.0, 1.1, 1.2` times the epidemic threshold;
- 1000 independent SIR simulations for each source node;
- Kendall correlation;
- Jaccard similarity for the top 20% influential nodes;
- monotonicity for score discrimination.

Baseline score dictionaries can be supplied externally to the main evaluation function.

## Section 4.4.1: truncation-radius sensitivity

`experiments/radius_sensitivity.py` provides the experiment logic for evaluating EQLG with `r = 1, 2, 3, 4` on the eight real-world networks.

Users provide:

- a mapping from network names to NetworkX graphs;
- five SIR reference-score dictionaries for each network.

The function `evaluate_radius_sensitivity(...)` returns:

- detailed results for every network, radius, and SIR setting;
- per-network averages;
- the cross-network summary corresponding to Table 6.

Example:

```python
from experiments.radius_sensitivity import evaluate_radius_sensitivity

results = evaluate_radius_sensitivity(
    graphs=my_graphs,
    sir_scores_by_network=my_sir_scores,
)

table6 = results["overall"]
```

## Section 4.4.2: runtime and radius-dependent computational cost

`experiments/runtime_test.py` provides two related timing procedures.

First, `compare_methods_runtime(...)` applies the same timing wrapper to caller-supplied implementations of DC, KC, BC, PR, ILC, GLC, MSIS, KSGC, WDKS, and EQLG. The manuscript timing experiment uses Dolphins, Email, Power, and Router, with **one untimed warm-up followed by five measured executions**.

Second, `radius_runtime_analysis(...)` evaluates EQLG at `r = 1, 2, 3, 4` under the same timing protocol.

Only centrality-score computation is timed. Dataset loading, SIR simulation, output writing, and structural-statistic calculation are excluded from the timed region.

Example:

```python
from experiments.runtime_test import (
    compare_methods_runtime,
    eqlg_at_radius,
    radius_runtime_analysis,
)

method_functions = {
    "DC": degree_method,
    "KC": kcore_method,
    "BC": betweenness_method,
    "PR": pagerank_method,
    "ILC": ilc_method,
    "GLC": glc_method,
    "MSIS": msis_method,
    "KSGC": ksgc_method,
    "WDKS": wdks_method,
    "EQLG": eqlg_at_radius(3),
}

runtime_table = compare_methods_runtime(
    graphs=representative_graphs,
    method_functions=method_functions,
)

radius_table = radius_runtime_analysis(
    graphs=representative_graphs,
)
```

Recent baseline source code is not redistributed. Users should use the original formulations or implementations corresponding to the cited methods.

## Section 4.4.3: scalability and memory

`experiments/scalability_test.py` contains the controlled scalability and memory-analysis logic used in the revised manuscript.

The final protocol is:

- Barabasi-Albert networks with `N = 1,000, 5,000, 10,000, 20,000, 50,000`;
- attachment parameter `m = 3`;
- EQLG radius `r = 3`;
- base random seed `2026`;
- graph seed `2026 + N`;
- three measured EQLG executions for each network size;
- 200 sampled source nodes for estimating three-hop coverage;
- a small untimed warm-up on a BA network with 500 nodes;
- `tracemalloc` started before graph construction so graph storage is included in peak traced memory.

Example:

```python
from experiments.scalability_test import run_scalability_analysis

rows, fit = run_scalability_analysis()
```

The returned rows contain the measurements used for Table 8, including runtime, graph traced memory, peak total traced memory, EQLG memory overhead, and three-hop coverage. The returned `fit` dictionary contains the empirical log-log runtime and memory scaling exponents.

In the reported environment, the largest tested network contained 50,000 nodes and 149,991 edges. The measured EQLG runtime was approximately **166.3 s**, peak traced memory was approximately **78.9 MB**, and the fitted runtime and memory exponents were approximately **1.34** and **1.00**, respectively.

Runtime and memory values can vary with Python version, NetworkX version, operating system, allocator behavior, and hardware.

## Experimental parameters

The final revision settings are summarized in `config/parameters.yaml`. Important values include:

| Analysis | Setting |
|---|---|
| Default EQLG radius | `3` |
| Radius sensitivity | `1, 2, 3, 4` |
| Jaccard top fraction | `20%` |
| SIR simulations per source | `1000` |
| Runtime repetitions | `5` after `1` warm-up |
| Scalability repetitions | `3` |
| Scalability BA parameter | `m = 3` |
| Scalability sizes | `1k, 5k, 10k, 20k, 50k` |
| Scalability seed | `2026 + N` |
| Three-hop coverage sample | `200` source nodes |

## Baseline methods and original references

The manuscript compares EQLG with four established centrality measures and five recent influential-node identification methods. Third-party source code is not redistributed in this repository. The corresponding original formulations are referenced below.

| Method | Original/reference formulation used in the manuscript |
|---|---|
| DC | Freeman LC. *Centrality in social networks conceptual clarification*. Social Networks, 1978, 1:215-239. DOI: 10.1016/0378-8733(78)90021-7 |
| KC | Kitsak M et al. *Identification of influential spreaders in complex networks*. Nature Physics, 2010, 6:888-893. DOI: 10.1038/nphys1746 |
| BC | Barthelemy M. *Betweenness centrality in large complex networks*. European Physical Journal B, 2004, 38:163-168. DOI: 10.1140/epjb/e2004-00111-4 |
| PR | Brin S, Page L. *The anatomy of a large-scale hypertextual Web search engine*. Computer Networks and ISDN Systems, 1998, 30:107-117. DOI: 10.1016/S0169-7552(98)00110-X |
| KSGC | Yang X, Xiao F. *An improved gravity model to identify influential nodes in complex networks based on k-shell method*. Knowledge-Based Systems, 2021, 227:107198. DOI: 10.1016/j.knosys.2021.107198 |
| MSIS | Ma J, Hu J. *Identifying critical nodes in complex networks via a Multi-Scale Influence Spread method*. Chaos, Solitons & Fractals, 2025, 198:116573. DOI: 10.1016/j.chaos.2025.116573 |
| WDKS | Zhu X, Hao R. *Finding influential nodes in complex networks by integrating nodal intrinsic and extrinsic centrality*. Chaos, Solitons & Fractals, 2025, 194:116278. DOI: 10.1016/j.chaos.2025.116278 |
| ILC | Zhu X, Hao R. *Identifying influential nodes in social networks via improved Laplacian centrality*. Chaos, Solitons & Fractals, 2024, 189:115675. DOI: 10.1016/j.chaos.2024.115675 |
| GLC | Ruan Y et al. *GLC: A dual-perspective approach for identifying influential nodes in complex networks*. Expert Systems with Applications, 2025, 268:126292. DOI: 10.1016/j.eswa.2024.126292 |

## Reproducibility scope

This repository is designed as a compact research-code release rather than a packaged software application. It therefore intentionally avoids:

- hard-coded local or server paths;
- redistribution of raw third-party datasets;
- redistribution of third-party baseline source code;
- private intermediate files and notebooks;
- a one-command pipeline tied to the authors' local environment.

Users can supply their own graph objects, SIR reference results, and baseline implementations while using the same EQLG core functions and experimental logic reported in the manuscript.

## Citation

If you use this code, please cite:

Siqi Zhu, Jie Zhan, Yan Hui, and Shibo Wei. *Identifying influential nodes in complex networks with entropy-normalized quasi-Laplacian gravity centrality*.

## License

MIT License.
