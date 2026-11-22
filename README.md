# EQLG

Research implementation of **Entropy-normalized Quasi-Laplacian Gravity Centrality (EQLG)** for identifying influential nodes in static, undirected, and unweighted complex networks.

This repository accompanies the manuscript *Identifying influential nodes in complex networks with entropy-normalized quasi-Laplacian gravity centrality*.

The repository provides the EQLG method, SIR-based evaluation functions, performance metrics, experimental parameters, dataset-preprocessing information, and reusable experiment logic. It does not redistribute raw third-party datasets, private intermediate files, or third-party baseline implementations.

## Repository layout

```text
EQLG/
|- src/
|  |- eqlg.py                 # EQLG implementation
|  |- sir.py                  # SIR spreading simulation
|  `- metrics.py              # Kendall, Jaccard, monotonicity
|- experiments/
|  |- main_experiment.py      # Main effectiveness-evaluation logic
|  |- radius_sensitivity.py   # Truncation-radius sensitivity
|  |- runtime_test.py         # Runtime and radius-cost wrappers
|  `- scalability_test.py     # BA scalability and memory analysis
|- config/
|  `- parameters.yaml         # Experimental settings
|- data/
|  `- README.md               # Dataset sources and preprocessing
|- supplementary/
|  |- README.md               # Supplementary runtime-data description
|  `- Supplementary_Table_S1_Learning_Runtime_8Networks.xlsx
|- requirements.txt
`- LICENSE
```

## Environment

The wall-clock experiments reported in the manuscript were conducted using **Python 3.9.23** and **NetworkX 3.2.1** on a Linux server with 32 logical CPU cores and approximately 98 GB of memory. Learning-based profiling used an NVIDIA GeForce RTX 4090 with PyTorch 2.7.1 and CUDA 12.8.

Install the public code dependencies with:

```bash
pip install -r requirements.txt
```

## Core EQLG method

```python
import networkx as nx
from src.eqlg import eqlg_centrality

G = nx.karate_club_graph()
scores = eqlg_centrality(G, radius=3)
```

The default truncation radius in the main experiments is `r = 3`.

## Data and preprocessing

Raw datasets are not redistributed. Sources, processed node/edge counts, and preprocessing conventions are documented in [`data/README.md`](data/README.md).

The main effectiveness evaluation uses eight real-world networks:

- Dolphins
- Email
- NS
- Power
- Router
- Euroroad
- Wikivote
- Yeast

PG, CM, and TO are additional large real-world networks used only for computational scalability. They are not part of the eight-network SIR effectiveness evaluation.

## Main effectiveness evaluation

`experiments/main_experiment.py` compares structural rankings with SIR-based spreading influence. EQLG is evaluated against eleven baselines spanning classical centralities, recent topology-based influential-node methods, and learning-based approaches:

- classical centralities: DC, KC, BC, and PR;
- recent topology-based methods: KSGC, MSIS, WDKS, ILC, and GLC;
- external learning-based methods: 1D-CGS and SGNN.

This gives **12 evaluated methods in total**, including EQLG.

The main setting uses:

- recovery rate `gamma = 1.0`;
- five transmission-rate settings: `0.8, 0.9, 1.0, 1.1, 1.2` times `beta_c`;
- 1000 independent SIR simulations for each source node;
- Kendall correlation;
- top-20% Jaccard similarity;
- monotonicity for score discrimination.

The public SIR utility is a standalone continuous-time implementation using exponential transmission and recovery waiting times. Its existing interface names and previously published defaults are retained; `beta_c` supplies a degree-based reference scale.

Evaluation uses stable descending rankings (input order resolves ties), `ceil(n * fraction)` nodes for top-fraction Jaccard, and scores rounded to 12 decimal places when counting monotonicity ties.

Baseline score dictionaries are supplied externally to the evaluation function. At `beta = beta_c`, EQLG ranks first on seven of the eight networks. Email is the exception, where 1D-CGS achieves a slightly higher Kendall correlation.

## Truncation-radius sensitivity

`experiments/radius_sensitivity.py` evaluates EQLG with `r = 1, 2, 3, 4` on the eight main networks using the same five SIR settings and the three main evaluation metrics.

```python
from experiments.radius_sensitivity import evaluate_radius_sensitivity

results = evaluate_radius_sensitivity(
    graphs=my_graphs,
    sir_scores_by_network=my_sir_scores,
)

table6 = results["overall"]
```

## Computational experiments

### Eight-network topology-based runtime comparison

The unified deterministic runtime comparison covers all eight main networks and uses one untimed warm-up followed by five measured executions. The methods are DC, KC, BC, PR, ILC, GLC, MSIS, KSGC, WDKS, and EQLG.

`experiments/runtime_test.py` provides the common timing wrapper. Callers supply the exact method implementations so that every method uses the same timing boundary. The two learning-based baselines are evaluated separately because training, neural inference, and post-processing do not follow the same computational pipeline as deterministic topology-based score computation.

### Synthetic BA scalability

The controlled BA experiment uses:

- `N = 1,000, 5,000, 10,000, 20,000, 50,000`;
- attachment parameter `m = 3`;
- EQLG radius `r = 3`;
- compared methods DC, KC, PR, ILC, MSIS, WDKS, and EQLG.

At `N = 50,000`, EQLG requires approximately **74.94 s** and **115.23 MB** peak traced memory in the reported environment. EQLG does not scale faster than the simpler baselines. It trades additional computation for richer quasi-local structural information.

`experiments/scalability_test.py` provides the public EQLG runtime, traced-memory, and three-hop-coverage logic. Third-party baseline implementations are not redistributed, so the seven-method comparison requires callers to supply the corresponding external callables.

### Additional large real-world scalability

Three KONECT networks are used only for computational scalability:

| Network | Description | Nodes | Edges | Mean EQLG runtime |
|---|---|---:|---:|---:|
| PG | Pretty Good Privacy | 10,680 | 24,316 | approximately 2.03 s |
| CM | Condensed matter | 27,519 | 116,181 | approximately 19.94 s |
| TO | Internet topology | 34,761 | 107,720 | approximately 354.15 s |

The direct topology-based comparison uses DC, KC, PR, ILC, MSIS, WDKS, and EQLG. TO is substantially slower because its three-hop neighborhoods cover a much larger fraction of the network. EQLG runtime therefore depends on local neighborhood expansion as well as node count.

The verified KONECT sources and preprocessing details are listed in [`data/README.md`](data/README.md). A standalone public loader for these three networks is not currently included.

### Learning-based computational profile

Learning-based costs are reported separately on all eight main networks:

- **1D-CGS:** inference time only;
- **SGNN:** network-specific training, neural inference, structural-entropy reranking, and total computational time.

The detailed summary and raw timings are provided in [`supplementary/Supplementary_Table_S1_Learning_Runtime_8Networks.xlsx`](supplementary/Supplementary_Table_S1_Learning_Runtime_8Networks.xlsx). See [`supplementary/README.md`](supplementary/README.md) for the timing definitions.

The workbook contains measured data, but the external 1D-CGS and SGNN implementations are not redistributed.

## Experimental parameters

The settings are summarized in [`config/parameters.yaml`](config/parameters.yaml).

| Analysis | Setting |
|---|---|
| Default EQLG radius | `3` |
| Radius sensitivity | `1, 2, 3, 4` |
| Jaccard top fraction | `20%` |
| SIR simulations per source | `1000` |
| Deterministic runtime repetitions | `5` after `1` warm-up |
| BA model | `m = 3`, sizes `1k` to `50k` |
| Main effectiveness networks | `8` |
| Main effectiveness methods | `12` including EQLG |

## Baseline methods and references

The manuscript compares EQLG with eleven baseline methods. Third-party source code is not redistributed. 1D-CGS and SGNN are explicitly treated as external baseline methods.

| Method | Category | Original/reference formulation used in the manuscript |
|---|---|---|
| DC | Classical | Freeman LC. *Centrality in social networks conceptual clarification*. Social Networks, 1978, 1:215-239. [DOI](https://doi.org/10.1016/0378-8733(78)90021-7) |
| KC | Classical | Kitsak M et al. *Identification of influential spreaders in complex networks*. Nature Physics, 2010, 6:888-893. [DOI](https://doi.org/10.1038/nphys1746) |
| BC | Classical | Barthelemy M. *Betweenness centrality in large complex networks*. European Physical Journal B, 2004, 38:163-168. [DOI](https://doi.org/10.1140/epjb/e2004-00111-4) |
| PR | Classical | Brin S, Page L. *The anatomy of a large-scale hypertextual Web search engine*. Computer Networks and ISDN Systems, 1998, 30:107-117. [DOI](https://doi.org/10.1016/S0169-7552(98)00110-X) |
| KSGC | Topology-based | Yang X, Xiao F. *An improved gravity model to identify influential nodes in complex networks based on k-shell method*. Knowledge-Based Systems, 2021, 227:107198. [DOI](https://doi.org/10.1016/j.knosys.2021.107198) |
| MSIS | Topology-based | Ma J, Hu J. *Identifying critical nodes in complex networks via a Multi-Scale Influence Spread method*. Chaos, Solitons & Fractals, 2025, 198:116573. [DOI](https://doi.org/10.1016/j.chaos.2025.116573) |
| WDKS | Topology-based | Zhu X, Hao R. *Finding influential nodes in complex networks by integrating nodal intrinsic and extrinsic centrality*. Chaos, Solitons & Fractals, 2025, 194:116278. [DOI](https://doi.org/10.1016/j.chaos.2025.116278) |
| ILC | Topology-based | Zhu X, Hao R. *Identifying influential nodes in social networks via improved Laplacian centrality*. Chaos, Solitons & Fractals, 2024, 189:115675. [DOI](https://doi.org/10.1016/j.chaos.2024.115675) |
| GLC | Topology-based | Ruan Y et al. *GLC: A dual-perspective approach for identifying influential nodes in complex networks*. Expert Systems with Applications, 2025, 268:126292. [DOI](https://doi.org/10.1016/j.eswa.2024.126292) |
| 1D-CGS | External learning-based baseline | Ramadhan MA, Mohammed AO. *A lightweight deep learning-based model for ranking influential nodes in complex networks*. Chaos, Solitons & Fractals, 2026, 209:118562. [DOI](https://doi.org/10.1016/j.chaos.2026.118562) |
| SGNN | External learning-based baseline | Huang X, Huang K. *Structure ranking Graph Neural Networks for critical node identification in complex networks*. Physica A: Statistical Mechanics and its Applications, 2026, 701:132008. [DOI](https://doi.org/10.1016/j.physa.2026.132008) |

## Reproducibility scope

The public repository contains:

- the EQLG implementation;
- SIR-based evaluation functions;
- Kendall, Jaccard, and monotonicity functions;
- experimental parameter settings;
- dataset preprocessing information;
- main effectiveness-evaluation logic;
- truncation-radius sensitivity logic;
- deterministic runtime wrappers;
- EQLG BA scalability and memory logic;
- verified learning-based computational-profile data;
- baseline references.

Users supply raw datasets, dataset paths, SIR reference results where applicable, and third-party baseline implementations. Standalone public scripts for the additional PG/CM/TO comparison and the external learning-based models are not included.

## Citation

If you use this code, please cite:

Siqi Zhu, Jie Zhan, Yan Hui, and Shibo Wei. *Identifying influential nodes in complex networks with entropy-normalized quasi-Laplacian gravity centrality*.

## License

MIT License.

