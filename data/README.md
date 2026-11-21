# Dataset Information

Raw network datasets are not redistributed in this repository. Users should obtain the original data from the cited public sources and apply the preprocessing described below.

## Main effectiveness datasets

The SIR-based effectiveness evaluation uses eight static, undirected, and unweighted real-world networks.

| Dataset | Nodes | Edges | Experimental graph |
|---|---:|---:|---|
| Dolphins | 62 | 159 | simple graph after self-loop removal |
| Email | 1,133 | 5,451 | simple graph after self-loop removal |
| NS | 379 | 914 | simple graph after self-loop removal |
| Power | 4,941 | 6,594 | simple graph after self-loop removal |
| Router | 5,022 | 6,258 | simple graph after self-loop removal |
| Euroroad | 1,039 | 1,305 | largest connected component |
| Wikivote | 889 | 2,914 | simple graph after self-loop removal |
| Yeast | 1,458 | 1,948 | largest connected component |

Euroroad and Yeast are disconnected in their raw forms, so the experiments use their largest connected components. The other six main networks use the corresponding simple undirected graph after self-loop removal.

Public sources used for the main datasets include:

- [Network Repository](https://networkrepository.com/)
- [Stanford Network Analysis Project (SNAP)](https://snap.stanford.edu/data/)

Refer to the manuscript and the original dataset providers for the dataset-specific citations.

## Additional scalability-only datasets

PG, CM, and TO are used only for computational scalability. They are not included in the eight-network SIR effectiveness evaluation.

The experiment downloads the KONECT TSV archives identified below, reads the first two columns as endpoints, constructs a static undirected `networkx.Graph`, removes self-loops, merges repeated edges, and retains the largest connected component.

| Code | Dataset | KONECT internal name and source | Experimental nodes | Experimental edges |
|---|---|---|---:|---:|
| PG | Pretty Good Privacy | [`arenas-pgp`](https://konect.cc/networks/arenas-pgp/) | 10,680 | 24,316 |
| CM | Condensed matter (2003) | [`dimacs10-cond-mat-2003`](https://konect.cc/networks/dimacs10-cond-mat-2003/) | 27,519 | 116,181 |
| TO | Internet topology | [`topology`](https://konect.cc/networks/topology/) | 34,761 | 107,720 |

Important source distinctions:

- KONECT already distributes PG as its giant connected component.
- The CM source contains 30,460 nodes and 120,029 edges before LCC extraction; the experimental LCC contains 27,519 nodes and 116,181 edges.
- The TO source contains 171,403 temporal edge records. Converting the downloaded edge list to a static simple graph merges repeated temporal records and produces the 107,720 edges used in the scalability experiment.

The additional-network selection follows:

Zhang M, Wang X, Jin L, Song M, Li Z. *A new approach for evaluating node importance in complex networks via deep learning methods*. Neurocomputing, 2022, 497:13-27. [DOI](https://doi.org/10.1016/j.neucom.2022.05.010)

KONECT reference:

Kunegis J. *KONECT: the Koblenz Network Collection*. Proceedings of the 22nd International Conference on World Wide Web Companion, 2013, 1343-1350. [KONECT](https://konect.cc/)

## Generic preprocessing pattern

```python
import networkx as nx

G = nx.Graph()
G.add_edges_from(edge_list)
G.remove_edges_from(nx.selfloop_edges(G))

if use_largest_connected_component and G.number_of_nodes() > 0:
    largest_component = max(nx.connected_components(G), key=len)
    G = G.subgraph(largest_component).copy()
```

Dataset loaders must also ignore weights, directions, timestamps, and duplicate edge records when the corresponding source contains them.
