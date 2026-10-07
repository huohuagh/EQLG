# Dataset Information

This directory provides information about the datasets used in the
experiments of the EQLG study.

Raw network datasets are not redistributed in this repository.
Users should obtain the original data from the corresponding
public data sources.

## Datasets

The experiments use eight real-world networks covering social,
communication, infrastructure, biological, transportation, and
scientific collaboration systems.

| Dataset | Nodes | Edges |
|---|---:|---:|
| Dolphins | 62 | 159 |
| Email | 1133 | 5451 |
| NS | 379 | 914 |
| Power | 4941 | 6594 |
| Router | 5022 | 6258 |
| Euroroad | 1174 | 1417 |
| Wikivote | 889 | 2914 |
| Yeast | 1870 | 2277 |

The node and edge counts above correspond to the network instances
used in the manuscript.

## Data sources

The network data used in this study are publicly available from
network-data repositories, including:

- Network Repository: https://networkrepository.com/
- Stanford Network Analysis Project (SNAP):
  https://snap.stanford.edu/data/

Please refer to the original dataset providers and the references
listed in the manuscript for detailed descriptions of individual
networks.

## Network representation

For consistency across the evaluated centrality methods, all
networks in the experiments are represented as:

- static networks;
- undirected networks;
- unweighted networks.

The EQLG implementation in this repository is therefore designed
for static, undirected, and unweighted graphs.

## Preprocessing

The preprocessing objective is to convert the original network data
into a consistent undirected and unweighted graph representation
before centrality calculation.

Because raw datasets may be distributed in different file formats,
users should first parse the original network file and construct a
NetworkX `Graph` object.

A generic preprocessing procedure can be expressed as:

```python
import networkx as nx

G = nx.Graph()

# Add the edges parsed from the original dataset.
# Edge weights or directions are not used in the experiments.
