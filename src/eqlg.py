"""
EQLG: Entropy-normalized Quasi-Laplacian Gravity Centrality

This module implements the EQLG centrality proposed in:

"Identifying influential nodes in complex networks with
entropy-normalized quasi-Laplacian gravity centrality"

The current implementation is designed for static,
undirected, and unweighted networks.
"""

import math
from collections import deque

import networkx as nx


def quasi_laplacian_energy(G):
    """
    Compute the quasi-Laplacian structural energy QL(v)
    for every node v.

    QL(v) = k(v)^2 + k(v) + 2 * sum_{u in Gamma(v)} k(u)

    Parameters
    ----------
    G : networkx.Graph
        Undirected and unweighted network.

    Returns
    -------
    dict
        Dictionary mapping each node to its QL value.
    """
    degree = dict(G.degree())
    ql = {}

    for v in G.nodes():
        neighbor_degree_sum = sum(degree[u] for u in G.neighbors(v))

        ql[v] = (
            degree[v] ** 2
            + degree[v]
            + 2.0 * neighbor_degree_sum
        )

    return ql


def neighborhood_evenness(G):
    """
    Compute normalized neighborhood degree-share evenness H(v).

    For neighbor u of node v,

        p_vu = k(u) / sum_{x in Gamma(v)} k(x)

    and, when k(v) >= 2,

        H(v) =
        - sum p_vu * ln(p_vu) / ln(k(v))

    For nodes with degree smaller than 2, H(v) is set to 0.

    Parameters
    ----------
    G : networkx.Graph
        Undirected and unweighted network.

    Returns
    -------
    dict
        Dictionary mapping each node to its normalized
        neighborhood degree-share evenness.
    """
    degree = dict(G.degree())
    entropy = {}

    for v in G.nodes():

        k_v = degree[v]

        if k_v < 2:
            entropy[v] = 0.0
            continue

        neighbors = list(G.neighbors(v))

        total_neighbor_degree = sum(
            degree[u] for u in neighbors
        )

        if total_neighbor_degree == 0:
            entropy[v] = 0.0
            continue

        h = 0.0

        for u in neighbors:
            p = degree[u] / total_neighbor_degree

            if p > 0:
                h -= p * math.log(p)

        entropy[v] = h / math.log(k_v)

    return entropy


def eqlg_mass(G):
    """
    Compute the entropy-modulated quasi-Laplacian mass.

        M(v) = QL(v) * [1 + H(v)]

    Parameters
    ----------
    G : networkx.Graph

    Returns
    -------
    dict
        Node masses.
    """
    ql = quasi_laplacian_energy(G)
    entropy = neighborhood_evenness(G)

    mass = {}

    for v in G.nodes():
        mass[v] = ql[v] * (1.0 + entropy[v])

    return mass


def truncated_bfs_distances(G, source, radius):
    """
    Compute shortest-path distances from a source node
    using breadth-first search truncated at a given radius.

    Parameters
    ----------
    G : networkx.Graph

    source : hashable
        Source node.

    radius : int
        Truncation radius.

    Returns
    -------
    dict
        Nodes within the radius and their shortest-path
        distances from the source.
    """
    distances = {source: 0}

    queue = deque([source])

    while queue:

        current = queue.popleft()

        current_distance = distances[current]

        if current_distance >= radius:
            continue

        for neighbor in G.neighbors(current):

            if neighbor not in distances:

                new_distance = current_distance + 1

                if new_distance <= radius:
                    distances[neighbor] = new_distance
                    queue.append(neighbor)

    return distances


def eqlg_centrality(G, radius=3):
    """
    Compute EQLG centrality for every node.

    EQLG(v) =
        sum_{u in N_r(v)}
        M(v) * M(u) / d(v,u)^2

    where N_r(v) contains nodes within shortest-path
    distance r from v, excluding v itself.

    Parameters
    ----------
    G : networkx.Graph
        Static, undirected, unweighted network.

    radius : int, default=3
        Truncation radius.

    Returns
    -------
    dict
        EQLG scores for all nodes.
    """

    if G.is_directed():
        raise ValueError(
            "The current EQLG implementation requires "
            "an undirected graph."
        )

    if radius < 1:
        raise ValueError("radius must be at least 1.")

    mass = eqlg_mass(G)

    scores = {}

    for v in G.nodes():

        distances = truncated_bfs_distances(
            G,
            source=v,
            radius=radius
        )

        score = 0.0

        for u, distance in distances.items():

            if u == v:
                continue

            score += (
                mass[v]
                * mass[u]
                / (distance ** 2)
            )

        scores[v] = score

    return scores


def rank_nodes(scores):
    """
    Rank nodes in descending order according to
    their EQLG scores.

    Parameters
    ----------
    scores : dict
        Mapping from node to centrality score.

    Returns
    -------
    list
        List of (node, score) tuples sorted in descending order.
    """
    return sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True
    )


if __name__ == "__main__":

    # Small demonstration network
    G = nx.karate_club_graph()

    scores = eqlg_centrality(G, radius=3)

    ranking = rank_nodes(scores)

    print("Top 10 nodes ranked by EQLG:")

    for node, score in ranking[:10]:
        print(f"Node {node}: {score:.6f}")
