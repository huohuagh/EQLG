"""Section 4.4.1: sensitivity of EQLG to the truncation radius.

This module exposes the experimental logic used to evaluate r = 1, 2, 3, 4
with Kendall correlation, top-20% Jaccard similarity, and monotonicity.

No dataset paths or command-line interface are imposed here. Users provide
NetworkX graphs and the corresponding SIR reference-score dictionaries.
"""

from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping

import networkx as nx

from src.eqlg import eqlg_centrality
from src.metrics import jaccard_similarity, kendall_correlation, monotonicity


RADII = (1, 2, 3, 4)
TOP_FRACTION = 0.20


def _check_same_nodes(
    G: nx.Graph,
    scores: Mapping[Any, float],
    label: str,
) -> None:
    graph_nodes = set(G.nodes())
    score_nodes = set(scores.keys())

    if graph_nodes != score_nodes:
        raise ValueError(
            f"{label} must contain exactly the graph nodes "
            f"(missing={len(graph_nodes - score_nodes)}, "
            f"extra={len(score_nodes - graph_nodes)})."
        )


def evaluate_radius_sensitivity(
    graphs: Mapping[str, nx.Graph],
    sir_scores_by_network: Mapping[
        str, Mapping[str, Mapping[Any, float]]
    ],
    radii: Iterable[int] = RADII,
    top_fraction: float = TOP_FRACTION,
) -> Dict[str, List[Dict[str, Any]]]:
    """Evaluate the truncation-radius sensitivity experiment.

    Parameters
    ----------
    graphs
        Mapping from network name to an undirected, unweighted NetworkX graph.

    sir_scores_by_network
        Mapping from network name to five SIR reference-score mappings.

    radii
        Truncation radii to evaluate. The manuscript uses (1, 2, 3, 4).

    top_fraction
        Fraction of top-ranked nodes used for Jaccard similarity.
        The manuscript uses 0.20.

    Returns
    -------
    dict
        Three tables represented as lists of dictionaries:
        detail, per_network, and overall. The overall table corresponds
        directly to Table 6 in the revised manuscript.
    """
    if set(graphs.keys()) != set(sir_scores_by_network.keys()):
        raise ValueError(
            "graphs and sir_scores_by_network must contain the same networks."
        )

    radii = tuple(radii)

    if not radii or any(radius < 1 for radius in radii):
        raise ValueError("All radii must be positive integers.")

    if not 0.0 < top_fraction <= 1.0:
        raise ValueError("top_fraction must lie in (0, 1].")

    detail: List[Dict[str, Any]] = []
    per_network: List[Dict[str, Any]] = []

    for network_name, G in graphs.items():
        sir_settings = sir_scores_by_network[network_name]

        if len(sir_settings) != 5:
            raise ValueError(
                f"{network_name}: five SIR reference settings are required."
            )

        for setting_name, sir_scores in sir_settings.items():
            _check_same_nodes(
                G,
                sir_scores,
                f"SIR scores for {network_name}/{setting_name}",
            )

        for radius in radii:
            eqlg_scores = eqlg_centrality(G, radius=radius)
            mono = monotonicity(eqlg_scores)

            kendall_values = []
            jaccard_values = []

            for setting_name, sir_scores in sir_settings.items():
                kendall = kendall_correlation(eqlg_scores, sir_scores)
                jaccard = jaccard_similarity(
                    eqlg_scores,
                    sir_scores,
                    fraction=top_fraction,
                )

                kendall_values.append(kendall)
                jaccard_values.append(jaccard)

                detail.append(
                    {
                        "network": network_name,
                        "nodes": G.number_of_nodes(),
                        "edges": G.number_of_edges(),
                        "radius": radius,
                        "sir_setting": setting_name,
                        "kendall": kendall,
                        "jaccard_top_20_percent": jaccard,
                        "monotonicity": mono,
                    }
                )

            per_network.append(
                {
                    "network": network_name,
                    "radius": radius,
                    "mean_kendall": mean(kendall_values),
                    "mean_jaccard": mean(jaccard_values),
                    "monotonicity": mono,
                }
            )

    overall = []

    for radius in radii:
        radius_detail = [
            row for row in detail
            if row["radius"] == radius
        ]
        radius_networks = [
            row for row in per_network
            if row["radius"] == radius
        ]

        overall.append(
            {
                "Radius": radius,
                "Mean Kendall": mean(
                    float(row["kendall"])
                    for row in radius_detail
                ),
                "Mean Jaccard": mean(
                    float(row["jaccard_top_20_percent"])
                    for row in radius_detail
                ),
                "Mean Monotonicity": mean(
                    float(row["monotonicity"])
                    for row in radius_networks
                ),
            }
        )

    return {
        "detail": detail,
        "per_network": per_network,
        "overall": overall,
    }
