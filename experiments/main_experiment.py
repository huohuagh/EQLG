"""
Main experimental evaluation for EQLG.

This module provides the core experimental procedure used to
compare EQLG rankings with SIR-based spreading influence.

The module intentionally does not include dataset paths or a
command-line entry point. Network loading and environment-specific
configuration are left to the user.
"""

import random

from src.eqlg import eqlg_centrality
from src.sir import epidemic_threshold, sir_all_nodes
from src.metrics import (
    kendall_correlation,
    jaccard_similarity,
    monotonicity,
)


DEFAULT_BETA_MULTIPLIERS = (
    0.8,
    0.9,
    1.0,
    1.1,
    1.2,
)


def evaluate_method(
    method_scores,
    sir_scores,
    top_fraction=0.20
):
    """
    Evaluate one centrality method against SIR spreading influence.

    Parameters
    ----------
    method_scores : dict
        Mapping from node to the score produced by a
        centrality method.

    sir_scores : dict
        Mapping from node to the mean SIR outbreak size.

    top_fraction : float, default=0.20
        Fraction of top-ranked nodes used for Jaccard similarity.

    Returns
    -------
    dict
        Kendall correlation and Jaccard similarity.
    """
    return {
        "kendall": kendall_correlation(
            method_scores,
            sir_scores
        ),
        "jaccard": jaccard_similarity(
            method_scores,
            sir_scores,
            fraction=top_fraction
        ),
    }


def run_main_experiment(
    G,
    baseline_scores=None,
    radius=3,
    recovery_probability=1.0,
    beta_multipliers=DEFAULT_BETA_MULTIPLIERS,
    simulations_per_source=1000,
    top_fraction=0.20,
    seed=None
):
    """
    Perform the main EQLG experimental evaluation.

    The procedure follows the experimental setting described
    in the manuscript:

    1. Compute the EQLG score for every node.
    2. Compute the epidemic threshold beta_c.
    3. Evaluate spreading influence using SIR under five
       infection probabilities around beta_c.
    4. Compare each centrality ranking with the SIR ranking
       using Kendall correlation and Jaccard similarity.
    5. Evaluate the discrimination capability of each
       centrality method using monotonicity.

    Parameters
    ----------
    G : networkx.Graph
        Static, undirected, and unweighted network.

    baseline_scores : dict, optional
        Scores obtained from baseline methods.

        Expected format:

        {
            "DC": {node: score, ...},
            "KC": {node: score, ...},
            "BC": {node: score, ...},
            ...
        }

        Baseline implementations may be supplied separately
        according to the original methods.

    radius : int, default=3
        Truncation radius used by EQLG.

    recovery_probability : float, default=1.0
        SIR recovery probability gamma.

    beta_multipliers : iterable of float
        Multipliers applied to the epidemic threshold beta_c.

    simulations_per_source : int, default=1000
        Number of independent SIR simulations performed for
        each initially infected source node.

    top_fraction : float, default=0.20
        Fraction of top-ranked nodes used in Jaccard similarity.

    seed : int, optional
        Base random seed for the SIR simulations.

    Returns
    -------
    dict
        Experimental results, including EQLG scores,
        SIR-based spreading influence, Kendall correlation,
        Jaccard similarity, and monotonicity.
    """
    if radius < 1:
        raise ValueError(
            "radius must be at least 1."
        )

    if simulations_per_source < 1:
        raise ValueError(
            "simulations_per_source must be at least 1."
        )

    if not 0.0 <= recovery_probability <= 1.0:
        raise ValueError(
            "recovery_probability must lie in [0, 1]."
        )

    if not 0.0 < top_fraction <= 1.0:
        raise ValueError(
            "top_fraction must lie in (0, 1]."
        )

    # --------------------------------------------------------
    # 1. Compute EQLG centrality
    # --------------------------------------------------------
    eqlg_scores = eqlg_centrality(
        G,
        radius=radius
    )

    methods = {
        "EQLG": eqlg_scores
    }

    # Baseline score dictionaries may be provided externally.
    if baseline_scores is not None:

        for method_name, scores in baseline_scores.items():

            if set(scores.keys()) != set(G.nodes()):
                raise ValueError(
                    f"Baseline method '{method_name}' does not "
                    "contain scores for exactly the same nodes "
                    "as the input network."
                )

            methods[method_name] = scores

    # --------------------------------------------------------
    # 2. Epidemic threshold
    # --------------------------------------------------------
    beta_c = epidemic_threshold(G)

    # --------------------------------------------------------
    # 3. Monotonicity
    # --------------------------------------------------------
    monotonicity_results = {
        method_name: monotonicity(scores)
        for method_name, scores in methods.items()
    }

    # --------------------------------------------------------
    # 4. SIR evaluation under different beta settings
    # --------------------------------------------------------
    rng = random.Random(seed)

    diffusion_results = {}

    for multiplier in beta_multipliers:

        beta = multiplier * beta_c

        simulation_seed = rng.randrange(
            0,
            2 ** 32
        )

        sir_scores = sir_all_nodes(
            G=G,
            beta=beta,
            gamma=recovery_probability,
            simulations=simulations_per_source,
            seed=simulation_seed
        )

        method_results = {}

        for method_name, scores in methods.items():

            method_results[method_name] = evaluate_method(
                method_scores=scores,
                sir_scores=sir_scores,
                top_fraction=top_fraction
            )

        diffusion_results[multiplier] = {
            "beta": beta,
            "sir_scores": sir_scores,
            "methods": method_results,
        }

    # --------------------------------------------------------
    # 5. Collect results
    # --------------------------------------------------------
    results = {
        "network": {
            "number_of_nodes": G.number_of_nodes(),
            "number_of_edges": G.number_of_edges(),
        },
        "parameters": {
            "radius": radius,
            "recovery_probability": recovery_probability,
            "beta_multipliers": list(beta_multipliers),
            "simulations_per_source": simulations_per_source,
            "top_fraction": top_fraction,
        },
        "epidemic_threshold": beta_c,
        "centrality_scores": methods,
        "monotonicity": monotonicity_results,
        "diffusion_results": diffusion_results,
    }

    return results
