"""
Sensitivity analysis for the truncation radius of EQLG.

This module evaluates how the ranking performance and
discrimination capability of EQLG change when the truncation
radius r varies.

The manuscript uses r = 3 as the default setting. The revised
experiments additionally consider r = 1, 2, 3, and 4.
"""

from src.eqlg import eqlg_centrality
from src.metrics import (
    kendall_correlation,
    jaccard_similarity,
    monotonicity,
)


DEFAULT_RADIUS_VALUES = (
    1,
    2,
    3,
    4,
)


def evaluate_radius(
    G,
    radius,
    sir_scores_by_setting,
    top_fraction=0.20
):
    """
    Evaluate EQLG for one truncation radius.

    Parameters
    ----------
    G : networkx.Graph
        Static, undirected, and unweighted network.

    radius : int
        Truncation radius used by EQLG.

    sir_scores_by_setting : dict
        Precomputed SIR spreading influence under different
        infection-probability settings.

        Expected format:

        {
            0.8: {node: score, ...},
            0.9: {node: score, ...},
            1.0: {node: score, ...},
            1.1: {node: score, ...},
            1.2: {node: score, ...},
        }

        The dictionary keys represent the multipliers applied
        to the epidemic threshold beta_c.

    top_fraction : float, default=0.20
        Fraction of top-ranked nodes used for Jaccard similarity.

    Returns
    -------
    dict
        EQLG scores, monotonicity, Kendall correlations,
        and Jaccard similarities for the specified radius.
    """
    if radius < 1:
        raise ValueError(
            "radius must be at least 1."
        )

    if not 0.0 < top_fraction <= 1.0:
        raise ValueError(
            "top_fraction must lie in (0, 1]."
        )

    # --------------------------------------------------------
    # 1. Compute EQLG scores under the specified radius
    # --------------------------------------------------------
    eqlg_scores = eqlg_centrality(
        G,
        radius=radius
    )

    # --------------------------------------------------------
    # 2. Evaluate discrimination capability
    # --------------------------------------------------------
    mono = monotonicity(
        eqlg_scores
    )

    # --------------------------------------------------------
    # 3. Compare with SIR spreading influence
    # --------------------------------------------------------
    diffusion_results = {}

    for setting, sir_scores in sir_scores_by_setting.items():

        if set(sir_scores.keys()) != set(G.nodes()):
            raise ValueError(
                "SIR scores must contain exactly the same "
                "nodes as the input network."
            )

        diffusion_results[setting] = {
            "kendall": kendall_correlation(
                eqlg_scores,
                sir_scores
            ),
            "jaccard": jaccard_similarity(
                eqlg_scores,
                sir_scores,
                fraction=top_fraction
            ),
        }

    return {
        "radius": radius,
        "eqlg_scores": eqlg_scores,
        "monotonicity": mono,
        "diffusion_results": diffusion_results,
    }


def radius_sensitivity_analysis(
    G,
    sir_scores_by_setting,
    radius_values=DEFAULT_RADIUS_VALUES,
    top_fraction=0.20
):
    """
    Perform a systematic sensitivity analysis with respect
    to the EQLG truncation radius.

    Parameters
    ----------
    G : networkx.Graph
        Static, undirected, and unweighted network.

    sir_scores_by_setting : dict
        Precomputed SIR spreading influence under different
        infection-probability settings.

    radius_values : iterable of int
        Radius values to evaluate.
        Default: (1, 2, 3, 4).

    top_fraction : float, default=0.20
        Fraction of top-ranked nodes used for Jaccard similarity.

    Returns
    -------
    dict
        Sensitivity results for all tested radius values.
    """
    results = {}

    for radius in radius_values:

        results[radius] = evaluate_radius(
            G=G,
            radius=radius,
            sir_scores_by_setting=sir_scores_by_setting,
            top_fraction=top_fraction
        )

    return results


def summarize_radius_results(results):
    """
    Summarize the sensitivity results for each radius.

    For each radius, the function reports:

    - mean Kendall correlation across SIR settings;
    - mean Jaccard similarity across SIR settings;
    - monotonicity.

    Parameters
    ----------
    results : dict
        Output returned by radius_sensitivity_analysis().

    Returns
    -------
    dict
        Summary statistics for each radius.
    """
    summary = {}

    for radius, radius_result in results.items():

        diffusion_results = (
            radius_result["diffusion_results"]
        )

        kendall_values = [
            values["kendall"]
            for values in diffusion_results.values()
        ]

        jaccard_values = [
            values["jaccard"]
            for values in diffusion_results.values()
        ]

        if kendall_values:
            mean_kendall = (
                sum(kendall_values)
                / len(kendall_values)
            )
        else:
            mean_kendall = None

        if jaccard_values:
            mean_jaccard = (
                sum(jaccard_values)
                / len(jaccard_values)
            )
        else:
            mean_jaccard = None

        summary[radius] = {
            "mean_kendall": mean_kendall,
            "mean_jaccard": mean_jaccard,
            "monotonicity": (
                radius_result["monotonicity"]
            ),
        }

    return summary
