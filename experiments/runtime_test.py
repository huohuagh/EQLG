"""
Runtime evaluation utilities for EQLG and baseline methods.

This module provides wall-clock timing functions for:

1. comparing EQLG with baseline centrality methods;
2. analyzing the computational cost of EQLG under
   different truncation radii.

Dataset loading, local paths, and command-line execution
are intentionally left to the user.
"""

import time
import statistics

from src.eqlg import eqlg_centrality


DEFAULT_RADIUS_VALUES = (
    1,
    2,
    3,
    4,
)


def measure_runtime(
    method,
    G,
    repeats=1
):
    """
    Measure the wall-clock execution time of a centrality method.

    Parameters
    ----------
    method : callable
        A function that accepts the network G and returns
        node centrality scores.

        Example:
            lambda graph: eqlg_centrality(graph, radius=3)

    G : networkx.Graph
        Input network.

    repeats : int, default=1
        Number of independent timing repetitions.

        The repeat count can be adjusted according to the
        experimental protocol used by the researcher.

    Returns
    -------
    dict
        Runtime statistics in seconds.
    """
    if repeats < 1:
        raise ValueError(
            "repeats must be at least 1."
        )

    elapsed_times = []

    for _ in range(repeats):

        start = time.perf_counter()

        method(G)

        end = time.perf_counter()

        elapsed_times.append(
            end - start
        )

    mean_time = statistics.mean(
        elapsed_times
    )

    if len(elapsed_times) > 1:
        std_time = statistics.stdev(
            elapsed_times
        )
    else:
        std_time = 0.0

    return {
        "times": elapsed_times,
        "mean": mean_time,
        "std": std_time,
        "minimum": min(elapsed_times),
        "maximum": max(elapsed_times),
    }


def compare_methods_runtime(
    G,
    method_functions,
    repeats=1
):
    """
    Compare wall-clock execution times of EQLG and
    baseline centrality methods.

    Parameters
    ----------
    G : networkx.Graph
        Input network.

    method_functions : dict
        Dictionary mapping method names to callable functions.

        Expected format:

        {
            "DC": dc_function,
            "KC": kc_function,
            "BC": bc_function,
            "PR": pagerank_function,
            "ILC": ilc_function,
            "GLC": glc_function,
            "MSIS": msis_function,
            "KSGC": ksgc_function,
            "WDKS": wdks_function,
            "EQLG": eqlg_function,
        }

        Each callable must accept the graph G as its
        only required argument.

    repeats : int, default=1
        Number of timing repetitions for each method.

    Returns
    -------
    dict
        Runtime statistics for all supplied methods.
    """
    if not method_functions:
        raise ValueError(
            "method_functions must not be empty."
        )

    results = {}

    for method_name, method in method_functions.items():

        if not callable(method):
            raise TypeError(
                f"Method '{method_name}' is not callable."
            )

        results[method_name] = measure_runtime(
            method=method,
            G=G,
            repeats=repeats
        )

    return results


def radius_runtime_analysis(
    G,
    radius_values=DEFAULT_RADIUS_VALUES,
    repeats=1
):
    """
    Evaluate the wall-clock computational cost of EQLG
    for different truncation radii.

    Parameters
    ----------
    G : networkx.Graph
        Static, undirected, and unweighted network.

    radius_values : iterable of int
        Truncation radii to evaluate.
        Default: (1, 2, 3, 4).

    repeats : int, default=1
        Number of timing repetitions for each radius.

    Returns
    -------
    dict
        Runtime statistics for every tested radius.
    """
    results = {}

    for radius in radius_values:

        if radius < 1:
            raise ValueError(
                "All radius values must be at least 1."
            )

        def eqlg_method(
            graph,
            r=radius
        ):
            return eqlg_centrality(
                graph,
                radius=r
            )

        results[radius] = measure_runtime(
            method=eqlg_method,
            G=G,
            repeats=repeats
        )

    return results


def summarize_runtime(results):
    """
    Extract the mean runtime and standard deviation from
    a runtime-result dictionary.

    Parameters
    ----------
    results : dict
        Output from compare_methods_runtime() or
        radius_runtime_analysis().

    Returns
    -------
    dict
        Compact summary containing mean and standard
        deviation in seconds.
    """
    summary = {}

    for name, values in results.items():

        summary[name] = {
            "mean_seconds": values["mean"],
            "std_seconds": values["std"],
        }

    return summary
