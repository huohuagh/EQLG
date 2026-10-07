"""
Scalability evaluation utilities for EQLG.

This module evaluates how the computational cost of EQLG
changes with network size.

The analysis records:

1. number of nodes;
2. number of edges;
3. network density;
4. average degree;
5. wall-clock execution time;
6. Python-level peak traced memory.

Dataset construction and graph-sampling strategies are
intentionally left to the user.
"""

import statistics
import time
import tracemalloc

from src.eqlg import eqlg_centrality


def graph_statistics(G):
    """
    Compute basic structural statistics of a network.

    Parameters
    ----------
    G : networkx.Graph
        Input network.

    Returns
    -------
    dict
        Basic network statistics.
    """
    n = G.number_of_nodes()
    m = G.number_of_edges()

    if n == 0:
        average_degree = 0.0
        density = 0.0

    else:
        average_degree = (
            2.0 * m / n
        )

        if n > 1:
            density = (
                2.0 * m
                / (n * (n - 1))
            )
        else:
            density = 0.0

    return {
        "nodes": n,
        "edges": m,
        "average_degree": average_degree,
        "density": density,
    }


def measure_eqlg_scalability(
    G,
    radius=3,
    repeats=1
):
    """
    Measure the runtime and peak traced memory of EQLG
    on a single network.

    Parameters
    ----------
    G : networkx.Graph
        Static, undirected, and unweighted network.

    radius : int, default=3
        Truncation radius used by EQLG.

    repeats : int, default=1
        Number of independent measurements.

    Returns
    -------
    dict
        Network statistics, runtime statistics, and
        peak traced memory usage.
    """
    if radius < 1:
        raise ValueError(
            "radius must be at least 1."
        )

    if repeats < 1:
        raise ValueError(
            "repeats must be at least 1."
        )

    structure = graph_statistics(G)

    elapsed_times = []
    peak_memory_values = []

    for _ in range(repeats):

        # Start Python memory tracing.
        tracemalloc.start()

        start = time.perf_counter()

        eqlg_centrality(
            G,
            radius=radius
        )

        end = time.perf_counter()

        _, peak_memory = (
            tracemalloc.get_traced_memory()
        )

        tracemalloc.stop()

        elapsed_times.append(
            end - start
        )

        # Convert bytes to MiB.
        peak_memory_values.append(
            peak_memory
            / (1024 ** 2)
        )

    mean_runtime = statistics.mean(
        elapsed_times
    )

    mean_peak_memory = statistics.mean(
        peak_memory_values
    )

    if repeats > 1:

        runtime_std = statistics.stdev(
            elapsed_times
        )

        memory_std = statistics.stdev(
            peak_memory_values
        )

    else:

        runtime_std = 0.0
        memory_std = 0.0

    return {
        "network": structure,
        "radius": radius,
        "runtime": {
            "times_seconds": elapsed_times,
            "mean_seconds": mean_runtime,
            "std_seconds": runtime_std,
        },
        "memory": {
            "peak_values_mib": peak_memory_values,
            "mean_peak_mib": mean_peak_memory,
            "std_peak_mib": memory_std,
        },
    }


def scalability_analysis(
    graphs,
    radius=3,
    repeats=1
):
    """
    Evaluate EQLG scalability over a sequence of networks.

    Parameters
    ----------
    graphs : dict
        Mapping from a network label to a graph.

        Example structure:

        {
            "network_1": G1,
            "network_2": G2,
            "network_3": G3,
        }

        The networks may be real-world networks,
        synthetic networks, or sampled networks of
        increasing size.

    radius : int, default=3
        Truncation radius used by EQLG.

    repeats : int, default=1
        Number of repeated measurements for each network.

    Returns
    -------
    dict
        Scalability measurements for all supplied networks.
    """
    if not graphs:
        raise ValueError(
            "graphs must not be empty."
        )

    results = {}

    for network_name, G in graphs.items():

        results[network_name] = (
            measure_eqlg_scalability(
                G=G,
                radius=radius,
                repeats=repeats
            )
        )

    return results


def summarize_scalability(results):
    """
    Convert scalability results into a compact summary.

    Parameters
    ----------
    results : dict
        Output returned by scalability_analysis().

    Returns
    -------
    list
        One summary record for each network.
    """
    summary = []

    for network_name, values in results.items():

        network = values["network"]
        runtime = values["runtime"]
        memory = values["memory"]

        summary.append(
            {
                "network": network_name,
                "nodes": network["nodes"],
                "edges": network["edges"],
                "average_degree": (
                    network["average_degree"]
                ),
                "density": network["density"],
                "mean_runtime_seconds": (
                    runtime["mean_seconds"]
                ),
                "runtime_std_seconds": (
                    runtime["std_seconds"]
                ),
                "mean_peak_memory_mib": (
                    memory["mean_peak_mib"]
                ),
                "memory_std_mib": (
                    memory["std_peak_mib"]
                ),
            }
        )

    return summary
