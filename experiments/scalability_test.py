"""Section 4.4.3: EQLG scalability and peak traced memory.

This module reproduces the experimental logic used for the controlled
Barabasi-Albert scaling series in the revised manuscript.

No dataset paths, command-line interface, or automatic output files are
defined. Users call the functions directly and decide how to store or plot the
returned results.

Protocol used in the manuscript:
- BA networks with N = 1,000, 5,000, 10,000, 20,000, 50,000;
- attachment parameter m = 3;
- EQLG truncation radius r = 3;
- base random seed = 2026 and graph seed = base_seed + N;
- three measured EQLG executions per network size;
- mean three-hop coverage estimated from 200 sampled source nodes;
- memory tracing started before graph construction so the reported peak traced
  memory includes the in-memory graph representation and temporary Python-level
  data structures used by EQLG.
"""

import gc
import time
import tracemalloc
from typing import Any, Dict, Iterable, List, Mapping, Tuple

import networkx as nx
import numpy as np

from src.eqlg import eqlg_centrality


NETWORK_SIZES = (1_000, 5_000, 10_000, 20_000, 50_000)
ATTACHMENT_PARAMETER = 3
RADIUS = 3
REPEATS = 3
BASE_SEED = 2026
COVERAGE_SAMPLE_SIZE = 200
WARMUP_NETWORK_SIZE = 500


def warm_up_eqlg(
    n: int = WARMUP_NETWORK_SIZE,
    m: int = ATTACHMENT_PARAMETER,
    radius: int = RADIUS,
    seed: int = BASE_SEED,
) -> None:
    """Run one small untimed EQLG call before the scaling series."""
    G = nx.barabasi_albert_graph(
        n=n,
        m=m,
        seed=seed,
    )

    eqlg_centrality(
        G,
        radius=radius,
    )

    del G
    gc.collect()


def estimate_rhop_coverage(
    G: nx.Graph,
    radius: int = RADIUS,
    sample_size: int = COVERAGE_SAMPLE_SIZE,
    seed: int = BASE_SEED,
) -> float:
    """Estimate the mean fraction of nodes reached within radius hops."""
    n = G.number_of_nodes()

    if n <= 1:
        return 0.0

    nodes = list(G.nodes())

    rng = np.random.default_rng(seed)

    actual_sample_size = min(sample_size, n)

    sampled_nodes = rng.choice(
        nodes,
        size=actual_sample_size,
        replace=False,
    )

    coverage_values = []

    for source in sampled_nodes:
        distances = nx.single_source_shortest_path_length(
            G,
            source,
            cutoff=radius,
        )

        reached = len(distances) - 1

        coverage_values.append(
            reached / (n - 1)
        )

    return float(
        np.mean(coverage_values)
    )


def measure_one_network(
    n: int,
    m: int = ATTACHMENT_PARAMETER,
    radius: int = RADIUS,
    repeats: int = REPEATS,
    base_seed: int = BASE_SEED,
    coverage_sample_size: int = COVERAGE_SAMPLE_SIZE,
) -> Dict[str, Any]:
    """Generate one BA network and measure EQLG runtime and traced memory.

    Network generation is excluded from wall-clock timing but is included in
    the traced-memory baseline. The memory values are divided by 1024**2,
    matching the numerical convention used for Table 8 in the manuscript.
    """
    if n <= m:
        raise ValueError("n must be greater than the BA attachment parameter m.")

    if repeats < 1:
        raise ValueError("repeats must be at least 1.")

    graph_seed = base_seed + n

    gc.collect()

    tracemalloc.start()

    G = nx.barabasi_albert_graph(
        n=n,
        m=m,
        seed=graph_seed,
    )

    number_of_nodes = G.number_of_nodes()
    number_of_edges = G.number_of_edges()

    runtime_values = []
    peak_total_memory_values = []

    for _ in range(repeats):
        gc.collect()

        tracemalloc.reset_peak()

        start = time.perf_counter()

        scores = eqlg_centrality(
            G,
            radius=radius,
        )

        runtime_values.append(
            time.perf_counter() - start
        )

        _, peak_bytes = tracemalloc.get_traced_memory()

        peak_total_memory_values.append(
            peak_bytes / (1024 ** 2)
        )

        del scores
        gc.collect()

    tracemalloc.stop()

    mean_coverage = estimate_rhop_coverage(
        G,
        radius=radius,
        sample_size=coverage_sample_size,
        seed=graph_seed,
    )

    result = {
        "N": number_of_nodes,
        "M": number_of_edges,
        "Average_degree": (
            2.0 * number_of_edges / number_of_nodes
        ),
        "Density": nx.density(G),
        "Runtime_mean_s": float(
            np.mean(runtime_values)
        ),
        "Runtime_std_s": float(
            np.std(runtime_values, ddof=1)
            if repeats > 1
            else 0.0
        ),
        "Peak_total_memory_mean_MB": float(
            np.mean(peak_total_memory_values)
        ),
        "Peak_total_memory_std_MB": float(
            np.std(peak_total_memory_values, ddof=1)
            if repeats > 1
            else 0.0
        ),
        "Mean_rhop_coverage": mean_coverage,
        "Mean_rhop_coverage_percent": 100.0 * mean_coverage,
        "graph_seed": graph_seed,
        "radius": radius,
        "repeats": repeats,
    }

    del G
    gc.collect()

    return result


def scaling_exponent(
    rows: Iterable[Mapping[str, Any]],
    field: str,
) -> float:
    """Return the log-log least-squares scaling exponent."""
    rows = list(rows)

    x = np.asarray(
        [float(row["N"]) for row in rows],
        dtype=float,
    )

    y = np.asarray(
        [float(row[field]) for row in rows],
        dtype=float,
    )

    if len(x) < 2 or np.any(x <= 0) or np.any(y <= 0):
        raise ValueError(
            "At least two positive observations are required."
        )

    return float(
        np.polyfit(
            np.log(x),
            np.log(y),
            1,
        )[0]
    )


def run_scalability_analysis(
    sizes: Iterable[int] = NETWORK_SIZES,
    attachment_parameter: int = ATTACHMENT_PARAMETER,
    radius: int = RADIUS,
    repeats: int = REPEATS,
    base_seed: int = BASE_SEED,
    coverage_sample_size: int = COVERAGE_SAMPLE_SIZE,
    perform_warmup: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """Run the full controlled scalability and memory experiment."""
    if perform_warmup:
        warm_up_eqlg(
            n=WARMUP_NETWORK_SIZE,
            m=attachment_parameter,
            radius=radius,
            seed=base_seed,
        )

    rows = []

    for n in sizes:
        rows.append(
            measure_one_network(
                n=n,
                m=attachment_parameter,
                radius=radius,
                repeats=repeats,
                base_seed=base_seed,
                coverage_sample_size=coverage_sample_size,
            )
        )

    fit = {
        "runtime_scaling_exponent": scaling_exponent(
            rows,
            "Runtime_mean_s",
        ),
        "memory_scaling_exponent": scaling_exponent(
            rows,
            "Peak_total_memory_mean_MB",
        ),
    }

    return rows, fit
