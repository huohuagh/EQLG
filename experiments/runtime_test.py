"""Section 4.4.2: wall-clock runtime and radius-dependent cost.

The functions in this module implement the timing logic used in the revision.
They do not prescribe dataset paths or baseline implementations. Users provide
NetworkX graphs and the exact callable used for each centrality method.

Only the centrality computation itself is timed. Data loading, SIR simulation,
result output, and graph-statistic calculation are kept outside the timed
region.
"""

import gc
import statistics
import time
from typing import Any, Callable, Dict, Iterable, List, Mapping

import networkx as nx

from src.eqlg import eqlg_centrality


RADII = (1, 2, 3, 4)
DEFAULT_REPEATS = 5
DEFAULT_WARMUP_RUNS = 1


def measure_runtime(
    method: Callable[[nx.Graph], Any],
    G: nx.Graph,
    repeats: int = DEFAULT_REPEATS,
    warmup_runs: int = DEFAULT_WARMUP_RUNS,
) -> Dict[str, Any]:
    """Measure repeated wall-clock execution time in seconds."""
    if repeats < 1:
        raise ValueError("repeats must be at least 1.")

    if warmup_runs < 0:
        raise ValueError("warmup_runs must be non-negative.")

    for _ in range(warmup_runs):
        method(G)

    values = []

    for _ in range(repeats):
        gc.collect()

        start = time.perf_counter()
        method(G)
        elapsed = time.perf_counter() - start

        values.append(elapsed)

    return {
        "times_seconds": values,
        "mean_seconds": statistics.mean(values),
        "std_seconds": (
            statistics.stdev(values)
            if len(values) > 1
            else 0.0
        ),
        "minimum_seconds": min(values),
        "maximum_seconds": max(values),
    }


def compare_methods_runtime(
    graphs: Mapping[str, nx.Graph],
    method_functions: Mapping[str, Callable[[nx.Graph], Any]],
    repeats: int = DEFAULT_REPEATS,
    warmup_runs: int = DEFAULT_WARMUP_RUNS,
) -> List[Dict[str, Any]]:
    """Measure wall-clock runtime for caller-supplied centrality methods.

    The manuscript compares ten methods (DC, KC, BC, PR, ILC, GLC, MSIS,
    KSGC, WDKS, and EQLG) on Dolphins, Email, Power, and Router.

    Recent baseline implementations are deliberately not redistributed here.
    Pass the exact implementation used for each method through
    method_functions so that all methods share the same timing wrapper.
    """
    if not method_functions:
        raise ValueError("method_functions must not be empty.")

    rows: List[Dict[str, Any]] = []

    for network_name, G in graphs.items():
        for method_name, method in method_functions.items():
            if not callable(method):
                raise TypeError(
                    f"Method {method_name!r} is not callable."
                )

            result = measure_runtime(
                method,
                G,
                repeats=repeats,
                warmup_runs=warmup_runs,
            )

            rows.append(
                {
                    "network": network_name,
                    "nodes": G.number_of_nodes(),
                    "edges": G.number_of_edges(),
                    "method": method_name,
                    "repeats": repeats,
                    "warmup_runs": warmup_runs,
                    "mean_seconds": result["mean_seconds"],
                    "std_seconds": result["std_seconds"],
                    "minimum_seconds": result["minimum_seconds"],
                    "maximum_seconds": result["maximum_seconds"],
                }
            )

    return rows


def radius_runtime_analysis(
    graphs: Mapping[str, nx.Graph],
    radii: Iterable[int] = RADII,
    repeats: int = DEFAULT_REPEATS,
    warmup_runs: int = DEFAULT_WARMUP_RUNS,
) -> List[Dict[str, Any]]:
    """Measure how EQLG wall-clock time changes with truncation radius."""
    radii = tuple(radii)

    if not radii or any(radius < 1 for radius in radii):
        raise ValueError("All radii must be positive integers.")

    rows: List[Dict[str, Any]] = []

    for network_name, G in graphs.items():
        n = G.number_of_nodes()
        m = G.number_of_edges()

        density = nx.density(G)
        diameter = nx.diameter(G) if n > 0 and nx.is_connected(G) else None

        for radius in radii:
            timing = measure_runtime(
                lambda graph, r=radius: eqlg_centrality(
                    graph,
                    radius=r,
                ),
                G,
                repeats=repeats,
                warmup_runs=warmup_runs,
            )

            rows.append(
                {
                    "network": network_name,
                    "nodes": n,
                    "edges": m,
                    "density": density,
                    "diameter": diameter,
                    "radius": radius,
                    "repeats": repeats,
                    "warmup_runs": warmup_runs,
                    "mean_seconds": timing["mean_seconds"],
                    "std_seconds": timing["std_seconds"],
                    "minimum_seconds": timing["minimum_seconds"],
                    "maximum_seconds": timing["maximum_seconds"],
                }
            )

    return rows


def eqlg_at_radius(radius: int = 3) -> Callable[[nx.Graph], Dict[Any, float]]:
    """Return an EQLG callable convenient for compare_methods_runtime."""
    if radius < 1:
        raise ValueError("radius must be at least 1.")

    def _method(G: nx.Graph) -> Dict[Any, float]:
        return eqlg_centrality(G, radius=radius)

    return _method
