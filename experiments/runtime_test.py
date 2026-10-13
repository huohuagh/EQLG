"""Section 4.4.2: wall-clock runtime and radius-dependent cost.

Only the method call is timed.  Graph loading, graph statistics, and radius
coverage estimation are deliberately kept outside the timed region.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import networkx as nx

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.eqlg import eqlg_centrality


RADII = (1, 2, 3, 4)
DEFAULT_REPEATS = 3
DEFAULT_WARMUP_RUNS = 1
DEFAULT_SEED = 42


def measure_runtime(
    method: Callable[[nx.Graph], Any],
    G: nx.Graph,
    *,
    repeats: int = DEFAULT_REPEATS,
    warmup_runs: int = DEFAULT_WARMUP_RUNS,
) -> dict[str, Any]:
    """Measure repeated wall-clock execution time in seconds."""
    if repeats < 1 or warmup_runs < 0:
        raise ValueError("repeats must be >= 1 and warmup_runs must be >= 0.")

    for _ in range(warmup_runs):
        method(G)

    elapsed: list[float] = []
    for _ in range(repeats):
        gc.collect()
        start = time.perf_counter()
        method(G)
        elapsed.append(time.perf_counter() - start)

    return {
        "times_seconds": elapsed,
        "mean_seconds": statistics.mean(elapsed),
        "std_seconds": statistics.stdev(elapsed) if repeats > 1 else 0.0,
        "minimum_seconds": min(elapsed),
        "maximum_seconds": max(elapsed),
    }


def compare_methods_runtime(
    G: nx.Graph,
    method_functions: Mapping[str, Callable[[nx.Graph], Any]],
    *,
    repeats: int = DEFAULT_REPEATS,
    warmup_runs: int = DEFAULT_WARMUP_RUNS,
) -> list[dict[str, Any]]:
    """Time EQLG and caller-supplied baseline implementations uniformly."""
    if not method_functions:
        raise ValueError("method_functions must not be empty.")

    records: list[dict[str, Any]] = []
    for method_name, method in method_functions.items():
        if not callable(method):
            raise TypeError(f"Method {method_name!r} is not callable.")
        timing = measure_runtime(
            method, G, repeats=repeats, warmup_runs=warmup_runs
        )
        records.append(
            {
                "method": method_name,
                "nodes": G.number_of_nodes(),
                "edges": G.number_of_edges(),
                "repeats": repeats,
                "warmup_runs": warmup_runs,
                "mean_seconds": timing["mean_seconds"],
                "std_seconds": timing["std_seconds"],
                "minimum_seconds": timing["minimum_seconds"],
                "maximum_seconds": timing["maximum_seconds"],
            }
        )
    return records


def estimate_radius_coverage(
    G: nx.Graph,
    radius: int,
    *,
    sample_size: int | None = None,
    seed: int = DEFAULT_SEED,
) -> dict[str, float | int]:
    """Estimate nodes reached within ``radius`` (the source is excluded)."""
    nodes = list(G.nodes())
    if sample_size is None or sample_size <= 0 or sample_size >= len(nodes):
        sources = nodes
    else:
        sources = random.Random(seed).sample(nodes, sample_size)

    counts = [
        len(nx.single_source_shortest_path_length(G, source, cutoff=radius)) - 1
        for source in sources
    ]
    mean_nodes = statistics.mean(counts) if counts else 0.0
    denominator = max(1, G.number_of_nodes() - 1)
    return {
        "coverage_sample_size": len(sources),
        "mean_nodes_within_radius": mean_nodes,
        "mean_coverage_fraction": mean_nodes / denominator,
    }


def graph_statistics(G: nx.Graph, *, compute_diameter: bool = True) -> dict[str, Any]:
    n = G.number_of_nodes()
    m = G.number_of_edges()
    if compute_diameter and n > 0:
        diameter: int | None = (
            nx.diameter(G) if nx.is_connected(G) else None
        )
    else:
        diameter = None
    return {
        "nodes": n,
        "edges": m,
        "density": nx.density(G),
        "average_degree": (2.0 * m / n) if n else 0.0,
        "diameter": diameter,
    }


def radius_runtime_analysis(
    graphs: Mapping[str, nx.Graph],
    *,
    radii: Iterable[int] = RADII,
    repeats: int = DEFAULT_REPEATS,
    warmup_runs: int = DEFAULT_WARMUP_RUNS,
    coverage_sample_size: int | None = None,
    seed: int = DEFAULT_SEED,
    compute_diameter: bool = True,
) -> list[dict[str, Any]]:
    """Measure how runtime and search coverage change with radius."""
    records: list[dict[str, Any]] = []
    for network_name, G in graphs.items():
        structure = graph_statistics(G, compute_diameter=compute_diameter)
        for radius in radii:
            if radius < 1:
                raise ValueError("All radii must be at least 1.")
            timing = measure_runtime(
                lambda graph, r=radius: eqlg_centrality(graph, radius=r),
                G,
                repeats=repeats,
                warmup_runs=warmup_runs,
            )
            coverage = estimate_radius_coverage(
                G,
                radius,
                sample_size=coverage_sample_size,
                seed=seed,
            )
            records.append(
                {
                    "network": network_name,
                    **structure,
                    "radius": radius,
                    **coverage,
                    "repeats": repeats,
                    "warmup_runs": warmup_runs,
                    "mean_seconds": timing["mean_seconds"],
                    "std_seconds": timing["std_seconds"],
                    "minimum_seconds": timing["minimum_seconds"],
                    "maximum_seconds": timing["maximum_seconds"],
                }
            )
    return records


def write_csv(records: list[Mapping[str, Any]], output_path: Path) -> None:
    if not records:
        raise ValueError("No records to write.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def load_graph_manifest(manifest_path: Path) -> dict[str, nx.Graph]:
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    casts: dict[str, Callable[[str], Any]] = {"str": str, "int": int, "float": float}
    graphs: dict[str, nx.Graph] = {}
    for network_name, spec in manifest.items():
        node_type = spec.get("node_type", "str")
        if node_type not in casts:
            raise ValueError("node_type must be one of: str, int, float.")
        graph_path = (manifest_path.parent / spec["edge_list"]).resolve()
        G = nx.read_edgelist(
            graph_path,
            comments=spec.get("comments", "#"),
            delimiter=spec.get("delimiter"),
            nodetype=casts[node_type],
            data=False,
            create_using=nx.Graph,
        )
        G.remove_edges_from(nx.selfloop_edges(G))
        graphs[network_name] = G
    return graphs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--radii", type=int, nargs="+", default=list(RADII))
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--warmup-runs", type=int, default=DEFAULT_WARMUP_RUNS)
    parser.add_argument("--coverage-sample-size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--skip-diameter", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=Path("results/radius_runtime.csv")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    graphs = load_graph_manifest(args.manifest)
    records = radius_runtime_analysis(
        graphs,
        radii=args.radii,
        repeats=args.repeats,
        warmup_runs=args.warmup_runs,
        coverage_sample_size=args.coverage_sample_size,
        seed=args.seed,
        compute_diameter=not args.skip_diameter,
    )
    write_csv(records, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

