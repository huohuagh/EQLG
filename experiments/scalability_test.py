"""Section 4.4.3: EQLG scalability and peak traced memory.

Synthetic Barabasi-Albert networks are generated before measurement.  The
timed/traced region contains only ``eqlg_centrality`` and its returned scores;
network generation and coverage estimation are excluded.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import random
import statistics
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any, Iterable, Mapping

import networkx as nx

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.eqlg import eqlg_centrality


NETWORK_SIZES = (1_000, 5_000, 10_000, 20_000, 50_000)
ATTACHMENT_PARAMETER = 3
RADIUS = 3
REPEATS = 3
WARMUP_RUNS = 1
SEED = 42
COVERAGE_SAMPLE_SIZE = 1_000


def estimate_coverage(
    G: nx.Graph,
    *,
    radius: int,
    sample_size: int,
    seed: int,
) -> dict[str, float | int]:
    nodes = list(G.nodes())
    if sample_size <= 0 or sample_size >= len(nodes):
        sources = nodes
    else:
        sources = random.Random(seed).sample(nodes, sample_size)
    counts = [
        len(nx.single_source_shortest_path_length(G, source, cutoff=radius)) - 1
        for source in sources
    ]
    mean_nodes = statistics.mean(counts) if counts else 0.0
    return {
        "coverage_sample_size": len(sources),
        "mean_nodes_within_radius": mean_nodes,
        "mean_coverage_fraction": mean_nodes / max(1, G.number_of_nodes() - 1),
    }


def measure_eqlg_scalability(
    G: nx.Graph,
    *,
    radius: int = RADIUS,
    repeats: int = REPEATS,
    warmup_runs: int = WARMUP_RUNS,
) -> dict[str, Any]:
    """Measure mean runtime and peak Python-traced memory for one graph."""
    if radius < 1 or repeats < 1 or warmup_runs < 0:
        raise ValueError(
            "radius/repeats must be >= 1 and warmup_runs must be >= 0."
        )

    for _ in range(warmup_runs):
        eqlg_centrality(G, radius=radius)

    elapsed: list[float] = []
    peak_mib: list[float] = []
    for _ in range(repeats):
        gc.collect()
        tracemalloc.start()
        start = time.perf_counter()
        scores = eqlg_centrality(G, radius=radius)
        elapsed.append(time.perf_counter() - start)
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_mib.append(peak_bytes / (1024.0**2))
        del scores

    return {
        "runtime_values_seconds": elapsed,
        "mean_runtime_seconds": statistics.mean(elapsed),
        "runtime_std_seconds": statistics.stdev(elapsed) if repeats > 1 else 0.0,
        "peak_memory_values_mib": peak_mib,
        "mean_peak_memory_mib": statistics.mean(peak_mib),
        "peak_memory_std_mib": statistics.stdev(peak_mib) if repeats > 1 else 0.0,
    }


def scaling_exponent(rows: Iterable[Mapping[str, Any]], field: str) -> float:
    """Return the OLS slope of log(field) against log(number of nodes)."""
    points = [(float(row["nodes"]), float(row[field])) for row in rows]
    if len(points) < 2 or any(x <= 0 or y <= 0 for x, y in points):
        raise ValueError("At least two positive observations are required.")
    log_x = [math.log(x) for x, _ in points]
    log_y = [math.log(y) for _, y in points]
    mean_x = statistics.mean(log_x)
    mean_y = statistics.mean(log_y)
    denominator = sum((value - mean_x) ** 2 for value in log_x)
    return sum(
        (x_value - mean_x) * (y_value - mean_y)
        for x_value, y_value in zip(log_x, log_y)
    ) / denominator


def run_ba_scalability(
    *,
    sizes: Iterable[int] = NETWORK_SIZES,
    attachment_parameter: int = ATTACHMENT_PARAMETER,
    radius: int = RADIUS,
    repeats: int = REPEATS,
    warmup_runs: int = WARMUP_RUNS,
    seed: int = SEED,
    coverage_sample_size: int = COVERAGE_SAMPLE_SIZE,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Run the controlled Barabasi-Albert scaling series."""
    rows: list[dict[str, Any]] = []
    for index, n in enumerate(sizes):
        if n <= attachment_parameter:
            raise ValueError("Every network size must exceed the BA parameter m.")
        graph_seed = seed + index
        G = nx.barabasi_albert_graph(n, attachment_parameter, seed=graph_seed)
        measurement = measure_eqlg_scalability(
            G,
            radius=radius,
            repeats=repeats,
            warmup_runs=warmup_runs,
        )
        coverage = estimate_coverage(
            G,
            radius=radius,
            sample_size=coverage_sample_size,
            seed=graph_seed,
        )
        rows.append(
            {
                "nodes": G.number_of_nodes(),
                "edges": G.number_of_edges(),
                "ba_m": attachment_parameter,
                "radius": radius,
                "graph_seed": graph_seed,
                "repeats": repeats,
                "warmup_runs": warmup_runs,
                **coverage,
                "mean_runtime_seconds": measurement["mean_runtime_seconds"],
                "runtime_std_seconds": measurement["runtime_std_seconds"],
                "mean_peak_memory_mib": measurement["mean_peak_memory_mib"],
                "peak_memory_std_mib": measurement["peak_memory_std_mib"],
            }
        )
        print(
            f"N={n:,}: {measurement['mean_runtime_seconds']:.3f} s, "
            f"{measurement['mean_peak_memory_mib']:.3f} MiB"
        )

    fit = {
        "runtime_scaling_exponent": scaling_exponent(
            rows, "mean_runtime_seconds"
        ),
        "memory_scaling_exponent": scaling_exponent(
            rows, "mean_peak_memory_mib"
        ),
    }
    return rows, fit


def write_outputs(
    rows: list[Mapping[str, Any]],
    fit: Mapping[str, float],
    csv_path: Path,
    fit_path: Path,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fit_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with fit_path.open("w", encoding="utf-8") as handle:
        json.dump(fit, handle, indent=2)
        handle.write("\n")


def plot_scalability(rows: list[Mapping[str, Any]], output_path: Path) -> None:
    """Create the log-log runtime/memory figure used by the section."""
    import matplotlib.pyplot as plt

    nodes = [int(row["nodes"]) for row in rows]
    runtime = [float(row["mean_runtime_seconds"]) for row in rows]
    memory = [float(row["mean_peak_memory_mib"]) for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(8.0, 3.2))
    axes[0].loglog(nodes, runtime, marker="o")
    axes[0].set(xlabel="Nodes", ylabel="Runtime (s)", title="Runtime scaling")
    axes[1].loglog(nodes, memory, marker="o", color="#C44E52")
    axes[1].set(xlabel="Nodes", ylabel="Peak traced memory (MiB)", title="Memory scaling")
    for axis in axes:
        axis.grid(True, which="both", linestyle=":", linewidth=0.6)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=list(NETWORK_SIZES))
    parser.add_argument("--attachment-parameter", type=int, default=ATTACHMENT_PARAMETER)
    parser.add_argument("--radius", type=int, default=RADIUS)
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--warmup-runs", type=int, default=WARMUP_RUNS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--coverage-sample-size", type=int, default=COVERAGE_SAMPLE_SIZE)
    parser.add_argument(
        "--output", type=Path, default=Path("results/scalability.csv")
    )
    parser.add_argument(
        "--fit-output", type=Path, default=Path("results/scalability_fit.json")
    )
    parser.add_argument(
        "--plot-output", type=Path, default=Path("results/scalability.png")
    )
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows, fit = run_ba_scalability(
        sizes=args.sizes,
        attachment_parameter=args.attachment_parameter,
        radius=args.radius,
        repeats=args.repeats,
        warmup_runs=args.warmup_runs,
        seed=args.seed,
        coverage_sample_size=args.coverage_sample_size,
    )
    write_outputs(rows, fit, args.output, args.fit_output)
    if not args.no_plot:
        plot_scalability(rows, args.plot_output)
    print(json.dumps(fit, indent=2))


if __name__ == "__main__":
    main()

