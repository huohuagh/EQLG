"""Section 4.4.1: sensitivity of EQLG to the truncation radius.

The experiment evaluates r = 1, 2, 3, and 4 with Kendall correlation,
top-20% Jaccard similarity, and monotonicity.  SIR reference scores are read
from JSON files so the costly diffusion simulations can be reused.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Iterable, Mapping

import networkx as nx

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.eqlg import eqlg_centrality
from src.metrics import jaccard_similarity, kendall_correlation, monotonicity


RADII = (1, 2, 3, 4)
TOP_FRACTION = 0.20


def _validate_scores(G: nx.Graph, scores: Mapping[Any, float], label: str) -> None:
    if set(scores) != set(G.nodes()):
        missing = set(G.nodes()) - set(scores)
        extra = set(scores) - set(G.nodes())
        raise ValueError(
            f"{label} must contain exactly the graph nodes "
            f"(missing={len(missing)}, extra={len(extra)})."
        )


def evaluate_single_network(
    G: nx.Graph,
    sir_reference_scores: Mapping[str, Mapping[Any, float]],
    *,
    network_name: str = "network",
    radii: Iterable[int] = RADII,
    top_fraction: float = TOP_FRACTION,
) -> list[dict[str, Any]]:
    """Return one detailed record per radius and SIR setting."""
    if not 0.0 < top_fraction <= 1.0:
        raise ValueError("top_fraction must lie in (0, 1].")
    if not sir_reference_scores:
        raise ValueError("sir_reference_scores must not be empty.")

    for setting, scores in sir_reference_scores.items():
        _validate_scores(G, scores, f"SIR scores for {setting}")

    records: list[dict[str, Any]] = []
    for radius in radii:
        if radius < 1:
            raise ValueError("All radii must be at least 1.")

        eqlg_scores = eqlg_centrality(G, radius=radius)
        score_monotonicity = monotonicity(eqlg_scores)

        for setting, sir_scores in sorted(
            sir_reference_scores.items(), key=lambda item: str(item[0])
        ):
            records.append(
                {
                    "network": network_name,
                    "nodes": G.number_of_nodes(),
                    "edges": G.number_of_edges(),
                    "radius": radius,
                    "sir_setting": setting,
                    "kendall": kendall_correlation(eqlg_scores, sir_scores),
                    "jaccard_top_20_percent": jaccard_similarity(
                        eqlg_scores, sir_scores, fraction=top_fraction
                    ),
                    "monotonicity": score_monotonicity,
                }
            )

    return records


def evaluate_networks(
    graphs: Mapping[str, nx.Graph],
    sir_scores_by_network: Mapping[
        str, Mapping[str, Mapping[Any, float]]
    ],
    *,
    radii: Iterable[int] = RADII,
    top_fraction: float = TOP_FRACTION,
) -> list[dict[str, Any]]:
    """Evaluate all networks using a shared experimental protocol."""
    if set(graphs) != set(sir_scores_by_network):
        raise ValueError("graphs and sir_scores_by_network must have the same keys.")

    records: list[dict[str, Any]] = []
    for network_name, G in graphs.items():
        records.extend(
            evaluate_single_network(
                G,
                sir_scores_by_network[network_name],
                network_name=network_name,
                radii=radii,
                top_fraction=top_fraction,
            )
        )
    return records


def summarize_radius_results(
    records: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Average Kendall/Jaccard over the five SIR settings."""
    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(str(record["network"]), int(record["radius"]))].append(record)

    summary: list[dict[str, Any]] = []
    for (network_name, radius), values in sorted(grouped.items()):
        summary.append(
            {
                "network": network_name,
                "radius": radius,
                "mean_kendall": mean(float(row["kendall"]) for row in values),
                "mean_jaccard_top_20_percent": mean(
                    float(row["jaccard_top_20_percent"]) for row in values
                ),
                "monotonicity": float(values[0]["monotonicity"]),
                "sir_settings": len(values),
            }
        )
    return summary


def write_csv(records: list[Mapping[str, Any]], output_path: Path) -> None:
    if not records:
        raise ValueError("No records to write.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def _node_cast(node_type: str) -> Callable[[str], Any]:
    casts: dict[str, Callable[[str], Any]] = {
        "str": str,
        "int": int,
        "float": float,
    }
    if node_type not in casts:
        raise ValueError("node_type must be one of: str, int, float.")
    return casts[node_type]


def load_manifest(
    manifest_path: Path,
) -> tuple[dict[str, nx.Graph], dict[str, dict[str, dict[Any, float]]]]:
    """Load edge lists and cached SIR scores from a JSON manifest."""
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)

    graphs: dict[str, nx.Graph] = {}
    references: dict[str, dict[str, dict[Any, float]]] = {}
    base_dir = manifest_path.parent

    for network_name, spec in manifest.items():
        cast = _node_cast(spec.get("node_type", "str"))
        graph_path = (base_dir / spec["edge_list"]).resolve()
        sir_path = (base_dir / spec["sir_scores"]).resolve()

        G = nx.read_edgelist(
            graph_path,
            comments=spec.get("comments", "#"),
            delimiter=spec.get("delimiter"),
            nodetype=cast,
            data=False,
            create_using=nx.Graph,
        )
        G.remove_edges_from(nx.selfloop_edges(G))

        with sir_path.open(encoding="utf-8") as handle:
            raw_scores = json.load(handle)
        parsed_scores = {
            str(setting): {cast(node): float(score) for node, score in scores.items()}
            for setting, scores in raw_scores.items()
        }

        graphs[network_name] = G
        references[network_name] = parsed_scores

    return graphs, references


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--radii", type=int, nargs="+", default=list(RADII))
    parser.add_argument("--top-fraction", type=float, default=TOP_FRACTION)
    parser.add_argument(
        "--detail-output",
        type=Path,
        default=Path("results/radius_sensitivity_detail.csv"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("results/radius_sensitivity_summary.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    graphs, references = load_manifest(args.manifest)
    detail = evaluate_networks(
        graphs,
        references,
        radii=args.radii,
        top_fraction=args.top_fraction,
    )
    summary = summarize_radius_results(detail)
    write_csv(detail, args.detail_output)
    write_csv(summary, args.summary_output)
    print(f"Wrote {args.detail_output} and {args.summary_output}")


if __name__ == "__main__":
    main()

