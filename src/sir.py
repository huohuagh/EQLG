"""
Continuous-time Markov SIR simulation utilities for node influence.

Transmission and recovery use independent exponential waiting times.
This standalone event-driven implementation retains the public function
interfaces and estimates influence from mean final recovered counts.
"""

from heapq import heappop, heappush
from itertools import count
from math import isfinite
import random

import networkx as nx


def epidemic_threshold(G):
    """
    Compute the degree-based reference scale beta_c.

    beta_c = <k> / (<k^2> - <k>)

    Parameters
    ----------
    G : networkx.Graph
        Undirected and unweighted network.

    Returns
    -------
    float
        Degree-based reference scale beta_c.
    """
    n = G.number_of_nodes()

    if n == 0:
        raise ValueError("The graph contains no nodes.")

    degrees = [degree for _, degree in G.degree()]

    mean_k = sum(degrees) / n
    mean_k2 = sum(k ** 2 for k in degrees) / n

    denominator = mean_k2 - mean_k

    if denominator <= 0:
        raise ValueError(
            "The epidemic threshold is undefined because "
            "<k^2> - <k> <= 0."
        )

    return mean_k / denominator


def sir_once(G, source, beta, gamma=1.0, rng=None):
    """Perform one continuous-time SIR simulation until extinction.

    beta is a finite, nonnegative transmission rate per susceptible-infected
    edge; gamma is a finite, positive recovery rate per infected node.
    Rates are not probabilities and may exceed one. rng may be a seeded
    random.Random instance. Return the final number of recovered nodes.
    """
    if source not in G:
        raise ValueError("The source node is not contained in the graph.")
    if not isfinite(beta) or beta < 0.0:
        raise ValueError("beta must be a finite, nonnegative rate.")
    if not isfinite(gamma) or gamma <= 0.0:
        raise ValueError("gamma must be a finite, positive rate.")
    if rng is None:
        rng = random.Random()

    susceptible = set(G.nodes())
    susceptible.remove(source)
    events = []
    serial = count()

    def schedule(node, infected_at):
        lifetime = rng.expovariate(gamma)
        recovered_at = infected_at + lifetime
        # A unique serial number avoids comparing arbitrary node labels.
        heappush(events, (recovered_at, next(serial), False, node))
        if beta == 0.0:
            return
        for neighbor in G.neighbors(node):
            if neighbor in susceptible:
                delay = rng.expovariate(beta)
                if delay < lifetime:
                    heappush(
                        events,
                        (infected_at + delay, next(serial), True, neighbor)
                    )

    schedule(source, 0.0)
    recovered = 0
    while events:
        event_time, _, infection, node = heappop(events)
        if infection:
            # Another source may already have infected this target.
            if node not in susceptible:
                continue
            susceptible.remove(node)
            schedule(node, event_time)
        else:
            recovered += 1

    return recovered


def sir_influence(
    G,
    source,
    beta,
    gamma=1.0,
    simulations=1000,
    seed=None
):
    """
    Estimate the spreading influence of one source node.

    The influence is the average final outbreak size over
    repeated independent SIR simulations.

    Parameters
    ----------
    G : networkx.Graph

    source : hashable
        Initially infected source node.

    beta : float
        Transmission rate per susceptible-infected edge.

    gamma : float, default=1.0
        Recovery rate per infected node.

    simulations : int, default=1000
        Number of independent simulations.

    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    float
        Mean final outbreak size.
    """
    if simulations < 1:
        raise ValueError("simulations must be at least 1.")

    rng = random.Random(seed)

    total_outbreak_size = 0.0

    for _ in range(simulations):

        total_outbreak_size += sir_once(
            G=G,
            source=source,
            beta=beta,
            gamma=gamma,
            rng=rng
        )

    return total_outbreak_size / simulations


def sir_all_nodes(
    G,
    beta,
    gamma=1.0,
    simulations=1000,
    seed=None
):
    """
    Estimate SIR spreading influence for every node.

    Parameters
    ----------
    G : networkx.Graph

    beta : float
        Transmission rate per susceptible-infected edge.

    gamma : float, default=1.0
        Recovery rate per infected node.

    simulations : int, default=1000
        Independent simulations for each source node.

    seed : int, optional
        Base random seed.

    Returns
    -------
    dict
        Mapping from each node to its mean spreading influence.
    """
    master_rng = random.Random(seed)

    influence = {}

    for source in G.nodes():

        node_seed = master_rng.randrange(0, 2 ** 32)

        influence[source] = sir_influence(
            G=G,
            source=source,
            beta=beta,
            gamma=gamma,
            simulations=simulations,
            seed=node_seed
        )

    return influence


def beta_settings(G):
    """
    Return the existing five beta_c-scaled transmission-rate settings.

    0.8 * beta_c
    0.9 * beta_c
    1.0 * beta_c
    1.1 * beta_c
    1.2 * beta_c
    """
    beta_c = epidemic_threshold(G)

    multipliers = [0.8, 0.9, 1.0, 1.1, 1.2]

    return {
        multiplier: multiplier * beta_c
        for multiplier in multipliers
    }


def rank_by_sir(influence):
    """
    Rank nodes in descending order according to
    their estimated SIR spreading influence.
    """
    return sorted(
        influence.items(),
        key=lambda item: item[1],
        reverse=True
    )


if __name__ == "__main__":

    # Demonstration using Zachary's Karate Club network.
    G = nx.karate_club_graph()

    beta_c = epidemic_threshold(G)

    print(f"beta_c = {beta_c:.6f}")

    settings = beta_settings(G)

    for multiplier, beta in settings.items():
        print(
            f"{multiplier:.1f} * beta_c = {beta:.6f}"
        )

