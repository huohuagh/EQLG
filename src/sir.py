"""
Discrete-time SIR simulation utilities for node influence.

At each synchronous time step, every currently infected node attempts to
infect each susceptible neighbor with probability beta and then recovers
with probability gamma. Newly infected nodes become active at the next
step. Influence is estimated from the mean final recovered-node count.
"""

from math import isfinite
import random

import networkx as nx


def epidemic_threshold(G):
    """
    Compute the epidemic threshold beta_c.

    beta_c = <k> / (<k^2> - <k>)

    Parameters
    ----------
    G : networkx.Graph
        Undirected and unweighted network.

    Returns
    -------
    float
        Epidemic threshold beta_c.
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
    """Perform one synchronous discrete-time SIR simulation until extinction.

    beta is the infection probability per susceptible neighbor per step,
    in [0, 1]. gamma is the recovery probability per infected node per step,
    in (0, 1]; gamma=1 gives each infected node one infectious step.
    Infection attempts precede recovery for the nodes infected at the start
    of the step. Newly infected nodes neither transmit nor recover until
    the next step. Recovered nodes cannot be infected again.
    rng may be a seeded random.Random instance. Return the final number
    of recovered nodes, including the initial source.
    """
    if source not in G:
        raise ValueError("The source node is not contained in the graph.")
    if not isfinite(beta) or not 0.0 <= beta <= 1.0:
        raise ValueError("beta must be a finite probability in [0, 1].")
    if not isfinite(gamma) or not 0.0 < gamma <= 1.0:
        raise ValueError("gamma must be a finite probability in (0, 1].")
    if rng is None:
        rng = random.Random()

    susceptible = set(G.nodes())
    susceptible.remove(source)
    infected = [source]
    recovered = 0

    while infected:
        newly_infected = []
        newly_infected_set = set()
        surviving_infected = []

        # Only nodes infected at the start of this step are processed.
        for node in infected:
            for neighbor in G.neighbors(node):
                if (
                    neighbor in susceptible
                    and neighbor not in newly_infected_set
                    and rng.random() < beta
                ):
                    newly_infected.append(neighbor)
                    newly_infected_set.add(neighbor)

            if rng.random() < gamma:
                recovered += 1
            else:
                surviving_infected.append(node)

        # Apply new infections simultaneously; preserve traversal order
        # so seeded runs do not depend on iteration over a set.
        susceptible.difference_update(newly_infected_set)
        infected = surviving_infected + newly_infected

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
        Infection probability per susceptible neighbor per step, in [0, 1].

    gamma : float, default=1.0
        Recovery probability per infected node per step, in (0, 1].

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
        Infection probability per susceptible neighbor per step, in [0, 1].

    gamma : float, default=1.0
        Recovery probability per infected node per step, in (0, 1].

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
    Return the five beta_c-scaled infection-probability settings.

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


