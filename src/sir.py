"""
SIR spreading simulation used for evaluating node influence.

The implementation follows the experimental setting described in the
EQLG manuscript. Each node is used in turn as the initially infected
source, and its spreading influence is estimated from the mean final
outbreak size over repeated SIR simulations.
"""

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
    """
    Perform one discrete-time SIR simulation.

    Parameters
    ----------
    G : networkx.Graph
        Input network.

    source : hashable
        Initially infected node.

    beta : float
        Infection probability.

    gamma : float, default=1.0
        Recovery probability.

    rng : random.Random, optional
        Random-number generator.

    Returns
    -------
    int
        Final outbreak size, i.e., the total number of nodes
        that have been infected during the spreading process.
    """
    if source not in G:
        raise ValueError("The source node is not contained in the graph.")

    if not 0.0 <= beta <= 1.0:
        raise ValueError("beta must lie in [0, 1].")

    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must lie in [0, 1].")

    if rng is None:
        rng = random.Random()

    susceptible = set(G.nodes())
    infected = {source}
    recovered = set()

    susceptible.remove(source)

    while infected:

        newly_infected = set()
        newly_recovered = set()

        # Infection attempts are made by nodes infected
        # at the beginning of the current time step.
        for v in infected:

            for u in G.neighbors(v):

                if u in susceptible and rng.random() < beta:
                    newly_infected.add(u)

            if rng.random() < gamma:
                newly_recovered.add(v)

        susceptible.difference_update(newly_infected)

        infected.update(newly_infected)
        infected.difference_update(newly_recovered)

        recovered.update(newly_recovered)

    return len(recovered)


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
        Infection probability.

    gamma : float, default=1.0
        Recovery probability.

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
        Infection probability.

    gamma : float, default=1.0
        Recovery probability.

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
    Return the five infection probabilities used in the manuscript.

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
