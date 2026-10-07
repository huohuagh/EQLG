"""
Evaluation metrics used in the EQLG experiments.

This module provides:

1. Kendall rank correlation
2. Jaccard similarity for top influential nodes
3. Monotonicity for ranking discrimination

These metrics correspond to the evaluation criteria
reported in the EQLG manuscript.
"""

from collections import Counter


def _check_same_nodes(scores_a, scores_b):
    """
    Check whether two score dictionaries contain
    exactly the same set of nodes.
    """
    nodes_a = set(scores_a.keys())
    nodes_b = set(scores_b.keys())

    if nodes_a != nodes_b:
        raise ValueError(
            "The two score dictionaries must contain "
            "the same set of nodes."
        )


def kendall_correlation(scores_a, scores_b):
    """
    Compute Kendall rank correlation between two
    node-score sequences.

    The implementation follows the pairwise definition:

        tau = 2 * (n_c - n_d) / [n * (n - 1)]

    where n_c and n_d denote the numbers of concordant
    and discordant node pairs, respectively.

    Tied pairs contribute neither to n_c nor to n_d.

    Parameters
    ----------
    scores_a : dict
        First node-score mapping.

    scores_b : dict
        Second node-score mapping.

    Returns
    -------
    float
        Kendall rank correlation.
    """
    _check_same_nodes(scores_a, scores_b)

    nodes = list(scores_a.keys())
    n = len(nodes)

    if n < 2:
        return 1.0

    concordant = 0
    discordant = 0

    for i in range(n - 1):

        node_i = nodes[i]

        for j in range(i + 1, n):

            node_j = nodes[j]

            diff_a = (
                scores_a[node_i]
                - scores_a[node_j]
            )

            diff_b = (
                scores_b[node_i]
                - scores_b[node_j]
            )

            product = diff_a * diff_b

            if product > 0:
                concordant += 1

            elif product < 0:
                discordant += 1

            # Tied pairs contribute 0.

    tau = (
        2.0
        * (concordant - discordant)
        / (n * (n - 1))
    )

    return tau


def top_nodes(scores, fraction=0.20, k=None):
    """
    Select the highest-ranked nodes.

    Parameters
    ----------
    scores : dict
        Mapping from node to score.

    fraction : float, default=0.20
        Fraction of top nodes to select when k is not given.

    k : int, optional
        Explicit number of nodes to select.

    Returns
    -------
    set
        Set containing the selected top-ranked nodes.
    """
    n = len(scores)

    if n == 0:
        return set()

    if k is None:

        if not 0 < fraction <= 1:
            raise ValueError(
                "fraction must lie in the interval (0, 1]."
            )

        k = max(1, int(n * fraction))

    if k < 1 or k > n:
        raise ValueError(
            "k must be between 1 and the number of nodes."
        )

    ranking = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True
    )

    return {
        node
        for node, _ in ranking[:k]
    }


def jaccard_similarity(
    scores_a,
    scores_b,
    fraction=0.20,
    k=None
):
    """
    Compute Jaccard similarity between the top-ranked
    node sets produced by two methods.

    J(X, Y) = |X intersection Y| / |X union Y|

    By default, the top 20% of nodes are compared.

    Parameters
    ----------
    scores_a : dict
        First node-score mapping.

    scores_b : dict
        Second node-score mapping.

    fraction : float, default=0.20
        Fraction of top-ranked nodes.

    k : int, optional
        Explicit top-k value.

    Returns
    -------
    float
        Jaccard similarity in [0, 1].
    """
    _check_same_nodes(scores_a, scores_b)

    set_a = top_nodes(
        scores_a,
        fraction=fraction,
        k=k
    )

    set_b = top_nodes(
        scores_b,
        fraction=fraction,
        k=k
    )

    union = set_a | set_b

    if not union:
        return 1.0

    intersection = set_a & set_b

    return len(intersection) / len(union)


def monotonicity(scores):
    """
    Compute ranking monotonicity.

    Let n_r denote the number of nodes sharing the same
    score or rank r. The monotonicity is defined as

        M(R) =
        [1 - sum_r n_r(n_r - 1) / (n(n - 1))]^2

    A value close to 1 indicates strong discrimination
    capability and few tied scores.

    Parameters
    ----------
    scores : dict
        Node-score mapping.

    Returns
    -------
    float
        Monotonicity in [0, 1].
    """
    n = len(scores)

    if n < 2:
        return 1.0

    counts = Counter(scores.values())

    tied_pairs_term = sum(
        count * (count - 1)
        for count in counts.values()
    )

    value = (
        1.0
        - tied_pairs_term / (n * (n - 1))
    )

    return value ** 2


if __name__ == "__main__":

    # Small demonstration
    method_scores = {
        1: 10.0,
        2: 8.0,
        3: 8.0,
        4: 3.0
    }

    reference_scores = {
        1: 9.0,
        2: 7.0,
        3: 5.0,
        4: 2.0
    }

    print(
        "Kendall correlation:",
        kendall_correlation(
            method_scores,
            reference_scores
        )
    )

    print(
        "Jaccard similarity:",
        jaccard_similarity(
            method_scores,
            reference_scores,
            fraction=0.5
        )
    )

    print(
        "Monotonicity:",
        monotonicity(method_scores)
    )
