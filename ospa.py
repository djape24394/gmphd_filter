from scipy.optimize import linear_sum_assignment
import numpy as np
import numpy.linalg as lin



def _calculation(X, Y, c, p):
    def d_c(x, y, c):
        return min(c, lin.norm(x - y))

    m = len(X)
    n = len(Y)
    if m == 0 and n == 0:
        return 0, 0, 0

    if m > n:
        # swap
        X, Y = Y, X
        m, n = n, m

    card_dist = c ** p * (n - m)
    local_dist = None

    D = np.zeros((n, n))
    for i in range(m):
        for j in range(n):
            D[i, j] = d_c(X[i], Y[j], c) ** p
    D[m:, :] = c ** p
    row_ind, col_ind = linear_sum_assignment(D)
    local_dist = D[row_ind[:m], col_ind[:m]].sum()

    return local_dist, card_dist, n


def ospa(X, Y, c, p):
    """
    Calculates Optimal Subpattern Assignment (OSPA) metric, defined by Dominic Schuhmacher, Ba-Tuong Vo, and Ba-Ngu Vo
    in "A Consistent Metric for Performance Evaluation of Multi-Object Filters". This is implementation using Hungarian
    method.
    https://en.wikipedia.org/wiki/Hungarian_algorithm
    http://www.hungarianalgorithm.com/examplehungarianalgorithm.php
    :param X: set of ndarray vectors
    :param Y: set of ndarray vectors
    :param c: c>0 . "The cut-off parameter c determines the relative weighting of the penalties assigned to
    cardinality and localization errors. A value of c which corresponds to the magnitude
    of a typical localization error can be considered small and has the effect of emphasizing
    localization errors. A value of c which corresponds to the maximal distance between
    targets can be considered large and has the effect of emphasizing cardinality errors."
    from Bayesian Multiple Target Filtering Using Random Finite Sets, BA-NGU VO, BA-TUONG VO, AND DANIEL CLARK
    :param p: The order parameter p determines the sensitivity of the metric to outliers. p>=1
    :return:
    """
    local_dist, card_dist, n = _calculation(X, Y, c, p)
    if n == 0:
        return 0
    return (1 / n * (local_dist + card_dist)) ** (1 / p)


def ospa_local_card(X, Y, c, p):
    local_dist, card_dist, n = _calculation(X, Y, c, p)
    if n == 0:
        return 0, 0
    return (1 / n * local_dist) ** (1 / p), (1 / n * card_dist) ** (1 / p)


def ospa_all(X, Y, c, p):
    local_dist, card_dist, n = _calculation(X, Y, c, p)
    if n == 0:
        return 0, 0, 0
    return (1 / n * (local_dist + card_dist)) ** (1 / p), (1 / n * local_dist) ** (1 / p), (1 / n * card_dist) ** (
            1 / p)
