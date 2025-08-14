import numpy as np
from numpy.typing import ArrayLike

def mean_precision_at_k(recommended: ArrayLike, relevant: ArrayLike) -> np.floating:
    """
    Compute the mean precision at K for a set of recommended items.

    Mean precision at K is the average of the precision values calculated
    at the rank positions where relevant items occur. Precision at a given
    position K is the proportion of recommended items in the top-K positions
    that are relevant.

    Parameters
    ----------
    recommended : ArrayLike
        Sequence of recommended item IDs or labels, ordered by rank (best first).
    relevant : ArrayLike
        Sequence of relevant item IDs or labels.

    Returns
    -------
    np.floating
        The mean precision at K, averaged over all relevant items.

    Notes
    -----
    - Both `recommended` and `relevant` can be array-like objects
      (lists, NumPy arrays, Pandas Series, etc.).
    - If there are no relevant items, the function will raise a division-by-zero
      error.

    Examples
    --------
    >>> mean_precision_at_k([1, 2, 3], [1, 2])
    0.75
    """
    recommended = np.asarray(recommended)
    relevant = np.asarray(relevant)
    rec_is_relevant = np.isin(recommended, relevant).astype(int)
    precision_at_k = np.cumsum(rec_is_relevant, axis=-1) / np.arange(1, recommended.shape[-1] + 1)
    avg_precision_at_k = precision_at_k * rec_is_relevant
    return np.sum(avg_precision_at_k, axis=-1) / relevant.size


def test_mean_precision_at_k():
    relevant = [1, 2]
    recommended = [1, 2, 3]
    expected = 1.
    np.testing.assert_equal(mean_precision_at_k(recommended, relevant), expected)

    recommended1 = [3, 1, 2]
    expected1 = (0 + .5 + 2/3) * 1/2
    assert mean_precision_at_k(recommended1, relevant) == expected1

    recommended2 = [1, 3, 2]
    expected2 = (1 + 0 + 2/3) * 1/2
    assert mean_precision_at_k(recommended2, relevant) == expected2

    recommended3 = [recommended, recommended1]
    expected3 = np.array([expected, expected1])
    np.testing.assert_equal(mean_precision_at_k(recommended3, relevant), expected3)


if __name__ == '__main__':
    test_mean_precision_at_k()
