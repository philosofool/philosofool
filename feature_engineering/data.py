"""Tools for data manipulation in philosofool/feature_engineering."""

from __future__ import annotations

from collections.abc import Hashable
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd


def split_pairs(df: pd.DataFrame, n_pairs_test: int, col1: Hashable, col2: Hashable, random_state: int | None = None):
    """Create train and test splits by pairs.

    This is useful for contexts where we want to validate interactions, such as
    with matrix-factorization.
    """
    train_idx, test_idx = split_pairs_indices(df, n_pairs_test, col1, col2, random_state)
    return (df.loc[train_idx], df.loc[test_idx])

def split_pairs_indices(df: pd.DataFrame, n_pairs_test: int, col1: Hashable, col2: Hashable, random_state: int | None = None) -> tuple[pd.Index, pd.Index]:
    """Split an index so pairs formed by col1 and col2 are in disjoint indexes."""
    all_pairs = _identify_pairs(df, col1, col2)
    unique_pairs = df.drop_duplicates(subset=[col1, col2])
    selected_pairs_idx = unique_pairs.sample(n=n_pairs_test, random_state=random_state).index
    test_identities = all_pairs.loc[selected_pairs_idx]
    test = all_pairs[all_pairs.isin(test_identities)]
    train_idx = df.index.difference(test.index)
    return train_idx, test.index

def _identify_pairs(df: pd.DataFrame, col1, col2) -> pd.Series:
    needed_shift = int(df[col1].max()).bit_length()
    return df[col1] + (df[col2].values << needed_shift)
