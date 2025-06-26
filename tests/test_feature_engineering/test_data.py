import sys
print()
print(sys.path)

from feature_engineering.data import (
    _identify_pairs,
    split_pairs_indices,
    split_pairs,
    batter_pitcher_matchups
)

import numpy as np
import pandas as pd


def test_identify_pairs():
    df = pd.DataFrame({
        'a': [1, 2, 1, 2, 0, 2, 3],
        'b': [1, 1, 2, 2, 2, 0, 0]
    })
    result = _identify_pairs(df, 'a', 'b')
    np.testing.assert_array_equal(result, [5, 6, 9, 10, 8, 2, 3])
    assert isinstance(result, pd.Series)

def test_split_index_pairs():
    df = pd.DataFrame({
        'a': [1, 2, 1, 2, 2, 0, 0],
        'b': [1, 1, 2, 2, 2, 0, 0]
    })
    train, test = split_pairs_indices(df, 2, 'a', 'b')
    # assert len(test) == 2
    assert (3 in train and 4 in train) or (3 in test and 4 in test), "Pairs should be in exactly one index."
    assert (5 in train and 6 in train) or (5 in test and 6 in test), "Pairs should be in exactly one index."
    assert test.intersection(train).empty, "The intersection of the indexes should be empty."
    combined_index = test.union(train)
    assert len(df.index.intersection(combined_index)) == len(df.index), "The combined index should contain all entities from the data."
    assert len(df.index.union(combined_index)) == len(df.index), "The combined index should only contain entities from the data."

def test_split_index_pairs__size():
    rng = np.random.default_rng(seed=34987)
    df = pd.DataFrame({
        'a': rng.integers(0, 20, 10_000),
        'b': rng.integers(0, 20, 10_000)
    })
    for i in range(1, 10):
        train, test = split_pairs_indices(df, i, 'a', 'b', random_state=2)
        test_df = df.loc[test]
        assert len(test_df.drop_duplicates()) == i

def test_split_pairs():
    df = pd.DataFrame({
        'a': [1, 2, 1, 2, 2, 0, 0],
        'b': [1, 1, 2, 2, 2, 0, 0]
    })
    train, test = split_pairs(df, 2, 'a', 'b')


def test_batter_pitcher_matchups():
    df = batter_pitcher_matchups()
    expected_columns = ['pitcher', 'batter', 'home_team', 'label']
    assert len(df.columns.intersection(expected_columns)) == 4
    assert len(df.columns) == 4
    assert all(df[col].dtype == np.int64 for col in df)
    np.testing.assert_array_equal(np.sort(df['label'].unique()), [0, 1, 2, 3, 4, 5, 6, 7])
