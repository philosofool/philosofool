"""Tools for data manipulation in philosofool/feature_engineering."""

from __future__ import annotations

from collections.abc import Hashable
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd


class PA:
    """Enumeration of baseball events."""
    Out_in_play = 0
    K_swinging = 1
    K_looking = 2
    BB = 3
    Single = 4
    Double = 5
    Triple = 6
    HR = 7
    Other = 8


_event_enumeration = {
    'field_out': PA.Out_in_play,
    'single': PA.Single,
    'force_out': PA.Out_in_play,
    'strikeout_swinging': PA.K_swinging,
    'strikeout_looking': PA.K_looking,
    'walk': PA.BB,
    'double': PA.Double,
    'grounded_into_double_play': PA.Out_in_play,
    'field_error': PA.Out_in_play,
    'home_run': PA.HR,
    'sac_bunt': PA.Out_in_play,
    'truncated_pa': PA.Other,
    'triple': PA.Triple,
    'hit_by_pitch': PA.Other,
    'double_play': PA.Out_in_play,
    'sac_fly': PA.Other,
    'fielders_choice': PA.Out_in_play,
    'catcher_interf': PA.Other,
    'fielders_choice_out': PA.Out_in_play,
    'strikeout_double_play': PA.K_swinging,
    'nan': PA.Other,
    'sac_fly_double_play': PA.Out_in_play,
    'triple_play': PA.Out_in_play,
}


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


def batter_pitcher_matchups() -> pd.DataFrame:
    """Return data of batter pitcher matchups.

    The data is all common outcomes which end the plate appearance for every batter-pitcher
    matchup in 2024.
    """
    path = '~/repos/philosofool/feature_engineering/batter_pitcher_matchups.parquet'
    data = pd.read_parquet(path).astype({'home_team': 'category'})
    return _preprocess_batter_pitcher_data(data)

def _preprocess_batter_pitcher_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['label'] = df.events.map(_event_enumeration)
    df = df[df['label'] != PA.Other].dropna()
    df['home_team'] = df.home_team.cat.codes
    values = df[['pitcher', 'batter', 'home_team', 'label']].astype(np.int64)
    return values