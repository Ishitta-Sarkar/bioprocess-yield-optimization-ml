"""Checks on the transcribed fermentation table."""

import pandas as pd
import pytest

from bioprocess.data import (
    CODED_FEATURES,
    CODED_TO_ACTUAL,
    OUTCOME_COLUMNS,
    TARGET_COLUMN,
    load_dataset,
)

# Red pigment yield (g/L RYrp equivalent) as printed in Table 2.
PUBLISHED_RED_YIELD = (
    0.83,
    1.44,
    0.82,
    1.08,
    1.23,
    1.37,
    1.04,
    1.47,
    0.95,
    1.48,
    0.85,
    1.68,
    1.31,
    0.43,
    1.03,
    1.39,
    0.57,
    1.48,
    1.50,
    1.73,
    1.18,
    1.12,
    1.12,
    0.98,
    1.92,
    1.92,
    1.93,
    1.93,
    1.93,
)


def test_table_matches_published_red_yields():
    dataset = load_dataset()
    observed = tuple(dataset[TARGET_COLUMN].tolist())
    assert observed == PUBLISHED_RED_YIELD


def test_coded_levels_decode_with_table_1():
    dataset = load_dataset()
    for coded, mapping in CODED_TO_ACTUAL.items():
        assert set(dataset[coded].astype(int)) == {-1, 0, 1}
        assert mapping[0] in set(dataset[coded].map(mapping))


def test_five_center_points():
    dataset = load_dataset()
    center = (dataset[list(CODED_FEATURES)].astype(int) == 0).all(axis=1)
    assert int(center.sum()) == 5


def test_run_14_keeps_printed_negative_orange_yield():
    dataset = load_dataset()
    run = dataset.loc[dataset["run_id"] == 14].iloc[0]
    assert run["orange_pigment_yield_g_l"] == pytest.approx(-0.10)
    assert run["red_pigment_yield_g_l"] == pytest.approx(0.43)


def test_rejects_a_short_table(tmp_path):
    dataset = load_dataset().head(3)
    path = tmp_path / "short.csv"
    dataset.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Expected 29"):
        load_dataset(path)


def test_outcomes_are_not_named_as_process_factors():
    assert TARGET_COLUMN not in CODED_FEATURES
    for column in OUTCOME_COLUMNS:
        assert column not in CODED_FEATURES


def test_load_returns_a_frame():
    assert isinstance(load_dataset(), pd.DataFrame)
