"""Load and check the published fermentation table."""

from pathlib import Path

import pandas as pd

from bioprocess.paths import DATA_PATH

CODED_FEATURES = (
    "ph_coded",
    "temperature_coded",
    "agitation_coded",
    "time_coded",
)
ACTUAL_FEATURES = (
    "ph",
    "temperature_c",
    "agitation_rpm",
    "fermentation_time_h",
)
TARGET_COLUMN = "red_pigment_yield_g_l"
OUTCOME_COLUMNS = (
    "orange_pigment_yield_g_l",
    TARGET_COLUMN,
    "biomass_g_l",
)

# Table 1 of Venkatachalam et al. (2020): coded level to setpoint.
CODED_TO_ACTUAL = {
    "ph_coded": {-1: 4.0, 0: 6.5, 1: 9.0},
    "temperature_coded": {-1: 21.0, 0: 24.0, 1: 27.0},
    "agitation_coded": {-1: 100.0, 0: 150.0, 1: 200.0},
    "time_coded": {-1: 24.0, 0: 168.0, 1: 336.0},
}
ACTUAL_COLUMN_FOR_CODED = {
    "ph_coded": "ph",
    "temperature_coded": "temperature_c",
    "agitation_coded": "agitation_rpm",
    "time_coded": "fermentation_time_h",
}

DATASET_DOI = "10.3390/microorganisms8050711"
DATASET_CITATION = (
    "Venkatachalam M, Shum-Chéong-Sing A, Dufossé L, Fouillaud M. "
    "Microorganisms. 2020;8(5):711. "
    f"https://doi.org/{DATASET_DOI}"
)

EXPECTED_ROWS = 29


def load_dataset(file_path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Read the fermentation table and reject a file that does not match Table 2."""
    dataset = pd.read_csv(file_path)
    _validate(dataset)
    return dataset


def _validate(dataset: pd.DataFrame) -> None:
    required = (
        "run_id",
        *CODED_FEATURES,
        *ACTUAL_FEATURES,
        *OUTCOME_COLUMNS,
    )
    missing = [column for column in required if column not in dataset.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")
    if len(dataset) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} runs, found {len(dataset)}")
    if dataset["run_id"].nunique() != EXPECTED_ROWS:
        raise ValueError("run_id values are not unique")
    if dataset[list(required)].isna().any().any():
        raise ValueError("Dataset contains missing values")

    for coded, mapping in CODED_TO_ACTUAL.items():
        levels = set(dataset[coded].astype(int))
        if levels != set(mapping):
            raise ValueError(f"{coded} has levels {sorted(levels)}, expected -1, 0, 1")
        actual = ACTUAL_COLUMN_FOR_CODED[coded]
        decoded = dataset[coded].astype(int).map(mapping)
        if not decoded.eq(dataset[actual].astype(float)).all():
            raise ValueError(f"{actual} does not match Table 1 decoding of {coded}")
