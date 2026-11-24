"""Descriptive summaries of the fermentation runs."""

import pandas as pd

from bioprocess.data import CODED_FEATURES, TARGET_COLUMN


def summarize_dataset(dataset: pd.DataFrame) -> dict[str, float | int]:
    """Summarise red-pigment yield. Biomass and orange pigment are co-outcomes."""
    yield_g_l = dataset[TARGET_COLUMN]
    center = (dataset[list(CODED_FEATURES)].astype(int) == 0).all(axis=1)
    return {
        "total_runs": int(len(dataset)),
        "n_process_factors": len(CODED_FEATURES),
        "n_center_points": int(center.sum()),
        "average_yield_g_l": float(yield_g_l.mean()),
        "yield_std_g_l": float(yield_g_l.std(ddof=1)),
        "highest_yield_g_l": float(yield_g_l.max()),
        "lowest_yield_g_l": float(yield_g_l.min()),
        "center_yield_mean_g_l": float(dataset.loc[center, TARGET_COLUMN].mean()),
        "center_yield_std_g_l": float(dataset.loc[center, TARGET_COLUMN].std(ddof=1)),
    }


def best_process_conditions(dataset: pd.DataFrame) -> pd.DataFrame:
    """Return every run that attains the maximum red-pigment yield."""
    highest = dataset[TARGET_COLUMN].max()
    columns = [
        "run_id",
        "ph",
        "temperature_c",
        "agitation_rpm",
        "fermentation_time_h",
        "red_pigment_yield_g_l",
        "orange_pigment_yield_g_l",
        "biomass_g_l",
    ]
    return dataset.loc[dataset[TARGET_COLUMN] == highest, columns].reset_index(drop=True)
