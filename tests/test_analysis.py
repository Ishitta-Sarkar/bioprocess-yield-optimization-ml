"""Summaries of the published runs."""

from bioprocess.analysis import best_process_conditions, summarize_dataset
from bioprocess.data import TARGET_COLUMN, load_dataset


def test_summary_counts_and_range():
    dataset = load_dataset()
    summary = summarize_dataset(dataset)
    assert summary["total_runs"] == 29
    assert summary["n_center_points"] == 5
    assert summary["highest_yield_g_l"] == dataset[TARGET_COLUMN].max()
    assert summary["lowest_yield_g_l"] == dataset[TARGET_COLUMN].min()
    assert summary["average_yield_g_l"] == pytest_approx_mean(dataset)


def test_best_runs_are_the_maximum_and_include_setpoints():
    dataset = load_dataset()
    best = best_process_conditions(dataset)
    assert set(best["run_id"]) == {27, 28, 29}
    assert best["red_pigment_yield_g_l"].nunique() == 1
    assert {"ph", "temperature_c", "agitation_rpm", "fermentation_time_h"} <= set(best.columns)


def pytest_approx_mean(dataset):
    return float(dataset[TARGET_COLUMN].mean())
