"""Print yield summaries and the runs with the highest red pigment yield."""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd

from bioprocess.analysis import best_process_conditions, summarize_dataset
from bioprocess.data import DATASET_CITATION, load_dataset
from bioprocess.paths import RESULTS_DIR


def main() -> None:
    dataset = load_dataset()
    summary = summarize_dataset(dataset)
    best = best_process_conditions(dataset)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd_summary = _summary_frame(summary)
    pd_summary.to_csv(RESULTS_DIR / "yield_summary.csv", index=False)
    best.to_csv(RESULTS_DIR / "best_runs.csv", index=False)

    print("Red pigment yield summary")
    print("Data:", DATASET_CITATION)
    print(f"Runs: {summary['total_runs']}")
    print(f"Mean yield (g/L): {summary['average_yield_g_l']:.3f}")
    print(f"SD (g/L): {summary['yield_std_g_l']:.3f}")
    print(f"Range (g/L): {summary['lowest_yield_g_l']:.3f} to {summary['highest_yield_g_l']:.3f}")
    print(
        "Center-point mean ± SD (g/L): "
        f"{summary['center_yield_mean_g_l']:.3f} ± {summary['center_yield_std_g_l']:.3f} "
        f"(n={summary['n_center_points']})"
    )
    print("Runs at the maximum red pigment yield:")
    print(best.to_string(index=False))


def _summary_frame(summary: dict) -> pd.DataFrame:
    return pd.DataFrame([summary])


if __name__ == "__main__":
    main()
