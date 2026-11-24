"""Cross-validate yield models and write the metrics tables."""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd

from bioprocess.data import load_dataset
from bioprocess.modeling import evaluate_models, quadratic_insample_check
from bioprocess.paths import RESULTS_DIR


def main() -> None:
    dataset = load_dataset()
    summary, predictions, repeats = evaluate_models(dataset)
    check = quadratic_insample_check(dataset)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(RESULTS_DIR / "model_performance.csv", index=False)
    predictions.to_csv(RESULTS_DIR / "loo_predictions.csv", index=False)
    repeats.to_csv(RESULTS_DIR / "repeated_cv_scores.csv", index=False)
    pd.DataFrame([check]).to_csv(RESULTS_DIR / "quadratic_insample_check.csv", index=False)

    print("Leave-one-out cross-validation (one held-out run at a time)")
    print(
        summary[
            [
                "label",
                "loo_mae",
                "loo_rmse",
                "loo_r2",
                "loo_abs_error_std",
                "loo_abs_error_min",
                "loo_abs_error_max",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.3f}")
    )
    print()
    print("Repeated 5-fold cross-validation, 10 repeats (mean ± sample SD)")
    for _, row in summary.iterrows():
        print(
            f"{row['label']}: MAE {row['repeated_cv_mae_mean']:.3f} ± "
            f"{row['repeated_cv_mae_std']:.3f}; "
            f"R² {row['repeated_cv_r2_mean']:.3f} ± {row['repeated_cv_r2_std']:.3f}"
        )
    print()
    print("In-sample quadratic check against the printed Equation (9), not a test score:")
    print(
        f"  fitted center prediction {check['predicted_center_g_l']:.3f} g/L; "
        f"center observations {check['center_observation_mean_g_l']:.3f} ± "
        f"{check['center_observation_std_g_l']:.3f} g/L"
    )
    print(
        f"  in-sample R² {check['insample_r2']:.3f} "
        f"(article reports {check['paper_reported_insample_r2']:.4f})"
    )


if __name__ == "__main__":
    main()
