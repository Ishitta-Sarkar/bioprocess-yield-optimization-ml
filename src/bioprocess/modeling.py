"""Cross-validated yield models and a mean baseline.

Hyperparameters are fixed. They are not chosen by searching the cross-validation score.
"""

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut, RepeatedKFold
from sklearn.preprocessing import PolynomialFeatures

from bioprocess.data import CODED_FEATURES, TARGET_COLUMN

N_SPLITS = 5
N_REPEATS = 10
CV_RANDOM_STATE = 42
FOREST_TREES = 200
FOREST_MIN_LEAF = 2

FitPredict = Callable[[pd.DataFrame, pd.Series, pd.DataFrame], np.ndarray]


def fit_predict_mean(
    _x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame
) -> np.ndarray:
    """Predict the training-fold mean for every test row."""
    return np.full(len(x_test), float(y_train.mean()))


def fit_predict_linear(
    x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame
) -> np.ndarray:
    model = LinearRegression()
    model.fit(x_train[list(CODED_FEATURES)], y_train)
    return np.asarray(model.predict(x_test[list(CODED_FEATURES)]), dtype=float)


def fit_predict_quadratic(
    x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame
) -> np.ndarray:
    """Second-order polynomial in the coded factors, including interactions."""
    expansion = PolynomialFeatures(degree=2, include_bias=False)
    model = LinearRegression()
    model.fit(expansion.fit_transform(x_train[list(CODED_FEATURES)]), y_train)
    return np.asarray(
        model.predict(expansion.transform(x_test[list(CODED_FEATURES)])),
        dtype=float,
    )


def fit_predict_forest(
    x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame
) -> np.ndarray:
    model = RandomForestRegressor(
        n_estimators=FOREST_TREES,
        min_samples_leaf=FOREST_MIN_LEAF,
        max_features=None,
        random_state=CV_RANDOM_STATE,
        n_jobs=1,
    )
    model.fit(x_train[list(CODED_FEATURES)], y_train)
    return np.asarray(model.predict(x_test[list(CODED_FEATURES)]), dtype=float)


MODELS: tuple[tuple[str, str, FitPredict], ...] = (
    ("mean_baseline", "Mean baseline", fit_predict_mean),
    ("linear_regression", "Linear regression", fit_predict_linear),
    ("quadratic_response_surface", "Quadratic response surface", fit_predict_quadratic),
    ("random_forest", "Random forest", fit_predict_forest),
)


def leave_one_out_predictions(dataset: pd.DataFrame, fit_predict: FitPredict) -> np.ndarray:
    """One held-out prediction per run. The held-out row is not used to fit."""
    features = dataset[list(CODED_FEATURES)]
    target = dataset[TARGET_COLUMN]
    predictions = np.empty(len(dataset), dtype=float)
    for train_index, test_index in LeaveOneOut().split(features):
        predictions[test_index] = fit_predict(
            features.iloc[train_index],
            target.iloc[train_index],
            features.iloc[test_index],
        )
    return predictions


def repeated_kfold_scores(
    dataset: pd.DataFrame,
    fit_predict: FitPredict,
    n_splits: int = N_SPLITS,
    n_repeats: int = N_REPEATS,
    random_state: int = CV_RANDOM_STATE,
) -> pd.DataFrame:
    """Out-of-fold MAE and R² for each repeat of k-fold cross-validation."""
    features = dataset[list(CODED_FEATURES)]
    target = dataset[TARGET_COLUMN].to_numpy(dtype=float)
    splitter = RepeatedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state,
    )
    folds = list(splitter.split(features))
    rows = []
    for repeat in range(n_repeats):
        predictions = np.empty(len(dataset), dtype=float)
        start = repeat * n_splits
        for train_index, test_index in folds[start : start + n_splits]:
            predictions[test_index] = fit_predict(
                features.iloc[train_index],
                dataset[TARGET_COLUMN].iloc[train_index],
                features.iloc[test_index],
            )
        rows.append(
            {
                "repeat": repeat,
                "mae": float(mean_absolute_error(target, predictions)),
                "r2": float(r2_score(target, predictions)),
            }
        )
    return pd.DataFrame(rows)


def regression_scores(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = np.abs(np.asarray(observed, dtype=float) - np.asarray(predicted, dtype=float))
    return {
        "mae": float(mean_absolute_error(observed, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(observed, predicted))),
        "r2": float(r2_score(observed, predicted)),
        "abs_error_std": float(error.std(ddof=1)),
        "abs_error_min": float(error.min()),
        "abs_error_median": float(np.median(error)),
        "abs_error_max": float(error.max()),
    }


def evaluate_models(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Leave-one-out and repeated cross-validation for every model.

    Returns the summary table, per-run leave-one-out predictions, and per-repeat scores.
    """
    observed = dataset[TARGET_COLUMN].to_numpy(dtype=float)
    prediction_frame = pd.DataFrame(
        {
            "run_id": dataset["run_id"].to_numpy(),
            "observed_red_pigment_yield_g_l": observed,
        }
    )
    summary_rows = []
    repeat_frames = []
    for slug, label, fit_predict in MODELS:
        loo = leave_one_out_predictions(dataset, fit_predict)
        prediction_frame[slug] = loo
        scores = regression_scores(observed, loo)
        repeats = repeated_kfold_scores(dataset, fit_predict)
        repeats.insert(0, "model", slug)
        repeat_frames.append(repeats)
        summary_rows.append(
            {
                "model": slug,
                "label": label,
                "loo_mae": scores["mae"],
                "loo_rmse": scores["rmse"],
                "loo_r2": scores["r2"],
                "loo_abs_error_std": scores["abs_error_std"],
                "loo_abs_error_min": scores["abs_error_min"],
                "loo_abs_error_median": scores["abs_error_median"],
                "loo_abs_error_max": scores["abs_error_max"],
                "repeated_cv_mae_mean": float(repeats["mae"].mean()),
                "repeated_cv_mae_std": float(repeats["mae"].std(ddof=1)),
                "repeated_cv_r2_mean": float(repeats["r2"].mean()),
                "repeated_cv_r2_std": float(repeats["r2"].std(ddof=1)),
                "n_splits": N_SPLITS,
                "n_repeats": N_REPEATS,
            }
        )
    summary = pd.DataFrame(summary_rows)
    repeats_all = pd.concat(repeat_frames, ignore_index=True)
    return summary, prediction_frame, repeats_all


def quadratic_insample_check(dataset: pd.DataFrame) -> dict[str, float]:
    """In-sample quadratic fit, used only to compare Table 2 with Equation (9).

    This is not a performance claim. Cross-validation scores are the reported metrics.
    """
    expansion = PolynomialFeatures(degree=2, include_bias=False)
    coded = dataset[list(CODED_FEATURES)]
    target = dataset[TARGET_COLUMN]
    model = LinearRegression()
    model.fit(expansion.fit_transform(coded), target)
    center = pd.DataFrame([{name: 0.0 for name in CODED_FEATURES}])
    center_prediction = float(model.predict(expansion.transform(center))[0])
    fitted = model.predict(expansion.transform(coded))
    is_center = (coded.astype(int) == 0).all(axis=1)
    return {
        "insample_r2": float(r2_score(target, fitted)),
        "insample_mae": float(mean_absolute_error(target, fitted)),
        "predicted_center_g_l": center_prediction,
        "center_observation_mean_g_l": float(target.loc[is_center].mean()),
        "center_observation_std_g_l": float(target.loc[is_center].std(ddof=1)),
        "paper_equation_9_intercept_g_l": 1.92,
        "paper_reported_insample_r2": 0.9734,
    }
