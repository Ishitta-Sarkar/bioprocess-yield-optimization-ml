"""Figure files are written and are non-empty PNGs."""

from bioprocess.data import load_dataset
from bioprocess.modeling import fit_predict_mean, leave_one_out_predictions
from bioprocess.plots import save_factor_figures, save_prediction_figure


def test_figures_are_pngs(tmp_path):
    dataset = load_dataset()
    factor_paths = save_factor_figures(dataset, tmp_path)
    predictions = dataset[["run_id"]].copy()
    predictions["observed_red_pigment_yield_g_l"] = dataset["red_pigment_yield_g_l"]
    predictions["mean_baseline"] = leave_one_out_predictions(dataset, fit_predict_mean)
    # The comparison figure expects every model column. Reuse the mean for the rest.
    for slug in ("linear_regression", "quadratic_response_surface", "random_forest"):
        predictions[slug] = predictions["mean_baseline"]
    model_path = save_prediction_figure(predictions, tmp_path)
    for path in [*factor_paths, model_path]:
        contents = path.read_bytes()
        assert contents[:8] == b"\x89PNG\r\n\x1a\n"
        assert len(contents) > 1000
