"""Write exploratory figures and the leave-one-out results figure."""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bioprocess.data import load_dataset
from bioprocess.modeling import evaluate_models
from bioprocess.paths import FIGURES_DIR
from bioprocess.plots import save_factor_figures, save_prediction_figure


def main() -> None:
    dataset = load_dataset()
    _, predictions, _ = evaluate_models(dataset)
    written = save_factor_figures(dataset, FIGURES_DIR)
    written.append(save_prediction_figure(predictions, FIGURES_DIR))
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
