# Changelog

## [0.2.0] - 2026-11-24

### Changed

- Replaced the 12-row table with Table 2 of Venkatachalam et al., *Microorganisms* 2020, 8, 711 (CC BY 4.0). The earlier table had no stated source.
- Replaced the 25% hold-out (three test rows) with leave-one-out cross-validation and repeated 5-fold cross-validation.
- Compared a random forest with a mean baseline, a linear model, and a quadratic response surface. The mean baseline has the lowest error.
- Moved scripts, figures, the notebook, and metric tables into `src/`, `figures/`, `notebooks/`, and `results/`. Removed the duplicate root-level `visualization.py`.

### Added

- Pinned `requirements.txt` and `requirements-dev.txt`.
- pytest suite and GitHub Actions CI (ruff and pytest).
- `CITATION.cff` and this changelog.
- Leave-one-out predicted-against-observed figure.

## [0.1.0] - 2026-06-21

### Added

- Initial scripts, an unlabelled 12-row fermentation table, and a hold-out random-forest score.
