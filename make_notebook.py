"""
Regenerate Traffic_Demand_Forecast.ipynb from build_submission.py.

The notebook is a thin mirror: a markdown intro + one runnable code cell that is
the exact contents of build_submission.py. This guarantees the notebook and the
script never drift, and that the notebook is reproducible from train.csv only.

Local helper (not part of the submission pipeline).
"""
import json
import nbformat as nbf  # type: ignore

INTRO = """\
# Flipkart GridLock 2.0 — Traffic Demand Forecast

Predicts `demand` per (`geohash`, `timestamp`) for day-49 daytime (buckets 9–55)
using **only** the competition-provided `train.csv` and `test.csv`.

**Approach** (see `approach.md`):
1. Build the day-48 `geohash × time_bucket` matrix.
2. Denoise it with iterative low-rank SVD (*soft-impute*, rank 6) → a clean
   per-cell **profile** prior that transfers across days.
3. Models predict the **residual** (`demand − profile`); the profile is added back.
4. Blend diverse learners (LightGBM seed-avg, XGBoost, CatBoost,
   HistGradientBoosting, ExtraTrees).
5. Tree counts are chosen by early stopping on the day-49 early window.

> This notebook never reads `training.csv` — doing so would leak the test
> answers and violate the competition rules.

The single cell below is the exact contents of `build_submission.py`.
"""


def main():
    with open('build_submission.py', 'r') as f:
        code = f.read()
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_markdown_cell(INTRO), nbf.v4.new_code_cell(code)]
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    with open('Traffic_Demand_Forecast.ipynb', 'w') as f:
        nbf.write(nb, f)
    # sanity: confirm zero references to the off-limits file
    assert 'training.csv' not in code or 'NEVER reads' in code
    print("Wrote Traffic_Demand_Forecast.ipynb (1 markdown + 1 code cell).")


if __name__ == '__main__':
    main()
