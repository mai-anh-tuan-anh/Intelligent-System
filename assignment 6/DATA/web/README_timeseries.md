# Interactive time-series RNN demo

The time-series demo uses `app.py`, `templates/index.html`, `static/style.css`,
and `static/app.js`, matching the folder structure in the screenshot. The former
WikiText next-word demo is preserved as `app_wikitext.py`.

## Run

From the `web` folder:

```bash
python -m pip install -r requirements_timeseries.txt
python app.py
```

The app opens at `http://127.0.0.1:8051`. Press `Ctrl+C` in the terminal to stop it.

## Notebook artifacts

Run the final export cell of `RNN_Assignment_A_to_Z(2).ipynb` first. The app expects:

```text
outputs/metrics/web_payload.json
models/ecommerce/feature_scaler.joblib
models/ecommerce/target_scaler.joblib
models/ecommerce/scratch_rnn.npz
models/ecommerce/keras_rnn.keras
models/ecommerce/pytorch_rnn.pt
models/stock/feature_scaler.joblib
models/stock/target_scaler.joblib
models/stock/scratch_rnn.npz
models/stock/keras_rnn.keras
models/stock/pytorch_rnn.pt
```

By default, the project root is the parent of `web/`. If those files were exported to another folder, set `RNN_PROJECT_ROOT` to that folder before starting the app.

## Input

The form starts with the most recent input window from the notebook, converted back to original feature units. Edit the table values or upload a CSV with a header containing all feature names and exactly 14 chronological rows (oldest first). Press **Predict** to run a new inference with the selected trained model. The web app does not retrain models.

Regression outputs are not probabilities. The displayed MAE is a test-set error reference, not a calibrated confidence interval.
