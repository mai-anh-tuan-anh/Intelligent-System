# A7 Sequence Lab — local web demo

This package is designed for the folder layout already used by the A7 project:

    DATA/
    ├── outputs/
    ├── web/          <-- extract this package here
    ├── notebooks/
    ├── models/
    └── data/

Extract the ZIP contents directly into the existing DATA/web folder. The package provides app.py, static/index.html, this README, and requirements-web.txt. The app automatically locates its parent DATA folder by finding outputs/ and models/.

The notebook must have completed export cell G1, which writes dashboard.json, model_registry.json, a7_inference.py, and the model checkpoints.

## Start

Open Anaconda Prompt, activate the notebook environment, and run:

    conda activate a7-rnn
    cd /d "C:\Users\ADMIN\Downloads\Intelligent System\assignment 7\DATA\web"
    python app.py

Open http://127.0.0.1:8000 in a browser. Stop with Ctrl+C.

The local HTTP server and charts use Python's standard library. Model inference uses the packages already installed for the notebook (PyTorch, pandas, NumPy; TensorFlow is needed for Keras inference).

## Dashboard pages

- Overview: experiment metrics, text and stock baselines, and measured training time.
- Review sentiment: test confusion matrix, per-class precision/recall, calibration chart, and review inference.
- Stock forecast: chronological test forecasts, persistence comparison, residuals, and optional OHLCV CSV inference.
- Model insights: learning curves, parameter counts, hidden-state norms, LSTM gates, and the selected experiment record.

## Notes

- Sentiment inference is for English text, matching the training data.
- Stock CSV inference expects completed rows with Date, Open, High, Low, Close, and Volume. It forecasts the next trading session after the final input row.
- This is a local research demo, not a live market feed or investment recommendation.
- If notebook exports are missing, the dashboard explains what to run instead of displaying fabricated results.
