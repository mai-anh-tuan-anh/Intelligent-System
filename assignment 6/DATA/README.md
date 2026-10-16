# Assignment 06 - Recurrent Neural Networks for Time-Series Forecasting

This project implements the complete Assignment 06 workflow as an executable technical report and a separate Flask web demonstration.

## Assignment coverage

Two time-related datasets are modeled:

1. **E-commerce service process**: the Olist order-lifecycle table is transformed into a chronological service-performance series. The main prediction target is the next observed service period's average delivery time.
2. **Stock market**: daily AAPL historical market data are used to predict the next trading step's closing price.

Each dataset is implemented three ways:

- RNN from scratch with NumPy and manual BPTT/Adam
- RNN with TensorFlow/Keras `SimpleRNN`
- RNN with PyTorch `nn.RNN`

The notebook includes mathematical derivations, time-series window construction, leakage prevention, EDA, training, early stopping, learning-rate reduction, gradient clipping, numerical gradient checking, evaluation, residual diagnostics, model comparison, and model-artifact export.

## Project structure

```text
assignment_06_RNN/
├── notebooks/
│   └── RNN_Assignment_A_to_Z.ipynb
├── data/
│   ├── ecommerce/
│   │   ├── README.md
│   │   └── olist_orders_dataset.csv        # downloaded/cached by the notebook if available
│   └── stock/
│       ├── README.md
│       └── AAPL_daily.csv                  # downloaded/cached by the notebook
├── models/
│   ├── ecommerce/
│   └── stock/
├── outputs/
│   ├── figures/
│   ├── metrics/
│   └── predictions/
└── web/
    ├── app.py
    ├── templates/index.html
    └── static/
        ├── app.js
        └── style.css
```

## Environment

Python 3.10-3.12 is recommended for a stable TensorFlow/Jupyter workflow.

Install the project dependencies:

```bash
pip install -r requirements.txt
```

The PyTorch package may need a platform-specific build depending on the local CPU/CUDA environment. The notebook itself forces the model experiments onto **CPU** so that Scratch, Keras, and PyTorch are compared under a common hardware setting.

## Dataset setup

### E-commerce

Primary dataset: **Brazilian E-Commerce Public Dataset by Olist**.

Place the Olist orders table at:

```text
data/ecommerce/olist_orders_dataset.csv
```

The notebook also attempts to download a public mirror automatically when the file is missing. For a formal submission, keep the source URL and the locally used CSV documented in the notebook.

### Stock

The notebook downloads and caches daily AAPL data using `yfinance` when the CSV is not already present.

The experiment uses a fixed historical period from `2010-01-01` through `2025-12-31`, with the API end date set to `2026-01-01` because the end parameter is exclusive.

## Notebook execution order

Open:

```text
notebooks/RNN_Assignment_A_to_Z.ipynb
```

Run the cells from top to bottom. The long-running sections are the six training experiments:

```text
E-commerce: Scratch → Keras → PyTorch
Stock:      Scratch → Keras → PyTorch
```

The notebook is intentionally written like a technical report. Markdown cells introduce the problem, formulas, tensor shapes, and design decisions before the corresponding code cells.

## Evaluation

Primary regression metrics:

- MAE
- RMSE
- MAPE
- R²

Additional analysis:

- persistence baseline
- baseline-relative MAE/RMSE improvement
- residual distributions
- rolling MAE
- actual-vs-predicted plots
- prediction scatter plots
- training/validation curves
- gradient-norm monitoring for Scratch
- hidden-state trajectories and activation maps
- stock directional accuracy and directional confusion matrices
- parameter-count and training-time comparison

## Web demo

After the notebook completes the export section:

```bash
cd web
python app.py
```

Open the local Flask address in a browser.

The dashboard provides:

- dataset and model selectors;
- latest observed value and next-step forecast;
- MAE, RMSE, MAPE, and R² cards;
- test-set actual-vs-predicted chart;
- training/validation loss chart;
- residual chart;
- absolute-error distribution;
- six-model comparison table including the persistence baseline;
- architecture and input-window information;
- recent normalized input window;
- metric definitions.

The website reads notebook-generated JSON artifacts. It does **not** retrain the neural networks in the browser.

## Important interpretation notes

The e-commerce series is an assignment-oriented transformation of order-level lifecycle data into aggregated service-performance observations. It is not an individual-order service-response model.

The stock experiment is a historical time-series learning exercise, not a trading system. Predictive error on historical data should not be interpreted as a guarantee of future financial performance.
