# Vergleich von xLSTM und LSTM für multimodale Aktienkursvorhersage

**Bachelor Thesis** — Angewandte Informatik (SPO 2)  
Hochschule Heilbronn | Julian Martens | March 2026  
Supervisors: Alexander Windberger, Florian Kauffeldt

---

## Abstract

This thesis examines classical and extended recurrent neural network architectures for stock price forecasting with unimodal and multimodal input data. A Long Short-Term Memory (LSTM) model is compared with Extended Long Short-Term Memory (xLSTM). The aim is to analyze whether architectural enhancements and the integration of sentiment information from financial news improve forecast quality.

Experiments are based on historical closing prices of ten S&P 500 companies (2009–2025). Sentiment information is extracted with FinBERT. Twenty model configurations are evaluated across two architectures, two feature sets, and five forecast horizons using RMSE and coefficient of determination (R²). A simplified portfolio simulation is performed as an additional evaluation step.

Results show that classical LSTM achieves a slightly higher and more consistent forecast quality than xLSTM in aggregated comparison. No systematic advantage of the extended architecture is identified. The integration of sentiment information does not lead to any significant performance improvement in the examined setting.

---

## Project Structure

```
.
├── config.json                          # Central hyperparameter and path configuration
├── src/
│   ├── data_prep.py                     # Data loading, normalization, windowing
│   ├── model_wrapper.py                 # Unified training/evaluation interface
│   └── models/
│       ├── LSTM.py                      # LSTM architecture (PyTorch)
│       └── xLSTM.py                     # xLSTM architecture (PyTorch)
├── notebooks/
│   ├── data/
│   │   ├── data_exploration.ipynb       # EDA on price and sentiment data
│   │   ├── news_import.ipynb            # News scraping and FinBERT sentiment extraction
│   │   └── google_colab_news_import.ipynb
│   ├── training/
│   │   ├── optuna_optimizing.ipynb      # Hyperparameter search with Optuna
│   │   ├── run_LSTM.ipynb               # Single LSTM training run
│   │   ├── run_xLSTM.ipynb              # Single xLSTM training run
│   │   └── train_best_models_all_tickers.ipynb  # Full cross-ticker training
│   └── evaluation/
│       ├── evaluate_saved_best_models.ipynb     # RMSE / R² evaluation across all tickers
│       ├── aapl_evaluation_xlstm_lstm.ipynb     # Detailed AAPL case study
│       └── portfolio_symulation.ipynb           # Simplified portfolio backtest
└── data/
    ├── ts_with_sentiment/               # Price + FinBERT sentiment CSVs per ticker
    ├── optuna/
    │   ├── studies/                     # Saved Optuna study databases
    │   └── checkpoints/                 # Intermediate Optuna trial checkpoints
    ├── best_models/                     # Saved weights of best-performing models
    └── results/
        ├── plots/                       # Generated figures
        └── csv/                         # Aggregated result tables
```

---

## Setup

### Requirements

- Python 3.10+
- PyTorch
- Transformers (HuggingFace) — FinBERT
- Optuna
- pandas, numpy, matplotlib, scikit-learn

Install dependencies:

```bash
pip install -r requirements.txt
```

### Configuration

All hyperparameters, data paths, and model settings are controlled via [`config.json`](config.json) at the project root. Adjust paths here if your data directory is located elsewhere.

---

## Usage

All workflows are implemented as Jupyter notebooks. Run them from the project root so that `src` resolves correctly as a package.

### 1 — Data preparation

| Notebook | Purpose |
|---|---|
| `notebooks/data/news_import.ipynb` | Scrape financial news and extract FinBERT sentiment scores |
| `notebooks/data/data_exploration.ipynb` | Explore and visualize price and sentiment data |

### 2 — Hyperparameter optimization

```
notebooks/training/optuna_optimizing.ipynb
```

Runs Optuna trials for LSTM and xLSTM across the configured search space. Results are stored in `data/optuna/studies/`.

### 3 — Training

| Notebook | Purpose |
|---|---|
| `notebooks/training/run_LSTM.ipynb` | Train a single LSTM configuration |
| `notebooks/training/run_xLSTM.ipynb` | Train a single xLSTM configuration |
| `notebooks/training/train_best_models_all_tickers.ipynb` | Train best Optuna configurations across all ten tickers |

### 4 — Evaluation

| Notebook | Purpose |
|---|---|
| `notebooks/evaluation/evaluate_saved_best_models.ipynb` | Compute RMSE and R² for all saved models |
| `notebooks/evaluation/aapl_evaluation_xlstm_lstm.ipynb` | In-depth analysis for AAPL |
| `notebooks/evaluation/portfolio_symulation.ipynb` | Simplified long/short portfolio backtest |

---

## Models

### LSTM (`src/models/LSTM.py`)

Standard stacked LSTM with configurable number of layers and hidden units. Trained with early stopping on validation loss.

### xLSTM (`src/models/xLSTM.py`)

Extended LSTM incorporating sLSTM (scalar memory) and mLSTM (matrix memory) blocks. Block positions are configurable via `xlstm_slstm_at` in `config.json`.

Both models share a unified interface through `src/model_wrapper.py`.

---

## Experimental Setup

| Parameter | Value |
|---|---|
| Tickers | 10 S&P 500 companies |
| Data period | 2009–2025 |
| Train / Val / Test split | 60 % / 20 % / 20 % |
| Input window | 60 trading days |
| Forecast horizons | 1, 3, 5, 10, 15, 30 days |
| Feature sets | Unimodal (Close only), Multimodal (Close + FinBERT sentiment) |
| Normalization | Percentage change |
| Metrics | RMSE, R² |

---

## Results Summary

Classical LSTM achieves slightly higher and more consistent forecast quality than xLSTM across the aggregated evaluation. No systematic advantage of the extended architecture is identified. The integration of FinBERT sentiment information does not lead to a significant performance improvement in the examined setting.

---

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.