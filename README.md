# best_predict_ml — setup & run

This repository contains a Jupyter notebook for analyzing registrations and experimenting with time-series forecasting.

Quick environment setup (recommended):

1) Use Python 3.11 and a venv (Windows PowerShell)

```powershell
py -3.11 -m venv .venv311
.\.venv311\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install --prefer-binary -r requirements.txt
```

2) Alternative: use conda (recommended on Windows if you prefer prebuilt packages)

```powershell
conda create -n best_predict python=3.11 -y
conda activate best_predict
conda install -c conda-forge pandas numpy matplotlib seaborn scikit-learn statsmodels pmdarima pytest -y
```

3) If you plan to use Prophet:
- Use the conda-forge channel or install cmdstan backend: `pip install 'prophet[cmdstan]'` and follow `cmdstanpy.install_cmdstan()`

Notes:
- This project prefers Python 3.11 on Windows to avoid building packages like pandas from source.
- Use the included `setup_venv.ps1` script to automate the venv creation & installation.
# Time series evaluation utilities — registration forecasting

This small toolkit prepares a monthly registrations time series and evaluates forecasting models (ARIMA) using walk-forward validation with standard metrics (MAE, RMSE, MAPE, R²).

Files added:
- `src/etl.py` — loading + robust date parsing + building a continuous monthly series
- `src/models_eval.py` — walk-forward evaluation for ARIMA and metric computation
- `run_evaluation.py` — CLI to evaluate your file or run a synthetic test
- `tests/test_models_eval.py` — lightweight tests verifying metrics and evaluation using synthetic data
- `requirements.txt` — pinned dependencies

Quick start:

1. Create and activate a venv (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Run synthetic verification (no data required):

```powershell
python run_evaluation.py --synthetic
```

3. Run against your dataset (replace path):

```powershell
python run_evaluation.py --file "C:/path/to/inscription.xlsx"
```

Notes:
- The notebook you provided included a `prophet` attempt which often requires additional backend setup (cmdstanpy). This toolkit focuses on ARIMA which is easy to run and evaluate automatically.
- If you want, I can refactor the notebook into this pipeline and add CI to run the tests automatically.
