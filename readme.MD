# Demand Forecasting: Statistical vs Machine Learning Models

This project compares traditional statistical models (like AR, ARIMA, ETS, Prophet) with modern machine learning methods (Random Forest, XGBoost, LSTM) for demand forecasting. The analysis uses the **M5 Walmart sales dataset** and evaluates performance based on multi-step forecasting accuracy.

---

## 📁 Project Structure

```
demand-forecasting-ml-vs-stats/
|
├── data/                  # Raw and cleaned datasets
│   └── m5/                # Contains downloaded or cleaned M5 CSVs
|
├── preprocessing/         # Data cleaning pipeline
|   |── download_m5.py     # Downloads M5 dataset ZIP from Google Drive
│   └── clean_m5.py        # Converts raw M5 data into tidy format
|
├── models/                # Forecasting models
│   ├── ar_model.py
│   ├── arima_model.py
│   ├── ets_model.py
│   ├── prophet_model.py
│   ├── rf_model.py
│   ├── xgboost_model.py
│   └── lstm_model.py
|
├── evaluation/            # Metrics, plots, rolling split logic
|
├── notebooks/             # Exploratory notebooks
|
├── main.py                # Main pipeline to run models and evaluate them
├── requirements.txt       # All Python dependencies
└── README.md
```

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/demand-forecasting-ml-vs-stats.git
cd demand-forecasting-ml-vs-stats
```

### 2. Install Requirements

Make sure you have Python ≥ 3.8 and then run:

```bash
python -m pip install -r requirements.txt
```

### 3. Download the M5 Dataset

Run the downloader script (saves ZIP to `data/m5/` and extracts automatically):

```bash
python scripts/download_m5.py
```

### 4. Clean the Data

Once the raw files are downloaded:

```bash
python preprocessing/clean_m5.py
```

This creates `data/m5/m5_clean.csv`, a long-format tidy version of the dataset.

---

## 🚀 Running the Models

To run the full forecasting + evaluation pipeline:

```bash
python main.py
```

Or run a specific model (example for ARIMA):

```python
from models.arima_model import ARIMAForecaster
```

(Use this within a notebook or script where you load the training and test data.)

---

## 📈 Output

Evaluation results are stored or printed from `main.py`, including metrics like:

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)

Forecast plots can also be generated in the `evaluation/` module.

---

## 📌 Notes

* This project currently uses **only the M5 dataset**.
* All models support **multi-step forecasting** (e.g. predicting the last 20% of a time series).
* The `models/` folder is modular — you can plug in new models with the same interface (`fit`, `predict`, `evaluate`).

---

## License

MIT License
