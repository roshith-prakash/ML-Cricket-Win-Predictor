# IPL Live Win Probability Estimator & Ball-by-Ball Simulator 🏏

A machine learning application that predicts real-time win probability for chasing teams in the Indian Premier League (IPL) ball-by-ball, modeled after systems like ESPN Cricinfo's Forecaster and WASP.

---

## 📌 Architecture & Design Principles

1. **Strict Temporal Validation (No Future Leakage):**
   - **Training Set:** Seasons $\le 2022$ (906 matches, 105,437 ball states)
   - **Test Set:** Seasons $\ge 2023$ (279 matches, 32,450 ball states)
   - Split strictly by `match_id` so deliveries from the same match never span across train and test partitions.
2. **Deterministic Terminal Boundary Overrides:**
   - Target achieved ($\text{runs\_left} \le 0$) $\rightarrow$ $P(\text{Win}) = 1.0$ (100%)
   - All out ($\text{wickets\_in\_hand} = 0$) $\rightarrow$ $P(\text{Win}) = 0.0$ (0%)
   - Overs completed ($\text{balls\_left} = 0$) $\rightarrow$ $P(\text{Win}) = 0.0$ (0%)
3. **Probability Calibration:**
   - Raw tree ensembles are calibrated via **Isotonic Regression** (`CalibratedClassifierCV`), lowering test set **Brier Score from 0.2139 to 0.1781** while maintaining strong discrimination (**ROC-AUC 0.8403**).

---

## 📊 Model Benchmark Results (Test Set: Seasons $\ge 2023$)

| Model | Brier Score (↓) | Log Loss (↓) | ROC-AUC (↑) | Accuracy @ 0.5 (↑) |
| :--- | :---: | :---: | :---: | :---: |
| **Calibrated LightGBM (Isotonic)** | **0.1781** | **0.7800** | **0.8403** | **75.28%** |
| **Logistic Regression (Baseline)** | 0.1887 | 0.6167 | 0.8357 | 74.42% |
| **Raw LightGBM** | 0.2139 | 0.8140 | 0.8224 | 73.14% |

---

## 📁 Repository Structure

```
ML-Mini-Project/
├── ipl_json/                        # Raw Cricsheet match JSONs (1,243 files)
├── data/
│   └── processed/
│       ├── matches_metadata.csv     # 1,185 clean regulation matches
│       ├── chases_ballbyball.csv    # 137,887 ball-by-ball chase snapshots
│       └── calibration_and_roc_curves.png
├── models/
│   ├── logistic_baseline.pkl        # Logistic Regression pipeline
│   ├── lgbm_model.pkl               # Raw LightGBM pipeline
│   ├── calibrator.pkl               # Isotonic Calibrated LightGBM
│   ├── feature_columns.json         # Input column schema guarantee
│   └── training_summary.json        # Evaluation benchmark metrics
├── notebooks/                       # 100% Self-Contained Interactive Notebooks
│   ├── 01_data_cleaning_and_eda.ipynb           # JSON parsing, DLS/tie filtering & EDA
│   ├── 02_feature_engineering_and_training.ipynb# Temporal split, LightGBM & calibration
│   └── 03_evaluation_and_testing.ipynb          # Reliability plots, What-If tests & replays
├── app/                             # Frontend UI & Prediction Server
│   ├── app.py                       # Streamlit server & live prediction engine
│   └── components/
│       ├── win_gauge.py             # Plotly dual-color win gauge
│       └── probability_chart.py     # Plotly ball-by-ball timeline chart
└── requirements.txt
```

---

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Self-Contained Notebooks
Open the notebooks in VS Code or Jupyter Lab. All cells have been pre-executed with interactive charts:
- `notebooks/01_data_cleaning_and_eda.ipynb`
- `notebooks/02_feature_engineering_and_training.ipynb`
- `notebooks/03_evaluation_and_testing.ipynb`

### 3. Launch the Streamlit Frontend & Server
```bash
python -m streamlit run app/app.py
```
