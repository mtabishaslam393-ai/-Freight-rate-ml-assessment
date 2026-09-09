# Freight Rate Prediction — Spotter ML Assessment

**Author:** Muhammad Tabish Aslam  
**Email:** mtabishaslam393@gmail.com  
**GitHub:** github.com/mtabishaslam  

---

## Overview

XGBoost regression model to predict freight load rates.  
- **Training data:** 48,000 loads (Jan–Oct 2025)  
- **Validation:** 12,000 loads predicted  
- **December:** 31-day fixed-route forecast (Lexington → Fort Wayne)

## Results

| Metric | Score |
|--------|-------|
| MAE    | $232.47 |
| RMSE   | $681.31 |
| MAPE   | 13.93% |
| R²     | 0.8013 |

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Add data files
Place the following files inside the `data/` folder:
- `train_test.csv`
- `validation.csv`
- `validation_predictions_template.csv`
- `december_chart_inputs.csv`

### 3. Run the solution
```bash
python solution.py
```

### 4. Run the scorer
```bash
python score.py --predictions validation_predictions.csv \
                --december-predictions data/december_chart_inputs.csv
```

## Repository Structure
