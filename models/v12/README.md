# V12 - Ultra Conservative Ensemble Model

## Overview

V12 is the most advanced model in the match prediction pipeline, combining:
- **Classification** (winner prediction Q3/Q4)
- **Regression** (over/under points prediction)
- **Risk Management** (asymmetric penalty, conservative gates)
- **League Filtering** (only bet on predictable leagues)

## Key Innovations

### 1. **Hybrid Architecture**
- **Winner Model**: Multi-algorithm ensemble (LogReg + GB + XGBoost + LightGBM + CatBoost)
- **Points Model**: Gender-separated regression (Ridge + GB + XGBoost + CatBoost)
- Both models work together to make final betting decisions

### 2. **Asymmetric Risk Management**
- **Loss penalty = 2x win reward** - Losing bets are penalized more heavily
- Default is **NO_BET** - only bet when VERY confident
- Minimum 65% confidence required for winner bets
- Minimum 4-point edge required for over/under bets

### 3. **League Filtering**
- Historical performance tracking per league
- Only bet on leagues with proven predictability (home win rate ≠ 50%)
- Minimum 50 samples to evaluate a league
- Strong leagues: home win rate ≥57% or ≤43%

### 4. **Conservative Gates**
- Minimum graph points: 20 (Q3) / 28 (Q4)
- Minimum PBP events: 18 (Q3) / 26 (Q4)
- Maximum volatility: 0.70
- Data quality must be "good" or "excellent"

### 5. **Best Features from All Versions**
From V4: Pressure/comeback features, clutch window
From V6: Monte Carlo simulation
From V9: Scoring run features (momentum/racha)
From V11: Gender-separated regression
NEW: League context features, acceleration, efficiency metrics

## File Structure

```
training/v12/
├── __init__.py                    # Package marker
├── train_v12.py                   # Main training script
├── infer_match_v12.py             # Inference engine
├── eval_v12.py                    # Evaluation script
├── v12_cli.py                     # CLI interface
└── model_outputs/                 # Trained models (after training)
    ├── q3_clf_ensemble.joblib     # Q3 winner ensemble
    ├── q4_clf_ensemble.joblib     # Q4 winner ensemble
    ├── q3_clf_*.joblib            # Individual Q3 classifiers
    ├── q4_clf_*.joblib            # Individual Q4 classifiers
    ├── q3_*_reg_ensemble.joblib   # Q3 regression ensembles
    ├── q4_*_reg_ensemble.joblib   # Q4 regression ensembles
    ├── league_stats.json          # Historical league performance
    ├── all_metrics.csv            # Training metrics
    └── training_summary.json      # Full training report
```

## Usage

### Training

```bash
# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Train all V12 models
python training/v12/v12_cli.py train

# Or directly
python training/v12/train_v12.py
```

Training will:
1. Build hybrid samples from DB (both classification + regression targets)
2. Compute league statistics
3. Train classifier ensembles for Q3 and Q4
4. Train regressor ensembles (per quarter, per target type, per gender)
5. Save all models and metrics to `training/v12/model_outputs/`

### Prediction

```bash
# Basic prediction
python training/v12/v12_cli.py predict <match_id>

# With sportsbook line for over/under
python training/v12/v12_cli.py predict <match_id> --line 185.5

# JSON output
python training/v12/v12_cli.py predict <match_id> --json

# Specify quarter
python training/v12/v12_cli.py predict <match_id> --target q3
```

### Evaluation

```bash
# Evaluate on all historical data
python training/v12/v12_cli.py eval

# With filters
python training/v12/v12_cli.py eval --limit 1000 --start-date 2024-01-01 --odds 1.91

# JSON output
python training/v12/v12_cli.py eval --json
```

### League Statistics

```bash
# Show league performance
python training/v12/v12_cli.py leagues
```

## Model Architecture

### Classification (Winner)
```
Features (70+) → DictVectorizer → StandardScaler →
  ├─ Logistic Regression (C=0.5)
  ├─ Gradient Boosting (100 est, depth=3)
  ├─ XGBoost (150 est, depth=4) [optional]
  ├─ LightGBM (150 est, depth=5) [optional]
  └─ CatBoost (200 iter, depth=5) [optional]
       ↓
  Weighted Ensemble → Final Probability
```

### Regression (Points)
```
Features (70+) → DictVectorizer → StandardScaler →
  ├─ Ridge (alpha=1.0)
  ├─ Gradient Boosting (100 est, depth=3)
  ├─ XGBoost (100 est, depth=3) [optional]
  └─ CatBoost (150 iter, depth=4) [optional]
       ↓
  Weighted Ensemble → Predicted Points
```

### Decision Logic
```
IF volatility >= 0.70: NO_BET
IF confidence < 65%: NO_BET
IF league not bettable: NO_BET
IF data quality poor: NO_BET

IF can bet winner AND confidence >= 65%:
  → BET_HOME or BET_AWAY
ELIF can bet over/under AND edge >= 4 pts:
  → OVER or UNDER
ELSE:
  → NO_BET
```

## Features

### Core Features (from V1-V2)
- League bucketing (top 25 leagues)
- Team bucketing (top 150 teams)
- Prior win rate (home/away, diff, sum)
- Quarter score differentials (Q1, Q2, Q3)

### Graph Features (from V4-V6)
- Graph area (home/away/diff)
- Peak values
- Swings count
- Slope 3m/5m
- **NEW**: Acceleration (change of slope)

### Play-by-Play Features (from V4-V9)
- Points per play
- 3PT rate
- Efficiency metrics
- **NEW**: Scoring runs (current/max/diff)
- **NEW**: Clutch window (last 6 min)

### Pressure Features (from V4)
- Score differential
- Trailing team context
- Required points per minute
- Urgency index
- Pressure ratio

### Monte Carlo (from V6)
- Simulated win probability
- 1000 simulations with variance

### **NEW**: League Context Features
- League home advantage
- League average total points
- League std deviation
- League sample size

## Risk Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| MIN_CONFIDENCE_TO_BET | 0.65 | Minimum 65% confidence |
| MIN_EDGE_FOR_OVER_UNDER | 4.0 | Need 4+ point edge |
| MAX_VOLATILITY | 0.70 | Block high volatility games |
| MIN_GRAPH_POINTS_Q3 | 20 | Min graph points for Q3 |
| MIN_GRAPH_POINTS_Q4 | 28 | Min graph points for Q4 |
| MIN_PBP_EVENTS_Q3 | 18 | Min PBP events for Q3 |
| MIN_PBP_EVENTS_Q4 | 26 | Min PBP events for Q4 |
| LOSS_PENALTY_MULTIPLIER | 2.0 | Loss costs 2x more than win |
| MIN_HIT_RATE_FOR_LEAGUE | 0.52 | 52% minimum for league |
| MIN_SAMPLES_FOR_LEAGUE | 50 | Min samples to judge league |

## Algorithm Comparison

| Version | Algorithms | Key Feature | Ensemble |
|---------|-----------|-------------|----------|
| V1-V2 | LogReg, RF, GB | Baseline + buckets | Average |
| V5 | XGB, HistGB, MLP | Non-linear only | Average |
| V6 | XGB, HistGB, MLP | Monte Carlo | Average |
| V7 | CatBoost, XGB(cat) | Native categoricals | - |
| V8 | PyTorch LSTM | Deep learning on graph | - |
| V9 | LogReg, GB | Fast + momentum | Weighted |
| V10 | Ridge, GB, XGB | Regression + stacking | Stacking |
| V11 | Ridge, GB, XGB | Gender-separated | Weighted |
| **V12** | **LogReg, GB, XGB, LGB, Cat** | **Hybrid + risk mgmt** | **Weighted** |

## Performance Expectations

### Classification
- Target accuracy: 65-70% on test set
- Target hit rate (bets only): 60-65%
- Target ROI: +0.05 to +0.15 per bet

### Regression
- Target MAE (total points): 3-5 points
- Target R²: 0.3-0.5
- Target edge detection: 70%+ accuracy when edge ≥ 4 pts

### Risk Management
- Coverage (bets/valid): 20-30% (very conservative)
- Loss rate: < 45%
- Profit factor: > 1.2

## Dependencies

Required:
- scikit-learn
- numpy
- joblib

Optional (improves performance):
- xgboost
- lightgbm
- catboost

Install all:
```bash
pip install scikit-learn numpy joblib xgboost lightgbm catboost
```

## Troubleshooting

### "Model not found" error
Run training first: `python training/v12/v12_cli.py train`

### "League stats not found" error
Generated during training. Run training first.

### Poor prediction quality
Check data quality - need sufficient graph points and PBP events.

### Models too conservative
Adjust thresholds in `infer_match_v12.py`:
- Lower `MIN_CONFIDENCE_TO_BET` (e.g., 0.60)
- Lower `MIN_EDGE_FOR_OVER_UNDER` (e.g., 3.0)
- Raise `MAX_VOLATILITY` (e.g., 0.75)

**Warning**: This increases risk!

## Notes

- V12 is **intentionally conservative** - expects to bet on only 20-30% of matches
- Quality over quantity - only bet when very confident
- League filtering is crucial - some leagues are unpredictable, others are not
- Always use sportsbook lines for over/under when available
- The 2x loss penalty makes the model avoid marginal bets

## Future Improvements

1. **Live updating** - Update models as new matches complete
2. **Odds integration** - Fetch real-time sportsbook lines
3. **Player injuries** - Incorporate lineup changes
4. **Rest days** - Factor in back-to-back games
5. **Home/away splits** - Some teams perform differently home vs away
6. **Streak features** - Current win/loss streak context
7. **Playoff vs regular** - Different models for different contexts
