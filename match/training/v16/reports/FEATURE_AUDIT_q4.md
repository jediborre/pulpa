# Feature Audit v16 - target=q4

- Muestras analizadas: **4381**
- Features totales: **64**
- Umbral de redundancia Pearson: **|r| >= 0.90**
- Generado: 2026-04-17 14:40:32

## 1. Features con varianza despreciable
Std < 1e-4 o >99% valores iguales. Candidatas a drop directo.

| feature | std | zero_frac |
|---|---:|---:|
| `meta_target_is_q4` | 0.00000 | 0.00% |

## 2. Correlacion con target_winner (top 30 absolutas)

| feature | corr(y) | top_feats_freq |
|---|---:|---:|
| `traj_last10_diff` | -0.169 | 24 |
| `tfm_trend_slope` | -0.151 | 15 |
| `gp_slope_3m` | +0.099 | 0 |
| `gp_slope_5m` | +0.092 | 3 |
| `pbp_home_pts` | +0.089 | 21 |
| `tfm_margin` | -0.086 | 13 |
| `tfm_winner_pick` | -0.085 | 0 |
| `pbp_away_pts` | -0.074 | 20 |
| `gp_sign_changes` | +0.049 | 10 |
| `gp_acceleration` | +0.046 | 0 |
| `league_ht_total_std` | +0.045 | 18 |
| `tfm_current_trend` | -0.043 | 0 |
| `pace_bucket_low` | -0.043 | 1 |
| `traj_largest_lead_home` | +0.041 | 22 |
| `traj_largest_lead_away` | -0.033 | 15 |
| `traj_lead_changes` | +0.032 | 13 |
| `traj_times_tied` | +0.031 | 12 |
| `pace_bucket_medium` | +0.028 | 6 |
| `score_cumulative_diff` | -0.027 | 14 |
| `traj_current_run_home` | +0.025 | 13 |
| `score_halftime_diff` | -0.020 | 11 |
| `score_q2_diff` | -0.020 | 15 |
| `score_q3_diff` | -0.019 | 15 |
| `league_samples` | +0.018 | 16 |
| `score_q2_total` | +0.017 | 23 |
| `traj_score_diff_end` | -0.016 | 21 |
| `traj_current_run_away` | +0.016 | 8 |
| `gp_latest_diff` | +0.015 | 1 |
| `score_halftime_diff_ratio` | -0.014 | 18 |
| `traj_last5_home_pts` | +0.014 | 7 |

## 3. Clusters redundantes y sugerencia de drop
Cada cluster comparte |r| >= 0.90. Se conserva la
feature con mayor |corr(y)|. Las tfm_* estan protegidas por diseno.

### Cluster 1 (n=4)
- **Mantener**: `score_halftime_total`  (|r_y|=0.010)
- **Drop**:
  - `pace_ratio_vs_median`  (|r_y|=0.009)
  - `pace_total_prior`  (|r_y|=0.009)
  - `score_cumulative_total`  (|r_y|=0.009)

### Cluster 2 (n=3)
- **Mantener**: `score_halftime_diff`  (|r_y|=0.020)
- **Drop**:
  - `score_halftime_diff_ratio`  (|r_y|=0.014)
  - `traj_score_diff_end`  (|r_y|=0.016)

### Cluster 3 (n=3)
- **Mantener**: `league_ht_total_mean`  (|r_y|=0.004)
- **Drop**:
  - `league_q3_total_mean`  (|r_y|=0.002)
  - `league_q4_total_mean`  (|r_y|=0.002)

### Cluster 4 (n=2)
- **Mantener**: `gp_stddev`  (|r_y|=0.004)
- **Drop**:
  - `gp_amplitude`  (|r_y|=0.004)

### Cluster 5 (n=2)
- **Mantener**: `traj_largest_lead_away`  (|r_y|=0.033)
- **Drop**:
  - `gp_valley`  (|r_y|=0.002)

### Cluster 6 (n=2)
- **Mantener**: `score_cumulative_diff`  (|r_y|=0.027)
- **Drop**:
  - `gp_latest_diff`  (|r_y|=0.015)

### Cluster 7 (n=2)
- **Mantener**: `meta_minutes_to_quarter_end`  (|r_y|=0.012)
- **Drop**:
  - `meta_snapshot_minute`  (|r_y|=0.010)


## 4. Features propuestas para drop (union)
- Por redundancia: **11** features
- Por varianza nula: **1** features
- **Total distinto**: **12**

```python
FEATURE_BLACKLIST = [
    'gp_amplitude',
    'gp_latest_diff',
    'gp_valley',
    'league_q3_total_mean',
    'league_q4_total_mean',
    'meta_snapshot_minute',
    'meta_target_is_q4',
    'pace_ratio_vs_median',
    'pace_total_prior',
    'score_cumulative_total',
    'score_halftime_diff_ratio',
    'traj_score_diff_end',
]
```
