# Feature Audit v16 - target=q3

- Muestras analizadas: **3619**
- Features totales: **64**
- Umbral de redundancia Pearson: **|r| >= 0.90**
- Generado: 2026-04-17 14:40:21

## 1. Features con varianza despreciable
Std < 1e-4 o >99% valores iguales. Candidatas a drop directo.

| feature | std | zero_frac |
|---|---:|---:|
| `meta_target_is_q4` | 0.00000 | 100.00% |
| `score_q3_diff` | 0.00000 | 100.00% |
| `score_q3_total` | 0.00000 | 100.00% |
| `score_q3_vs_ht_momentum` | 0.00000 | 100.00% |

## 2. Correlacion con target_winner (top 30 absolutas)

| feature | corr(y) | top_feats_freq |
|---|---:|---:|
| `gp_slope_3m` | +0.118 | 0 |
| `pbp_home_pts` | +0.110 | 21 |
| `traj_largest_lead_away` | -0.099 | 15 |
| `pbp_away_pts` | -0.092 | 20 |
| `gp_slope_5m` | +0.088 | 3 |
| `gp_latest_diff` | +0.072 | 1 |
| `traj_largest_lead_home` | +0.071 | 22 |
| `gp_valley` | +0.068 | 14 |
| `gp_acceleration` | +0.064 | 0 |
| `traj_score_diff_end` | +0.058 | 21 |
| `score_q1_diff` | +0.052 | 11 |
| `tfm_trend_slope` | -0.044 | 15 |
| `league_ht_total_std` | -0.039 | 18 |
| `traj_current_run_away` | -0.038 | 8 |
| `score_halftime_diff_ratio` | +0.037 | 18 |
| `score_cumulative_diff` | +0.037 | 14 |
| `score_halftime_diff` | +0.037 | 11 |
| `gp_peak` | +0.036 | 12 |
| `gp_stddev` | -0.036 | 15 |
| `gp_amplitude` | -0.032 | 11 |
| `tfm_current_trend` | +0.029 | 0 |
| `pbp_home_3pt_rate` | +0.029 | 12 |
| `traj_last10_diff` | -0.023 | 24 |
| `gp_last_sign` | +0.022 | 3 |
| `gp_swings` | +0.019 | 4 |
| `traj_last5_away_pts` | -0.019 | 9 |
| `pace_bucket_low` | -0.019 | 1 |
| `tfm_uncertainty` | +0.018 | 14 |
| `league_ht_total_mean` | -0.017 | 17 |
| `pbp_scoring_density` | -0.017 | 0 |

## 3. Clusters redundantes y sugerencia de drop
Cada cluster comparte |r| >= 0.90. Se conserva la
feature con mayor |corr(y)|. Las tfm_* estan protegidas por diseno.

### Cluster 1 (n=4)
- **Mantener**: `pace_ratio_vs_median`  (|r_y|=0.008)
- **Drop**:
  - `pace_total_prior`  (|r_y|=0.008)
  - `score_cumulative_total`  (|r_y|=0.008)
  - `score_halftime_total`  (|r_y|=0.008)

### Cluster 2 (n=4)
- **Mantener**: `gp_latest_diff`  (|r_y|=0.072)
- **Drop**:
  - `score_cumulative_diff`  (|r_y|=0.037)
  - `score_halftime_diff`  (|r_y|=0.037)
  - `score_halftime_diff_ratio`  (|r_y|=0.037)

### Cluster 3 (n=3)
- **Mantener**: `league_ht_total_mean`  (|r_y|=0.017)
- **Drop**:
  - `league_q3_total_mean`  (|r_y|=0.014)
  - `league_q4_total_mean`  (|r_y|=0.015)

### Cluster 4 (n=2)
- **Mantener**: `gp_stddev`  (|r_y|=0.036)
- **Drop**:
  - `gp_amplitude`  (|r_y|=0.032)

### Cluster 5 (n=2)
- **Mantener**: `meta_snapshot_minute`  (|r_y|=0.009)
- **Drop**:
  - `meta_minutes_to_quarter_end`  (|r_y|=0.005)

### Cluster 6 (n=2)
- **Mantener**: `traj_score_diff_end`  (|r_y|=0.058)
- **Drop**:
  - `score_q1_diff`  (|r_y|=0.052)

### Cluster 7 (n=2)
- **Mantener**: `traj_largest_lead_away`  (|r_y|=0.099)
- **Drop**:
  - `gp_valley`  (|r_y|=0.068)


## 4. Features propuestas para drop (union)
- Por redundancia: **12** features
- Por varianza nula: **4** features
- **Total distinto**: **16**

```python
FEATURE_BLACKLIST = [
    'gp_amplitude',
    'gp_valley',
    'league_q3_total_mean',
    'league_q4_total_mean',
    'meta_minutes_to_quarter_end',
    'meta_target_is_q4',
    'pace_total_prior',
    'score_cumulative_diff',
    'score_cumulative_total',
    'score_halftime_diff',
    'score_halftime_diff_ratio',
    'score_halftime_total',
    'score_q1_diff',
    'score_q3_diff',
    'score_q3_total',
    'score_q3_vs_ht_momentum',
]
```
