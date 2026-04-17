# FEATURE AUDIT v16 (combined)

Generado: 2026-04-17 14:40:32

## Resumen q3
- Samples: 3619
- Features: 64
- Clusters redundantes: 7
- Blacklist: 16

## Resumen q4
- Samples: 4381
- Features: 64
- Clusters redundantes: 7
- Blacklist: 12

## Blacklist global (union q3 + q4)

```python
FEATURE_BLACKLIST = [
    'gp_amplitude',
    'gp_latest_diff',
    'gp_valley',
    'league_q3_total_mean',
    'league_q4_total_mean',
    'meta_minutes_to_quarter_end',
    'meta_snapshot_minute',
    'meta_target_is_q4',
    'pace_ratio_vs_median',
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
    'traj_score_diff_end',
]
```

Detalle completo por target en `FEATURE_AUDIT_q3.md` y `FEATURE_AUDIT_q4.md`.
