> **Ubicación original:** match/training/model_outputs_v6_2/V6_2_REPORT.md

---

# V6.2 Report

## Config
- league_min_train_rows: 30
- league_min_effect_abs_diff: 0.015
- league_other_token: LEAGUE_OTHER_SIGNAL_WEAK
- league_name_exclusion_config: C:\Users\App\Desktop\pulpa\match\training\v6_2_league_name_exclusions.json
- league_name_exclusion_patterns: 67
- league_name_exclusion_match_mode: contains(case-insensitive)
- split: temporal 80/20 (same as v6)
- models: xgb, hist_gb
- champion_q3: xgb
- champion_q4: 0.6*xgb + 0.4*hist_gb

## League Name Exclusion Summary
| target | rows_before | rows_excluded | rows_after | exclude_ratio |
|---|---:|---:|---:|---:|
| q3 | 21616 | 9216 | 12400 | 0.4264 |
| q4 | 21639 | 9240 | 12399 | 0.4270 |

## League Filter Summary
| target | kept_leagues | rows_replaced | replace_ratio | features_after_vectorizer |
|---|---:|---:|---:|---:|
| q3 | 73 | 3145 | 0.2536 | 390 |
| q4 | 68 | 5891 | 0.4751 | 381 |

## Holdout Metrics (V6.2)
| target | model | accuracy | f1 | log_loss | brier | roc_auc |
|---|---|---:|---:|---:|---:|---:|
| q3 | xgb | 0.702823 | 0.731707 | 0.556539 | 0.189368 | 0.781122 |
| q3 | hist_gb | 0.703226 | 0.731583 | 0.561291 | 0.191571 | 0.777202 |
| q3 | champion_q3_xgb | 0.702823 | 0.731707 | 0.556539 | 0.189368 | 0.781122 |
| q4 | xgb | 0.762097 | 0.777190 | 0.471528 | 0.156155 | 0.853516 |
| q4 | hist_gb | 0.762097 | 0.775836 | 0.470482 | 0.156152 | 0.853773 |
| q4 | champion_q4_blend_xgb_0.6_hist_0.4 | 0.763306 | 0.777736 | 0.469482 | 0.155603 | 0.854483 |

## A/B vs V6
| target | v6_ref | v6.2_model | d_acc | d_f1 | d_log_loss | d_brier | d_auc |
|---|---|---|---:|---:|---:|---:|---:|
| q3 | xgb | champion_q3_xgb | -0.013400 | -0.009113 | +0.013349 | +0.005527 | -0.013716 |
| q3 | hist_gb | hist_gb | -0.010659 | -0.007547 | +0.015890 | +0.006791 | -0.015257 |
| q3 | xgb | xgb | -0.013400 | -0.009113 | +0.013349 | +0.005527 | -0.013716 |
| q4 | ensemble_avg_prob | champion_q4_blend_xgb_0.6_hist_0.4 | +0.006126 | +0.004316 | -0.005583 | -0.002460 | +0.003090 |
| q4 | hist_gb | hist_gb | +0.008186 | +0.006673 | -0.015654 | -0.005805 | +0.011126 |
| q4 | xgb | xgb | +0.002116 | +0.002530 | -0.009423 | -0.003997 | +0.007168 |
