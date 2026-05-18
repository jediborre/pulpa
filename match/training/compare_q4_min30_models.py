"""Compare Q4 models at snapshot minute 30 side-by-side.

Models compared:
- m30_v1 (independent, global, all leagues)
- v6.3 m30 (legacy, 10m-only snapshot 30)

Both run on the same test set with Kelly betting simulation.
"""

from __future__ import annotations

import sqlite3
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import train_q3_q4_models_v6 as v6
import train_q4_m30_v1 as m30_v1_train
import train_q4_models_v6_3 as v63_train
import infer_match as infer_live

BASE_M30_V1 = ROOT / "match" / "training" / "model_outputs_m30_v1"
BASE_V63 = ROOT / "match" / "training" / "model_outputs_v6_3"
BASE_M_V1 = ROOT / "match" / "training" / "model_outputs_m_v1"
OUT_PATH = BASE_M_V1 / "comparison_min30_q4.csv"

M30_V1_CHAMPION_PATH = BASE_M30_V1 / "q4_m30_v1_champion.joblib"
V63_M30_CHAMPION_PATH = BASE_V63 / "q4_m30_champion.joblib"

ODDS = 1.4
BREAK_EVEN = 1.0 / ODDS
BANK_START = 1000.0
KELLY_MULT = 0.25
KELLY_CAP = 0.05
MIN_CONF_PROB = 0.58
STAKE_STEP = 25.0
MIN_STAKE = 25.0
MAX_STAKE = 100.0
STAKE_BUCKETS = (25.0, 50.0, 75.0, 100.0)


def _quarter_points(data, q):
    return infer_live._quarter_points(data, q)


def _score_upto(data, minute):
    return infer_live._score_upto(data, minute)


def _clip_pbp_to_minute_m30(match_data, cutoff_minute):
    pbp = match_data.get("play_by_play") or {}
    quarter_minutes = m30_v1_train._infer_regulation_quarter_minutes(match_data)
    out = {"Q1": [], "Q2": [], "Q3": []}
    cutoff = float(cutoff_minute)
    for quarter_label, plays in pbp.items():
        q_idx = infer_live._quarter_index(str(quarter_label))
        if q_idx is None or q_idx < 1 or q_idx > 3:
            continue
        q_start = (q_idx - 1) * quarter_minutes
        kept = []
        for play in plays or []:
            rem_sec = infer_live._clock_to_seconds(str(play.get("time", "") or ""))
            if rem_sec is None or rem_sec > int(quarter_minutes * 60.0):
                continue
            global_min = q_start + (quarter_minutes - rem_sec / 60.0)
            if global_min <= cutoff + 1e-9:
                kept.append(play)
        out[f"Q{q_idx}"] = kept
    return out


def _build_live_like_q4_snapshot_m30(match_data, snapshot_minute):
    clipped = dict(match_data)
    clipped["graph_points"] = [
        point for point in (match_data.get("graph_points") or [])
        if int(point.get("minute", 0)) <= int(snapshot_minute)
    ]
    clipped["play_by_play"] = _clip_pbp_to_minute_m30(match_data, snapshot_minute)
    score_obj = dict(match_data.get("score") or {})
    quarters = dict(score_obj.get("quarters") or {})
    q1h, q1a = _quarter_points(match_data, "Q1")
    q2h, q2a = _quarter_points(match_data, "Q2")
    ht_home = int(q1h or 0) + int(q2h or 0)
    ht_away = int(q1a or 0) + int(q2a or 0)
    game_home, game_away = m30_v1_train._score_upto_m30(clipped, float(snapshot_minute))
    q3_home = max(0, int(game_home) - ht_home)
    q3_away = max(0, int(game_away) - ht_away)
    quarters["Q3"] = {"home": q3_home, "away": q3_away}
    score_obj["quarters"] = quarters
    clipped["score"] = score_obj
    return clipped


def _kelly_fraction(p, odds):
    b = odds - 1.0
    q = 1.0 - p
    return ((b * p) - q) / b if b > 0 else 0.0


def _round_down_step(x, step):
    return (x // step) * step if step > 0 else x


def _stake_from_signal(p_pick, edge, k_used, kelly_cap=0.05):
    if k_used is None or k_used <= 0:
        return 0.0
    denom = kelly_cap if kelly_cap and kelly_cap > 0 else 0.05
    k_strength = max(0.0, min(k_used / denom, 1.0))
    if p_pick >= 0.97 and edge >= 0.24 and k_strength >= 0.995:
        return STAKE_BUCKETS[3]
    if p_pick >= 0.90 and edge >= 0.18 and k_strength >= 0.85:
        return STAKE_BUCKETS[2]
    if p_pick >= 0.80 and edge >= 0.09 and k_strength >= 0.55:
        return STAKE_BUCKETS[1]
    if k_strength < 0.15:
        return 0.0
    return STAKE_BUCKETS[0]


def _bet_candidate(p_home):
    if p_home is None:
        return {"accept": False, "reason": "missing_prob"}
    p_pick = p_home if p_home >= 0.5 else (1.0 - p_home)
    edge = p_pick - BREAK_EVEN
    if p_pick < MIN_CONF_PROB:
        return {"accept": False, "reason": "confidence_filter"}
    k_raw = _kelly_fraction(p_pick, ODDS)
    if k_raw <= 0:
        return {"accept": False, "reason": "no_edge"}
    k_used = min(k_raw * KELLY_MULT, KELLY_CAP)
    stake = _stake_from_signal(p_pick, edge, k_used, KELLY_CAP)
    stake = min(stake, MAX_STAKE)
    stake = _round_down_step(stake, STAKE_STEP)
    if stake < MIN_STAKE:
        return {"accept": False, "reason": "stake_below_25"}
    return {"accept": True, "reason": "accepted"}


def _load_test_set():
    probe_sql = """
        SELECT m.date, m.time, m.match_id
        FROM matches m
        WHERE EXISTS (SELECT 1 FROM quarter_scores WHERE match_id = m.match_id AND quarter = 'Q4')
          AND EXISTS (SELECT 1 FROM play_by_play WHERE match_id = m.match_id AND quarter = 'Q1')
          AND (SELECT COUNT(*) FROM graph_points WHERE match_id = m.match_id) >= 20
        ORDER BY m.date, m.time, m.match_id
    """
    conn = sqlite3.connect(str(v6.DB_PATH))
    conn.row_factory = sqlite3.Row
    probe_rows = conn.execute(probe_sql).fetchall()
    conn.close()

    n_total = len(probe_rows)
    n_train = int(n_total * 0.70)
    n_val = int(n_total * 0.15)
    val_start_date = str(probe_rows[n_train]["date"])

    samples = v6._build_samples(v6.DB_PATH, date_gte=val_start_date)
    rows = sorted(
        [s for s in samples if s.target_q4 is not None],
        key=lambda s: s.dt,
    )
    test_rows = rows[n_val:]
    print(f"[TEST_SET] total_samples={n_total} test_rows={len(test_rows)}")
    return test_rows


def _load_data_map(test_rows):
    conn = v6.db_mod.get_conn(str(v6.DB_PATH))
    v6.db_mod.init_db(conn)
    data_map = {}
    for sample in tqdm(test_rows, desc="loading match data"):
        data_map[str(sample.match_id)] = v6.db_mod.get_match(conn, str(sample.match_id))
    conn.close()
    return data_map


def _predict_m30_v1(test_rows, data_map, champion_path):
    artifact = joblib.load(champion_path)
    vectorizer = artifact["vectorizer"]
    models = artifact["models"]
    snap = int(artifact.get("snapshot_minute", 30))
    probs = np.zeros(len(test_rows))
    y_true = np.zeros(len(test_rows), dtype=int)

    for i, sample in enumerate(tqdm(test_rows, desc="m30_v1")):
        md = data_map.get(str(sample.match_id))
        y_true[i] = int(sample.target_q4 or 0)
        if not md:
            probs[i] = 0.5
            continue
        live_like = _build_live_like_q4_snapshot_m30(md, snap)
        feat = m30_v1_train._build_m30_v1_features(sample, live_like)
        x_row = vectorizer.transform([feat])
        p_xgb = float(models["xgb"].predict_proba(x_row)[0, 1])
        p_hist = float(models["hist_gb"].predict_proba(x_row)[0, 1])
        probs[i] = (p_xgb + p_hist) / 2.0

    return probs, y_true


def _predict_v63_m30(test_rows, data_map, champion_path):
    artifact = joblib.load(champion_path)
    vectorizer = artifact["vectorizer"]
    models = artifact["models"]
    snap = int(artifact.get("snapshot_minute", 30))
    probs = np.zeros(len(test_rows))
    y_true = np.zeros(len(test_rows), dtype=int)

    for i, sample in enumerate(tqdm(test_rows, desc="v6.3_m30")):
        md = data_map.get(str(sample.match_id))
        y_true[i] = int(sample.target_q4 or 0)
        if not md:
            probs[i] = 0.5
            continue
        feat = v63_train._build_q4_features_at_window(sample, md, snap)
        x_row = vectorizer.transform([feat])
        p_xgb = float(models["xgb"].predict_proba(x_row)[0, 1])
        p_hist = float(models["hist_gb"].predict_proba(x_row)[0, 1])
        probs[i] = (p_xgb + p_hist) / 2.0

    return probs, y_true


def _compute_metrics(y_true, probs, label):
    preds = (probs >= 0.5).astype(int)
    return {
        "modelo": label,
        "test_samples": len(y_true),
        "accuracy": round(accuracy_score(y_true, preds), 4),
        "f1": round(f1_score(y_true, preds, zero_division=0), 4),
        "precision": round(precision_score(y_true, preds, zero_division=0), 4),
        "recall": round(recall_score(y_true, preds, zero_division=0), 4),
        "brier": round(brier_score_loss(y_true, probs), 4),
        "roc_auc": round(roc_auc_score(y_true, probs), 4),
    }


def _simulate_betting(test_rows, probs, label):
    bank = BANK_START
    bets = 0
    wins = 0
    total_staked = 0.0
    peak = BANK_START
    max_dd = 0.0
    no_bet = 0

    for i, sample in enumerate(test_rows):
        p_home = float(probs[i])
        y_true = int(sample.target_q4 or 0)
        candidate = _bet_candidate(p_home)
        if not candidate["accept"]:
            no_bet += 1
            continue
        p_pick = p_home if p_home >= 0.5 else (1.0 - p_home)
        stake = candidate.get("stake", 25.0)
        side_home = p_home >= 0.5
        is_win = (side_home and y_true == 1) or (not side_home and y_true == 0)
        pnl = stake * (ODDS - 1.0) if is_win else -stake
        bank += pnl
        bets += 1
        wins += int(is_win)
        total_staked += stake
        peak = max(peak, bank)
        dd = (peak - bank) / peak if peak > 0 else 0.0
        max_dd = max(max_dd, dd)

    return {
        "modelo": label,
        "test_samples": len(test_rows),
        "apuestas": bets,
        "sin_apuesta": no_bet,
        "ganadas": wins,
        "efectividad": round(wins / bets, 4) if bets else 0.0,
        "banco_final": round(bank, 2),
        "ganancia": round(bank - BANK_START, 2),
        "roi_bank": round((bank - BANK_START) / BANK_START, 4),
        "total_apostado": round(total_staked, 2),
        "yield": round((bank - BANK_START) / total_staked, 4) if total_staked > 0 else 0.0,
        "max_drawdown": round(max_dd, 4),
    }


def main():
    print("=" * 60)
    print("Comparacion Q4 snapshot minute 30")
    print("=" * 60)

    print("\n[1/4] Cargando test set...")
    test_rows = _load_test_set()

    print("\n[2/4] Cargando match data...")
    data_map = _load_data_map(test_rows)

    print("\n[3/4] Predicciones...")
    probs_m30, y_true = _predict_m30_v1(test_rows, data_map, M30_V1_CHAMPION_PATH)
    probs_v63, _ = _predict_v63_m30(test_rows, data_map, V63_M30_CHAMPION_PATH)

    print("\n[4/4] Resultados...")
    metrics = [
        _compute_metrics(y_true, probs_m30, "m30_v1"),
        _compute_metrics(y_true, probs_v63, "v6.3_m30"),
    ]
    metrics_df = pd.DataFrame(metrics)
    print("\n--- METRICAS ---")
    print(metrics_df.to_string(index=False))

    roi = [
        _simulate_betting(test_rows, probs_m30, "m30_v1"),
        _simulate_betting(test_rows, probs_v63, "v6.3_m30"),
    ]
    roi_df = pd.DataFrame(roi)
    print("\n--- SIMULACION APUESTAS (Kelly, odds 1.4) ---")
    print(roi_df.to_string(index=False))

    BASE_M_V1.mkdir(parents=True, exist_ok=True)
    combined = pd.concat([metrics_df, roi_df.drop(columns="test_samples")], axis=1)
    combined.to_csv(OUT_PATH, index=False)
    print(f"\nResultados guardados en: {OUT_PATH}")
    print("Done.")


if __name__ == "__main__":
    main()
