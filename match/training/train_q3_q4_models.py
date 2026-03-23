"""Train multi-model predictors for quarter winners (Q3, Q4)."""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import joblib
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import db as db_mod

DB_PATH = ROOT / "matches.db"
OUT_DIR = ROOT / "training" / "model_outputs"


@dataclass
class MatchSample:
    match_id: str
    dt: datetime
    features_q3: dict
    target_q3: int | None
    features_q4: dict
    target_q4: int | None


def _quarter_points(data: dict, quarter: str) -> tuple[int | None, int | None]:
    q = data.get("score", {}).get("quarters", {}).get(quarter)
    if not q:
        return None, None
    return int(q.get("home", 0)), int(q.get("away", 0))


def _safe_rate(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def _count_sign_swings(values: list[int]) -> int:
    swings = 0
    prev_sign = 0
    for v in values:
        sign = 1 if v > 0 else (-1 if v < 0 else 0)
        if sign == 0:
            continue
        if prev_sign != 0 and sign != prev_sign:
            swings += 1
        prev_sign = sign
    return swings


def _infer_gender(league: str, home_team: str, away_team: str) -> str:
    text = f"{league} {home_team} {away_team}".lower()
    markers = [
        "women", "woman", "female", "femen", "fem.", " ladies ",
        "(w)", " w ", "wnba", "girls",
    ]
    for marker in markers:
        if marker in text:
            return "women"
    return "men_or_open"


def _graph_stats_upto(graph_points: list[dict], max_minute: int) -> dict:
    points = [p for p in graph_points if int(p.get("minute", 0)) <= max_minute]
    values = [int(p.get("value", 0)) for p in points]
    if not values:
        return {
            "gp_count": 0,
            "gp_last": 0,
            "gp_peak_home": 0,
            "gp_peak_away": 0,
            "gp_area_home": 0,
            "gp_area_away": 0,
            "gp_area_diff": 0,
            "gp_mean_abs": 0.0,
            "gp_swings": 0,
        }

    area_home = sum(max(v, 0) for v in values)
    area_away = sum(max(-v, 0) for v in values)
    mean_abs = sum(abs(v) for v in values) / len(values)
    return {
        "gp_count": len(values),
        "gp_last": values[-1],
        "gp_peak_home": max(values),
        "gp_peak_away": abs(min(values)),
        "gp_area_home": area_home,
        "gp_area_away": area_away,
        "gp_area_diff": area_home - area_away,
        "gp_mean_abs": mean_abs,
        "gp_swings": _count_sign_swings(values),
    }


def _pbp_stats_upto(pbp: dict, quarters: list[str]) -> dict:
    home_plays = 0
    away_plays = 0
    home_3pt = 0
    away_3pt = 0
    for q in quarters:
        for play in pbp.get(q, []):
            team = play.get("team")
            pts = int(play.get("points", 0))
            if team == "home":
                home_plays += 1
                if pts == 3:
                    home_3pt += 1
            elif team == "away":
                away_plays += 1
                if pts == 3:
                    away_3pt += 1

    return {
        "pbp_home_plays": home_plays,
        "pbp_away_plays": away_plays,
        "pbp_plays_diff": home_plays - away_plays,
        "pbp_home_3pt": home_3pt,
        "pbp_away_3pt": away_3pt,
        "pbp_3pt_diff": home_3pt - away_3pt,
    }


def _is_complete_match(data: dict) -> bool:
    quarters = data.get("score", {}).get("quarters", {})
    required = {"Q1", "Q2", "Q3", "Q4"}
    if not required.issubset(quarters.keys()):
        return False
    if not data.get("graph_points"):
        return False
    if not data.get("play_by_play"):
        return False
    return True


def _build_samples(db_path: Path) -> list[MatchSample]:
    conn = db_mod.get_conn(str(db_path))
    db_mod.init_db(conn)

    rows = conn.execute(
        "SELECT match_id, date, time "
        "FROM matches ORDER BY date, time, match_id"
    ).fetchall()

    team_history: dict[str, list[int]] = defaultdict(list)
    samples: list[MatchSample] = []

    for row in rows:
        match_id = str(row["match_id"])
        dt = datetime.strptime(
            f"{row['date']} {row['time']}",
            "%Y-%m-%d %H:%M",
        )
        data = db_mod.get_match(conn, match_id)
        if data is None or not _is_complete_match(data):
            continue

        m = data["match"]
        s = data["score"]
        pbp = data.get("play_by_play", {})
        gp = data.get("graph_points", [])

        q1h, q1a = _quarter_points(data, "Q1")
        q2h, q2a = _quarter_points(data, "Q2")
        q3h, q3a = _quarter_points(data, "Q3")
        q4h, q4a = _quarter_points(data, "Q4")
        if None in (q1h, q1a, q2h, q2a, q3h, q3a, q4h, q4a):
            continue

        home_hist = team_history[m["home_team"]][-10:]
        away_hist = team_history[m["away_team"]][-10:]
        home_prior_wr = _safe_rate(sum(home_hist), len(home_hist))
        away_prior_wr = _safe_rate(sum(away_hist), len(away_hist))

        base = {
            "league": m.get("league", ""),
            "gender_bucket": _infer_gender(
                m.get("league", ""),
                m.get("home_team", ""),
                m.get("away_team", ""),
            ),
            "home_prior_wr": home_prior_wr,
            "away_prior_wr": away_prior_wr,
            "prior_wr_diff": home_prior_wr - away_prior_wr,
            "q1_diff": q1h - q1a,
            "q2_diff": q2h - q2a,
        }

        ht_home = q1h + q2h
        ht_away = q1a + q2a

        f_q3 = dict(base)
        f_q3.update({
            "ht_home": ht_home,
            "ht_away": ht_away,
            "ht_diff": ht_home - ht_away,
        })
        f_q3.update(_graph_stats_upto(gp, 24))
        f_q3.update(_pbp_stats_upto(pbp, ["Q1", "Q2"]))

        f_q4 = dict(base)
        f_q4.update({
            "q3_diff": q3h - q3a,
            "score_3q_home": ht_home + q3h,
            "score_3q_away": ht_away + q3a,
            "score_3q_diff": (ht_home + q3h) - (ht_away + q3a),
        })
        f_q4.update(_graph_stats_upto(gp, 36))
        f_q4.update(_pbp_stats_upto(pbp, ["Q1", "Q2", "Q3"]))

        t_q3 = None if q3h == q3a else int(q3h > q3a)
        t_q4 = None if q4h == q4a else int(q4h > q4a)

        samples.append(
            MatchSample(
                match_id=match_id,
                dt=dt,
                features_q3=f_q3,
                target_q3=t_q3,
                features_q4=f_q4,
                target_q4=t_q4,
            )
        )

        home_win = int(s["home"] > s["away"])
        away_win = 1 - home_win
        team_history[m["home_team"]].append(home_win)
        team_history[m["away_team"]].append(away_win)

    conn.close()
    return samples


def _write_dataset_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as f:
            f.write("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _train_for_target(
    samples: list[MatchSample],
    target_name: str,
    model_dir: Path,
) -> dict:
    if target_name == "q3":
        rows = [s for s in samples if s.target_q3 is not None]
        rows = sorted(rows, key=lambda x: x.dt)
        x_dict = [s.features_q3 for s in rows]
        y = [int(s.target_q3) for s in rows]
    else:
        rows = [s for s in samples if s.target_q4 is not None]
        rows = sorted(rows, key=lambda x: x.dt)
        x_dict = [s.features_q4 for s in rows]
        y = [int(s.target_q4) for s in rows]

    n = len(rows)
    if n < 200:
        raise RuntimeError(
            f"Not enough rows for {target_name}. Need at least 200, got {n}."
        )

    split = int(n * 0.8)
    if split <= 0 or split >= n:
        raise RuntimeError(
            f"Invalid train/test split for {target_name}: n={n}"
        )

    vec = DictVectorizer(sparse=False)
    x_all = vec.fit_transform(x_dict)
    x_train, x_test = x_all[:split], x_all[split:]
    y_train, y_test = y[:split], y[split:]

    models = {
        "logreg": LogisticRegression(
            solver="liblinear",
            max_iter=4000,
        ),
        "rf": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=4,
            random_state=42,
            n_jobs=-1,
        ),
        "gb": GradientBoostingClassifier(
            n_estimators=250,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
        ),
    }

    metrics_rows: list[dict] = []
    proba_map: dict[str, list[float]] = {}

    model_dir.mkdir(parents=True, exist_ok=True)

    for name, model in models.items():
        model.fit(x_train, y_train)
        probs = model.predict_proba(x_test)[:, 1]
        preds = [1 if p >= 0.5 else 0 for p in probs]

        acc = accuracy_score(y_test, preds)
        ll = log_loss(y_test, probs)
        brier = brier_score_loss(y_test, probs)

        metrics_rows.append(
            {
                "target": target_name,
                "model": name,
                "samples_total": n,
                "samples_train": split,
                "samples_test": n - split,
                "accuracy": round(float(acc), 6),
                "log_loss": round(float(ll), 6),
                "brier": round(float(brier), 6),
            }
        )
        proba_map[name] = list(probs)

        artifact = {
            "vectorizer": vec,
            "model": model,
            "target": target_name,
            "trained_rows": n,
            "feature_count": len(vec.feature_names_),
        }
        joblib.dump(artifact, model_dir / f"{target_name}_{name}.joblib")

    ensemble_probs = []
    for i in range(len(y_test)):
        vals = [proba_map[name][i] for name in models.keys()]
        ensemble_probs.append(sum(vals) / len(vals))

    ensemble_preds = [1 if p >= 0.5 else 0 for p in ensemble_probs]
    ens_acc = accuracy_score(y_test, ensemble_preds)
    ens_ll = log_loss(y_test, ensemble_probs)
    ens_brier = brier_score_loss(y_test, ensemble_probs)

    metrics_rows.append(
        {
            "target": target_name,
            "model": "ensemble_avg_prob",
            "samples_total": n,
            "samples_train": split,
            "samples_test": n - split,
            "accuracy": round(float(ens_acc), 6),
            "log_loss": round(float(ens_ll), 6),
            "brier": round(float(ens_brier), 6),
        }
    )

    agreement = sum(
        1
        for i in range(len(y_test))
        if len(
            {
                int(proba_map["logreg"][i] >= 0.5),
                int(proba_map["rf"][i] >= 0.5),
                int(proba_map["gb"][i] >= 0.5),
            }
        )
        == 1
    )

    consensus = {
        "target": target_name,
        "n_test": len(y_test),
        "agreement_rate_all_models": round(agreement / len(y_test), 6),
    }

    return {
        "metrics_rows": metrics_rows,
        "consensus": consensus,
        "dataset_rows": n,
    }


def _write_metrics_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    samples = _build_samples(DB_PATH)

    q3_export = []
    q4_export = []
    for sample in samples:
        if sample.target_q3 is not None:
            row = {
                "match_id": sample.match_id,
                "datetime": sample.dt.isoformat(),
            }
            row.update(sample.features_q3)
            row["target_q3_home_win"] = sample.target_q3
            q3_export.append(row)

        if sample.target_q4 is not None:
            row = {
                "match_id": sample.match_id,
                "datetime": sample.dt.isoformat(),
            }
            row.update(sample.features_q4)
            row["target_q4_home_win"] = sample.target_q4
            q4_export.append(row)

    _write_dataset_csv(OUT_DIR / "q3_dataset.csv", q3_export)
    _write_dataset_csv(OUT_DIR / "q4_dataset.csv", q4_export)

    q3_result = _train_for_target(samples, "q3", OUT_DIR)
    q4_result = _train_for_target(samples, "q4", OUT_DIR)

    _write_metrics_csv(OUT_DIR / "q3_metrics.csv", q3_result["metrics_rows"])
    _write_metrics_csv(OUT_DIR / "q4_metrics.csv", q4_result["metrics_rows"])

    with (OUT_DIR / "q3_consensus.json").open("w", encoding="utf-8") as f:
        json.dump(q3_result["consensus"], f, indent=2, ensure_ascii=False)
    with (OUT_DIR / "q4_consensus.json").open("w", encoding="utf-8") as f:
        json.dump(q4_result["consensus"], f, indent=2, ensure_ascii=False)

    print("[train] done")
    print(f"[train] samples_complete={len(samples)}")
    print(f"[train] q3_rows={q3_result['dataset_rows']}")
    print(f"[train] q4_rows={q4_result['dataset_rows']}")
    print(f"[train] outputs={OUT_DIR}")


if __name__ == "__main__":
    main()
