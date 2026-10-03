"""
feature_audit.py - Auditoria de features de V16.

Mejora #1 del ROADMAP (18-abr-2026). Objetivo:
- Detectar features redundantes (|corr Pearson| > umbral).
- Detectar features casi-constantes / con varianza despreciable.
- Rankear features por correlacion con target_winner (proxy de utilidad).
- Correr permutation importance opcional sobre una liga grande para validar.
- Producir `reports/FEATURE_AUDIT.md` con recomendaciones de drop.

Uso:
    python -m training.v16.feature_audit                 # full audit
    python -m training.v16.feature_audit --quick         # sin permutation
    python -m training.v16.feature_audit --corr 0.92     # umbral personalizado
    python -m training.v16.feature_audit --no-tfm        # ignora tfm_* en drops
    python -m training.v16.feature_audit --sample 15000  # muestrea N filas
    python -m training.v16.feature_audit --trained-only  # solo ligas entrenables

Requiere: numpy, pandas, scikit-learn, joblib.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from training.v16 import config, dataset as ds, features as feat


REPORTS_DIR = Path(__file__).parent / "reports"
MODEL_DIR = Path(__file__).parent / "model_outputs"
AUDIT_JSON = REPORTS_DIR / "feature_audit.json"
AUDIT_MD = REPORTS_DIR / "FEATURE_AUDIT.md"


# ==========================================================================
# Utilidades numericas
# ==========================================================================

def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() < 1e-9 or b.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _zero_frac(col: np.ndarray) -> float:
    return float((col == 0).sum()) / float(len(col)) if len(col) else 0.0


# ==========================================================================
# Pipeline de construccion de feature matrix (train + val)
# ==========================================================================

def _build_feature_matrix(samples: list[ds.Sample], verbose: bool = True):
    """Construye (X_dict_list, y_win, league_per_row) para una lista de samples.
    Usa el cache TimesFM/Chronos persistido en disco (si existe) y
    league_stats walk-forward real, igual que train.py."""
    if verbose:
        print(f"[audit] features para {len(samples)} muestras...")
    match_ids = sorted({s.match_id for s in samples})
    conn = ds.get_db_connection()
    try:
        league_stats_map = ds.compute_league_stats_walkforward(samples, conn=conn)
        gp = ds.load_graph_points(conn, match_ids)
        pbp = ds.load_pbp_events(conn, match_ids)
    finally:
        conn.close()

    pace_thresholds = ds.calculate_pace_thresholds(samples)

    dicts: list[dict[str, Any]] = []
    y_win: list[int] = []
    leagues: list[str] = []
    targets: list[str] = []
    t0 = time.time()
    for i, s in enumerate(samples):
        f = feat.build_features_for_sample(
            s, gp, pbp,
            league_stats=league_stats_map.get(s.match_id),
            pace_thresholds=pace_thresholds,
            tfm_cache=None,  # fallback al cache persistido de timesfm/chronos
        )
        dicts.append(f)
        y_win.append(int(s.target_winner) if s.target_winner is not None else 0)
        leagues.append(s.league)
        targets.append(s.target)
        if verbose and (i + 1) % 2000 == 0:
            print(f"  {i+1}/{len(samples)}  t={time.time()-t0:.0f}s")
    return dicts, np.array(y_win, dtype=int), leagues, targets


def _dicts_to_matrix(dicts: list[dict[str, Any]]) -> tuple[np.ndarray, list[str]]:
    # Union de todas las keys que hayan aparecido (estables en orden alfabetico)
    keys = sorted({k for d in dicts for k in d.keys()})
    X = np.zeros((len(dicts), len(keys)), dtype=np.float32)
    for i, d in enumerate(dicts):
        for j, k in enumerate(keys):
            v = d.get(k, 0.0)
            if isinstance(v, bool):
                v = 1.0 if v else 0.0
            try:
                X[i, j] = float(v)
            except Exception:
                X[i, j] = 0.0
    return X, keys


# ==========================================================================
# Analisis de redundancia
# ==========================================================================

def _corr_matrix(X: np.ndarray, names: list[str]) -> np.ndarray:
    # np.corrcoef pide shape (features, observations)
    std = X.std(axis=0)
    safe = std > 1e-9
    out = np.zeros((X.shape[1], X.shape[1]), dtype=np.float32)
    # Solo computamos sobre columnas con varianza
    idx = np.where(safe)[0]
    if len(idx) < 2:
        return out
    sub = X[:, idx]
    c = np.corrcoef(sub, rowvar=False)
    # volcar a matriz global
    for a_local, a_global in enumerate(idx):
        for b_local, b_global in enumerate(idx):
            out[a_global, b_global] = c[a_local, b_local]
    return out


def _find_redundant_pairs(
    corr: np.ndarray, names: list[str], threshold: float,
) -> list[tuple[str, str, float]]:
    pairs: list[tuple[str, str, float]] = []
    n = corr.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            r = float(corr[i, j])
            if abs(r) >= threshold:
                pairs.append((names[i], names[j], r))
    pairs.sort(key=lambda t: -abs(t[2]))
    return pairs


def _cluster_by_correlation(
    pairs: list[tuple[str, str, float]],
) -> list[list[str]]:
    # Union-find simple sobre nombres
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for a, b, _ in pairs:
        parent.setdefault(a, a)
        parent.setdefault(b, b)
        union(a, b)

    clusters: dict[str, list[str]] = defaultdict(list)
    for name in parent:
        clusters[find(name)].append(name)
    return [sorted(names) for names in clusters.values() if len(names) > 1]


def _pick_representative(
    cluster: list[str],
    corr_with_target: dict[str, float],
    freq_in_top_features: dict[str, int],
    is_tfm_protected: bool = True,
) -> tuple[str, list[str]]:
    """Elige la feature representante del cluster (mas correlacionada con target
    y ademas con mas apariciones en top_features). El resto se marca como drop.
    Si is_tfm_protected, las features tfm_* nunca se proponen para drop."""
    def score(n: str) -> tuple[float, int]:
        return (abs(corr_with_target.get(n, 0.0)), freq_in_top_features.get(n, 0))

    protected = [n for n in cluster if n.startswith("tfm_")] if is_tfm_protected else []
    if protected:
        keep = max(protected, key=score)
    else:
        keep = max(cluster, key=score)
    drops = [n for n in cluster if n != keep and (
        not is_tfm_protected or not n.startswith("tfm_")
    )]
    return keep, drops


# ==========================================================================
# Permutation importance opcional (sobre un modelo entrenado)
# ==========================================================================

def _permutation_importance_for_league(
    league_slug: str, target: str, samples_val: list[ds.Sample], dicts_val: list[dict],
    y_val: np.ndarray, n_repeats: int = 3,
) -> dict[str, float] | None:
    """Carga el clf + vectorizer vigentes de esa liga y corre permutation
    importance sobre el val set. Devuelve dict {feature_name: delta_accuracy}.
    None si no existe modelo."""
    clf_path = MODEL_DIR / f"league_{league_slug}_{target}_clf.joblib"
    vec_path = MODEL_DIR / f"league_{league_slug}_{target}_vectorizer.joblib"
    if not clf_path.exists() or not vec_path.exists():
        return None
    try:
        import joblib
        from sklearn.inspection import permutation_importance
        clf = joblib.load(clf_path)
        vec = joblib.load(vec_path)
        X_val = vec.transform(dicts_val)
        if hasattr(X_val, "toarray"):
            X_val = X_val.toarray()
        if X_val.shape[0] < 20:
            return None
        result = permutation_importance(
            clf, X_val, y_val,
            n_repeats=n_repeats, random_state=42, scoring="accuracy",
        )
        names = vec.get_feature_names_out()
        return {str(n): float(v) for n, v in zip(names, result.importances_mean)}
    except Exception as e:
        print(f"[audit] perm-imp fallo para {league_slug}/{target}: {e}")
        return None


# ==========================================================================
# Leer top_features del summary (frecuencia de aparicion)
# ==========================================================================

def _feature_frequency_in_top() -> dict[str, int]:
    summary_path = MODEL_DIR / "training_summary_v16.json"
    if not summary_path.exists():
        return {}
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    counter: dict[str, int] = defaultdict(int)
    for m in summary.get("models", []):
        for ft in m.get("top_features", []):
            name = ft if isinstance(ft, str) else (ft.get("name") or ft.get("feature"))
            if name:
                counter[name] += 1
    return dict(counter)


# ==========================================================================
# Report generation
# ==========================================================================

def _render_markdown(
    corr_threshold: float,
    target: str,
    n_samples: int,
    n_features: int,
    low_variance: list[tuple[str, float, float]],
    corr_with_target: dict[str, float],
    top_features_freq: dict[str, int],
    redundancy_clusters: list[tuple[str, list[str], list[str]]],
    perm_importance: dict[str, dict[str, float]] | None,
) -> str:
    lines: list[str] = []
    lines.append(f"# Feature Audit v16 - target={target}")
    lines.append("")
    lines.append(f"- Muestras analizadas: **{n_samples}**")
    lines.append(f"- Features totales: **{n_features}**")
    lines.append(f"- Umbral de redundancia Pearson: **|r| >= {corr_threshold:.2f}**")
    lines.append(f"- Generado: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    lines.append("## 1. Features con varianza despreciable")
    lines.append("Std < 1e-4 o >99% valores iguales. Candidatas a drop directo.")
    lines.append("")
    if low_variance:
        lines.append("| feature | std | zero_frac |")
        lines.append("|---|---:|---:|")
        for name, std, zfrac in low_variance:
            lines.append(f"| `{name}` | {std:.5f} | {zfrac:.2%} |")
    else:
        lines.append("_Ninguna detectada._")
    lines.append("")

    lines.append("## 2. Correlacion con target_winner (top 30 absolutas)")
    ranked = sorted(corr_with_target.items(), key=lambda kv: -abs(kv[1]))[:30]
    lines.append("")
    lines.append("| feature | corr(y) | top_feats_freq |")
    lines.append("|---|---:|---:|")
    for name, r in ranked:
        lines.append(f"| `{name}` | {r:+.3f} | {top_features_freq.get(name, 0)} |")
    lines.append("")

    lines.append("## 3. Clusters redundantes y sugerencia de drop")
    lines.append(f"Cada cluster comparte |r| >= {corr_threshold:.2f}. Se conserva la")
    lines.append("feature con mayor |corr(y)|. Las tfm_* estan protegidas por diseno.")
    lines.append("")
    if redundancy_clusters:
        for i, (rep, cluster, drops) in enumerate(redundancy_clusters, 1):
            lines.append(f"### Cluster {i} (n={len(cluster)})")
            lines.append(f"- **Mantener**: `{rep}`  (|r_y|={abs(corr_with_target.get(rep,0)):.3f})")
            if drops:
                lines.append(f"- **Drop**:")
                for d in drops:
                    lines.append(f"  - `{d}`  (|r_y|={abs(corr_with_target.get(d,0)):.3f})")
            else:
                lines.append("- **Drop**: _ninguno (cluster 100% protegido)_")
            lines.append("")
    else:
        lines.append("_Ningun cluster super el umbral._")
    lines.append("")

    lines.append("## 4. Features propuestas para drop (union)")
    all_drops = sorted({d for _, _, ds_list in redundancy_clusters for d in ds_list})
    constant_drops = [n for n, _, _ in low_variance]
    lines.append(f"- Por redundancia: **{len(all_drops)}** features")
    lines.append(f"- Por varianza nula: **{len(constant_drops)}** features")
    lines.append(f"- **Total distinto**: **{len(set(all_drops) | set(constant_drops))}**")
    lines.append("")
    lines.append("```python")
    lines.append("FEATURE_BLACKLIST = [")
    for n in sorted(set(all_drops) | set(constant_drops)):
        lines.append(f"    {n!r},")
    lines.append("]")
    lines.append("```")
    lines.append("")

    if perm_importance:
        lines.append("## 5. Permutation importance (ligas grandes)")
        lines.append("Delta accuracy al permutar la feature en val (n_repeats=3).")
        lines.append("Valores negativos = feature no aporta / mete ruido.")
        lines.append("")
        for lg, imp in perm_importance.items():
            lines.append(f"### {lg}")
            lines.append("| feature | delta_acc |")
            lines.append("|---|---:|")
            for name, v in sorted(imp.items(), key=lambda kv: kv[1])[:20]:
                lines.append(f"| `{name}` | {v:+.4f} |")
            lines.append("")

    return "\n".join(lines)


# ==========================================================================
# Main
# ==========================================================================

def _trained_leagues_from_summary() -> set[str]:
    summary_path = MODEL_DIR / "training_summary_v16.json"
    if not summary_path.exists():
        return set()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    out: set[str] = set()
    for m in summary.get("models", []):
        lg = m.get("league")
        if lg and (m.get("n_train") or 0) > 0:
            out.add(lg)
    return out


def run_audit(
    corr_threshold: float = 0.90,
    do_permutation: bool = True,
    protect_tfm: bool = True,
    sample_size: int | None = 15000,
    trained_only: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    print("[audit] cargando samples...")
    samples, meta = ds.build_samples(use_cache=True, verbose=verbose)
    splits = ds.split_temporal(samples, train_days=config.TRAIN_DAYS,
                               val_days=config.VAL_DAYS,
                               cal_days=config.CAL_DAYS,
                               holdout_days=0)
    # Audit usa train + val (NO cal ni holdout para no mirar el futuro)
    audit_samples = splits["train"] + splits["val"]
    print(f"[audit] train+val (todas las ligas) = {len(audit_samples)} muestras")

    if trained_only:
        trained_set = _trained_leagues_from_summary()
        if trained_set:
            before = len(audit_samples)
            audit_samples = [s for s in audit_samples if s.league in trained_set]
            splits["val"] = [s for s in splits["val"] if s.league in trained_set]
            print(f"[audit] filtrado a ligas entrenables "
                  f"({len(trained_set)} ligas): {before} -> {len(audit_samples)}")

    if sample_size and len(audit_samples) > sample_size:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(audit_samples), size=sample_size, replace=False)
        audit_samples = [audit_samples[i] for i in sorted(idx.tolist())]
        print(f"[audit] muestreo a {len(audit_samples)} muestras (seed=42)")

    # Separamos por target porque features difieren (q3 no tiene score_q3_*)
    results: dict[str, Any] = {}
    for target in ("q3", "q4"):
        tgt_samples = [s for s in audit_samples if s.target == target]
        val_samples = [s for s in splits["val"] if s.target == target]
        if not tgt_samples:
            continue
        print(f"\n[audit] === target={target} ({len(tgt_samples)} muestras) ===")
        dicts, y_win, leagues, _ = _build_feature_matrix(tgt_samples, verbose=verbose)
        X, names = _dicts_to_matrix(dicts)

        # Varianza
        std = X.std(axis=0)
        low_var = []
        for i, n in enumerate(names):
            zfrac = _zero_frac(X[:, i])
            if std[i] < 1e-4 or zfrac > 0.99:
                low_var.append((n, float(std[i]), zfrac))

        # Correlacion con target
        corr_y = {}
        for j in range(X.shape[1]):
            corr_y[names[j]] = _safe_corr(X[:, j], y_win.astype(np.float32))

        # Correlacion inter-feature
        C = _corr_matrix(X, names)
        pairs = _find_redundant_pairs(C, names, corr_threshold)
        clusters = _cluster_by_correlation(pairs)

        # Elegir representante por cluster
        top_feat_freq = _feature_frequency_in_top()
        cluster_recs: list[tuple[str, list[str], list[str]]] = []
        for cl in clusters:
            keep, drops = _pick_representative(
                cl, corr_y, top_feat_freq, is_tfm_protected=protect_tfm,
            )
            cluster_recs.append((keep, cl, drops))
        cluster_recs.sort(key=lambda t: -len(t[1]))

        # Permutation importance opcional en la liga mas grande
        perm_imp: dict[str, dict[str, float]] | None = None
        if do_permutation and val_samples:
            by_league = defaultdict(list)
            for s in val_samples:
                by_league[s.league].append(s)
            # top 3 ligas con mas val samples
            top_leagues = sorted(by_league.items(), key=lambda kv: -len(kv[1]))[:3]
            perm_imp = {}
            for league, lg_samples in top_leagues:
                if len(lg_samples) < 40:
                    continue
                slug = ds.slugify_league(league)
                print(f"[audit] perm-imp {league} ({target}): {len(lg_samples)} samples")
                # dicts+y para esta liga+target
                lg_dicts, lg_y, _, _ = _build_feature_matrix(lg_samples, verbose=False)
                imp = _permutation_importance_for_league(
                    slug, target, lg_samples, lg_dicts, lg_y,
                )
                if imp is not None:
                    perm_imp[league] = imp

        # Render
        md = _render_markdown(
            corr_threshold=corr_threshold, target=target,
            n_samples=len(tgt_samples), n_features=len(names),
            low_variance=low_var, corr_with_target=corr_y,
            top_features_freq=top_feat_freq,
            redundancy_clusters=cluster_recs,
            perm_importance=perm_imp,
        )
        REPORTS_DIR.mkdir(exist_ok=True, parents=True)
        out_md = REPORTS_DIR / f"FEATURE_AUDIT_{target}.md"
        out_md.write_text(md, encoding="utf-8")
        print(f"[audit] reporte guardado en {out_md}")

        results[target] = {
            "n_samples": len(tgt_samples),
            "n_features": len(names),
            "low_variance": low_var,
            "redundant_clusters": cluster_recs,
            "corr_with_target_top30": sorted(
                corr_y.items(), key=lambda kv: -abs(kv[1]),
            )[:30],
            "blacklist": sorted({d for _, _, ds_ in cluster_recs for d in ds_} |
                                {n for n, _, _ in low_var}),
            "permutation_importance": perm_imp or {},
        }

    # Union global para blacklist
    union_bl = sorted(
        {x for r in results.values() for x in r.get("blacklist", [])}
    )
    results["global_blacklist"] = union_bl
    print(f"\n[audit] blacklist union (q3+q4) = {len(union_bl)} features")

    # Combined markdown
    combined_md = "# FEATURE AUDIT v16 (combined)\n\n"
    combined_md += f"Generado: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    for tgt in ("q3", "q4"):
        if tgt in results:
            combined_md += f"## Resumen {tgt}\n"
            combined_md += f"- Samples: {results[tgt]['n_samples']}\n"
            combined_md += f"- Features: {results[tgt]['n_features']}\n"
            combined_md += f"- Clusters redundantes: {len(results[tgt]['redundant_clusters'])}\n"
            combined_md += f"- Blacklist: {len(results[tgt]['blacklist'])}\n\n"
    combined_md += "## Blacklist global (union q3 + q4)\n\n"
    combined_md += "```python\n"
    combined_md += "FEATURE_BLACKLIST = [\n"
    for n in union_bl:
        combined_md += f"    {n!r},\n"
    combined_md += "]\n```\n\n"
    combined_md += "Detalle completo por target en `FEATURE_AUDIT_q3.md` y `FEATURE_AUDIT_q4.md`.\n"
    AUDIT_MD.write_text(combined_md, encoding="utf-8")
    print(f"[audit] combined report -> {AUDIT_MD}")

    AUDIT_JSON.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"[audit] JSON -> {AUDIT_JSON}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corr", type=float, default=0.90,
                    help="Umbral de correlacion Pearson para clustering (default 0.90)")
    ap.add_argument("--quick", action="store_true",
                    help="Omite permutation importance")
    ap.add_argument("--no-tfm", action="store_true",
                    help="Permite drop de features tfm_*")
    ap.add_argument("--sample", type=int, default=15000,
                    help="Muestreo aleatorio (0 = sin muestreo, default 15000)")
    ap.add_argument("--trained-only", action="store_true", default=True,
                    help="Filtrar a ligas entrenables (default ON)")
    ap.add_argument("--all-leagues", dest="trained_only", action="store_false",
                    help="Usar todas las ligas (aun las no entrenables)")
    args = ap.parse_args()
    run_audit(
        corr_threshold=args.corr,
        do_permutation=not args.quick,
        protect_tfm=not args.no_tfm,
        sample_size=args.sample if args.sample > 0 else None,
        trained_only=args.trained_only,
    )


if __name__ == "__main__":
    main()
