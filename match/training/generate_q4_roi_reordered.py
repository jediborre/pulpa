"""Generate Q4 ROI reports with reordered columns (betting metrics before league)."""

from pathlib import Path
import csv
import json
import joblib
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
from openpyxl.utils import get_column_letter

import train_q3_q4_models_v6 as v6

ROOT = Path('match')
BASE_V6 = ROOT / 'training' / 'model_outputs_v6'
BASE_V62 = ROOT / 'training' / 'model_outputs_v6_2'
OUT_AB = BASE_V62 / 'Q4_ROI_AB_v6_vs_v6_2.xlsx'
OUT_MM = BASE_V62 / 'Q4_ROI_match_by_match_v6_vs_v6_2.xlsx'

ODDS = 1.4
BANK_START = 1000.0
BREAK_EVEN = 1.0 / ODDS
TARGET = 'q4'

# Staking params for detailed match-by-match
MODE = 'kelly_non_compound'
KELLY_MULT = 0.25
KELLY_CAP = 0.05
MIN_CONF_PROB = 0.58
STAKE_STEP = 25.0
MIN_STAKE = 25.0


def _kelly_fraction(p, odds):
    b = odds - 1.0
    q = 1.0 - p
    return ((b * p) - q) / b if b > 0 else 0.0


def _round_down_step(x, step):
    if step <= 0:
        return x
    return (x // step) * step


def _prepare_rows():
    samples = v6._build_samples(v6.DB_PATH)
    rows = [s for s in samples if s.target_q4 is not None]
    rows = sorted(rows, key=lambda s: s.dt)
    n_total = len(rows)
    n_train = int(n_total * 0.8)
    test_rows = rows[n_train:]
    return rows, test_rows, n_total, n_train


def _predict_v6_probs(test_rows):
    art_xgb = joblib.load(BASE_V6 / 'q4_xgb.joblib')
    art_hgb = joblib.load(BASE_V6 / 'q4_hist_gb.joblib')
    art_mlp = joblib.load(BASE_V6 / 'q4_mlp.joblib')

    x_dict = [s.features_q4 for s in test_rows]
    p_xgb = art_xgb['model'].predict_proba(art_xgb['vectorizer'].transform(x_dict))[:, 1]
    p_hgb = art_hgb['model'].predict_proba(art_hgb['vectorizer'].transform(x_dict))[:, 1]
    p_mlp = art_mlp['model'].predict_proba(art_mlp['vectorizer'].transform(x_dict))[:, 1]

    return ((p_xgb + p_hgb + p_mlp) / 3.0).tolist()


def _load_v62_exclusion_rules():
    cfg = json.loads((ROOT / 'training' / 'v6_2_league_name_exclusions.json').read_text(encoding='utf-8'))
    pats = []
    for cat in cfg.get('categories', []):
        for p in cat.get('patterns', []):
            p = str(p).strip()
            if p:
                pats.append((cat.get('name', 'uncategorized'), p, p.lower()))
    return pats


def _predict_v62_probs(test_rows):
    art = joblib.load(BASE_V62 / 'q4_champion.joblib')
    vec = art['vectorizer']
    models = art['models']
    keep_leagues = set(art['league_filter']['kept_leagues'])
    other_token = art['league_filter']['other_token']

    rules = _load_v62_exclusion_rules()

    probs = [None] * len(test_rows)
    excluded_flags = [False] * len(test_rows)
    excluded_reasons = [None] * len(test_rows)

    transformed = []
    transformed_idx = []
    
    for i, s in enumerate(test_rows):
        rec = dict(s.features_q4)
        lg = str(rec.get('league', ''))
        lg_lc = lg.lower()
        hit = None
        for cname, raw, low in rules:
            if low in lg_lc:
                hit = (cname, raw)
                break
        if hit is not None:
            excluded_flags[i] = True
            excluded_reasons[i] = f"excluded_league_name:{hit[0]}:{hit[1]}"
            continue

        if lg not in keep_leagues:
            rec['league'] = other_token
            rec['league_bucket'] = other_token
        transformed.append(rec)
        transformed_idx.append(i)

    if transformed:
        x_valid = vec.transform(transformed)
        p_xgb = models['xgb'].predict_proba(x_valid)[:, 1]
        p_hgb = models['hist_gb'].predict_proba(x_valid)[:, 1]
        p_blend = (0.6 * p_xgb + 0.4 * p_hgb)
        for j, orig_idx in enumerate(transformed_idx):
            probs[orig_idx] = float(p_blend[j])

    return probs, excluded_flags, excluded_reasons


def _simulate(model_name, test_rows, p_home_list, excluded_flags=None, excluded_reasons=None,
              mode='kelly_non_compound', kelly_mult=1.0, kelly_cap=1.0, min_conf_prob=0.5,
              stake_step=1.0, min_stake=1.0):
    excluded_flags = excluded_flags or [False] * len(test_rows)
    excluded_reasons = excluded_reasons or [None] * len(test_rows)

    bank = BANK_START
    details = []
    bets = wins = losses = no_bet = 0
    total_staked = 0.0
    peak = BANK_START
    max_dd = 0.0

    for i, s in enumerate(test_rows):
        y = int(s.target_q4)
        p_home = p_home_list[i]

        # REORDERED COLUMN SEQUENCE: betting metrics first, then league/team info
        rec = {
            # Betting decision & outcome (PRIORITY)
            'resultado_apuesta': None,
            'apuesta': None,
            'monto_apostado': 0.0,
            'ganancia': 0.0,
            'bank_final': BANK_START,
            
            # League and match identification (SECONDARY)
            'modelo': model_name,
            'partida_test': i + 1,
            'match_id': s.match_id,
            'fecha_hora': s.dt.isoformat(),
            'liga': str(s.features_q4.get('league', '')),
            'equipo_local': str(s.features_q4.get('home_team', '')),
            'equipo_visitante': str(s.features_q4.get('away_team', '')),
            
            # Match result
            'resultado_q4_home_gana': y,
            
            # Probabilities
            'prob_local': None if p_home is None else float(p_home),
            'prob_visitante': None if p_home is None else float(1.0 - p_home),
            
            # Model prediction
            'lado_predicho': None,
            'confianza_prob': None,
            'confianza_score_0_100': None,
            'nivel_confianza': None,
            
            # Betting parameters
            'apuestas_odds': ODDS,
            'probabilidad_empate': BREAK_EVEN,
            'edge': None,
            'kelly_fraction_raw': None,
            'kelly_fraction_used': None,
            'step_apuesta': stake_step,
            'razon_sin_apuesta': None,
            'pnl': 0.0,
            'banco_antes': bank,
            'ganancia_acumulada': bank - BANK_START,
            'roi_banco_acumulado': (bank - BANK_START) / BANK_START,
        }

        if excluded_flags[i]:
            rec['razon_sin_apuesta'] = excluded_reasons[i]
            no_bet += 1
            details.append(rec)
            continue

        if p_home is None:
            rec['razon_sin_apuesta'] = 'missing_prob'
            no_bet += 1
            details.append(rec)
            continue

        p_pick = p_home if p_home >= 0.5 else (1.0 - p_home)
        pick_home = p_home >= 0.5
        pick_side = 'local' if pick_home else 'visitante'

        rec['lado_predicho'] = pick_side
        rec['confianza_prob'] = p_pick
        rec['confianza_score_0_100'] = p_pick * 100.0
        rec['nivel_confianza'] = (
            'muy_alta' if p_pick >= 0.80 else
            'alta' if p_pick >= 0.70 else
            'media' if p_pick >= 0.60 else
            'baja'
        )
        rec['edge'] = p_pick - BREAK_EVEN

        if p_pick < min_conf_prob:
            rec['razon_sin_apuesta'] = 'confidence_filter'
            no_bet += 1
            details.append(rec)
            continue

        k_raw = _kelly_fraction(p_pick, ODDS)
        rec['kelly_fraction_raw'] = k_raw
        if k_raw <= 0:
            rec['razon_sin_apuesta'] = 'no_edge'
            no_bet += 1
            details.append(rec)
            continue

        k_used = min(k_raw * kelly_mult, kelly_cap)
        rec['kelly_fraction_used'] = k_used

        base_bank = BANK_START if mode == 'kelly_non_compound' else bank
        stake_raw = base_bank * k_used
        stake = _round_down_step(stake_raw, stake_step)
        if stake < min_stake:
            rec['razon_sin_apuesta'] = 'stake_below_25'
            no_bet += 1
            details.append(rec)
            continue
        if stake > bank:
            rec['razon_sin_apuesta'] = 'insufficient_bank'
            no_bet += 1
            details.append(rec)
            continue

        is_win = (y == 1 and pick_home) or (y == 0 and (not pick_home))
        pnl = stake * (ODDS - 1.0) if is_win else -stake
        bank += pnl

        # UPDATE PRIORITY COLUMNS with actual values
        rec['apuesta'] = 'SI' if is_win else 'NO'
        rec['resultado_apuesta'] = 'GANADA' if is_win else 'PERDIDA'
        rec['monto_apostado'] = stake
        rec['ganancia'] = pnl
        rec['bank_final'] = bank
        rec['pnl'] = pnl
        rec['banco_antes'] = BANK_START if mode == 'kelly_non_compound' else (bank - pnl)
        rec['ganancia_acumulada'] = bank - BANK_START
        rec['roi_banco_acumulado'] = (bank - BANK_START) / BANK_START

        bets += 1
        wins += int(is_win)
        losses += int(not is_win)
        total_staked += stake
        peak = max(peak, bank)
        dd = (peak - bank) / peak if peak > 0 else 0.0
        max_dd = max(max_dd, dd)

        details.append(rec)

    summary = {
        'modelo': model_name,
        'partidos_test': len(test_rows),
        'apuestas': bets,
        'ganadas': wins,
        'perdidas': losses,
        'empates_contados_como_perdida': 0,
        'sin_apuesta': no_bet,
        'tasa_ganancia': (wins / bets) if bets else 0.0,
        'banco_inicio': BANK_START,
        'banco_final': bank,
        'ganancia': bank - BANK_START,
        'roi_bank': (bank - BANK_START) / BANK_START,
        'total_apostado': total_staked,
        'apuesta_promedio': (total_staked / bets) if bets else 0.0,
        'yield_sobre_apostado': ((bank - BANK_START) / total_staked) if total_staked > 0 else 0.0,
        'max_drawdown': max_dd,
    }
    return details, summary


def _apply_excel_formatting(ws, header_row=1):
    """Apply Spanish headers and conditional formatting to worksheet."""
    header_fill = PatternFill(start_color='4472C4', end_color='4472C4', fill_type='solid')
    header_font = Font(color='FFFFFF', bold=True)
    
    thin_border = Border(
        left=Side(style='thin'),
        right=Side(style='thin'),
        top=Side(style='thin'),
        bottom=Side(style='thin')
    )

    for col in range(1, ws.max_column + 1):
        cell = ws.cell(row=header_row, column=col)
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
        cell.border = thin_border

    for row in range(header_row + 1, ws.max_row + 1):
        for col in range(1, ws.max_column + 1):
            cell = ws.cell(row=row, column=col)
            cell.border = thin_border
            if col in [5, 6]:  # ganancia, roi columns
                cell.number_format = '0.00'

    ws.freeze_panes = f'A{header_row + 1}'


def main():
    print("[Q4_ROI] preparando datos de prueba...")
    all_rows, test_rows, n_total, n_train = _prepare_rows()

    print("[Q4_ROI] predicciones v6...")
    p_v6 = _predict_v6_probs(test_rows)

    print("[Q4_ROI] predicciones v6.2...")
    p_v62, excl_flags, excl_reasons = _predict_v62_probs(test_rows)

    print("[Q4_ROI] simulación match-by-match...")
    v6_details, v6_summary = _simulate('v6', test_rows, p_v6, None, None, 
                                       MODE, KELLY_MULT, KELLY_CAP, MIN_CONF_PROB, STAKE_STEP, MIN_STAKE)
    v62_details, v62_summary = _simulate('v6.2', test_rows, p_v62, excl_flags, excl_reasons, 
                                         MODE, KELLY_MULT, KELLY_CAP, MIN_CONF_PROB, STAKE_STEP, MIN_STAKE)

    print("[Q4_ROI] escribiendo match-by-match Excel...")
    # Eliminar archivo si existe para evitar permission denied
    if OUT_MM.exists():
        OUT_MM.unlink()
    
    v6_df = pd.DataFrame(v6_details)
    v62_df = pd.DataFrame(v62_details)
    summary_df = pd.DataFrame([v6_summary, v62_summary])

    # Asegurar que las columnas estén en el orden correcto
    col_order = [
        'resultado_apuesta', 'apuesta', 'monto_apostado', 'ganancia', 'bank_final',
        'modelo', 'partida_test', 'match_id', 'fecha_hora', 'liga', 'equipo_local', 
        'equipo_visitante', 'resultado_q4_home_gana', 'prob_local', 'prob_visitante',
        'lado_predicho', 'confianza_prob', 'confianza_score_0_100', 'nivel_confianza',
        'apuestas_odds', 'probabilidad_empate', 'edge', 'kelly_fraction_raw', 
        'kelly_fraction_used', 'step_apuesta', 'razon_sin_apuesta', 'pnl',
        'banco_antes', 'ganancia_acumulada', 'roi_banco_acumulado'
    ]
    v6_df = v6_df[[c for c in col_order if c in v6_df.columns]]
    v62_df = v62_df[[c for c in col_order if c in v62_df.columns]]

    with pd.ExcelWriter(OUT_MM, engine='openpyxl') as writer:
        summary_df.to_excel(writer, sheet_name='summary', index=False)
        v6_df.to_excel(writer, sheet_name='v6_matches', index=False)
        v62_df.to_excel(writer, sheet_name='v6_2_matches', index=False)

        # Apply formatting
        for sheet_name in writer.sheets:
            ws = writer.sheets[sheet_name]
            _apply_excel_formatting(ws)

    print("[Q4_ROI] OK")
    print(f"[Q4_ROI] output={OUT_MM}")


if __name__ == "__main__":
    main()
