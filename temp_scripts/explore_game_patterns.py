"""Deep-dive into actual game graph and PBP patterns."""
import sqlite3
import json
import numpy as np
from collections import Counter

DB = r"C:\Users\borre\OneDrive\OLD\Escritorio\pulpa\match\matches.db"
conn = sqlite3.connect(DB)
cur = conn.cursor()

# --- Get some 10m and 12m games that are in the m30_v1 dataset ---
# First find games with quarter_scores to identify regulation length
cur.execute("""
    SELECT q.match_id, m.league, m.home_team, m.away_team, 
           COUNT(DISTINCT q.quarter) as quarter_count,
           MAX(CASE WHEN q.quarter='Q1' THEN q.home END) as q1h,
           MAX(CASE WHEN q.quarter='Q1' THEN q.away END) as q1a,
           MAX(CASE WHEN q.quarter='Q2' THEN q.home END) as q2h,
           MAX(CASE WHEN q.quarter='Q2' THEN q.away END) as q2a,
           MAX(CASE WHEN q.quarter='Q3' THEN q.home END) as q3h,
           MAX(CASE WHEN q.quarter='Q3' THEN q.away END) as q3a,
           MAX(CASE WHEN q.quarter='Q4' THEN q.home END) as q4h,
           MAX(CASE WHEN q.quarter='Q4' THEN q.away END) as q4a
    FROM quarter_scores q
    JOIN matches m ON q.match_id = m.match_id
    WHERE q.quarter IN ('Q1','Q2','Q3','Q4')
    GROUP BY q.match_id
    HAVING quarter_count = 4
    LIMIT 5
""")
rows = cur.fetchall()
print(f"Found {len(rows)} complete games\n")
sample_mids = []
for r in rows:
    mid = r[0]
    sample_mids.append(mid)
    # Estimate reg length
    q1h, q1a, q2h, q2a, q3h, q3a, q4h, q4a = r[4], r[5], r[6], r[7], r[8], r[9], r[10], r[11]
    # Check total minutes from graph_points
    cur.execute("SELECT MAX(minute) FROM graph_points WHERE match_id=?", (mid,))
    max_min = cur.fetchone()[0]
    print(f"{r[1]:40s} | {r[2]:25s} vs {r[3]:25s} | max_min={max_min}")

# --- Graph points deep analysis ---
print("\n\n=== GRAPH POINTS DEEP DIVE ===")
for mid in sample_mids[:3]:
    cur.execute("SELECT minute, value FROM graph_points WHERE match_id=? ORDER BY minute", (mid,))
    gp = cur.fetchall()
    minutes = [g[0] for g in gp]
    values = [g[1] for g in gp]
    
    if not gp:
        continue
    
    print(f"\n--- Match: {mid} ---")
    print(f"  Points: {len(gp)}")
    
    # Basic stats
    gp_arr = np.array(values)
    print(f"  Value range: {gp_arr.min()} to {gp_arr.max()}")
    print(f"  Value mean: {gp_arr.mean():.1f}, std: {gp_arr.std():.1f}")
    print(f"  Value at min 30: {values[-1] if minutes and minutes[-1] >= 30 else 'N/A'}")
    
    # Differences (momentum at each step)
    diffs = np.diff(values)
    print(f"  Avg step change: {diffs.mean():.2f}")
    print(f"  Max step change: {abs(diffs).max()}")
    
    # Count sign changes (turning points)
    sign_changes = sum(1 for i in range(1, len(diffs)) if diffs[i-1] * diffs[i] < 0)
    print(f"  Turning points (sign changes): {sign_changes}")
    
    # Volatility: std of diffs
    print(f"  Step volatility (std of diffs): {diffs.std():.2f}")
    
    # Time above/below zero
    above_zero = sum(1 for v in values if v > 0)
    below_zero = sum(1 for v in values if v < 0)
    at_zero = sum(1 for v in values if v == 0)
    print(f"  Minutes above zero: {above_zero}, below: {below_zero}, zero: {at_zero}")
    
    # Feature: max lead
    max_lead = max(abs(v) for v in values)
    print(f"  Max lead (abs): {max_lead}")
    
    # Feature: lead changes
    # Count how many times the sign of the value changes
    lead_changes = 0
    prev_sign = 0
    for v in values:
        sign = 1 if v > 0 else (-1 if v < 0 else 0)
        if sign != 0 and prev_sign != 0 and sign != prev_sign:
            lead_changes += 1
        if sign != 0:
            prev_sign = sign
    print(f"  Lead changes: {lead_changes}")
    
    # Recent slope (last 5 mins)
    recent = values[-5:] if len(values) >= 5 else values
    recent_slope = recent[-1] - recent[0]
    print(f"  Last-5 slope: {recent_slope}")
    
    # Last 3 mins
    recent3 = values[-3:] if len(values) >= 3 else values
    recent3_slope = recent3[-1] - recent3[0]
    print(f"  Last-3 slope: {recent3_slope}")
    
    # Full time series for visual
    print(f"  Full values: {values}")
    
    # Analyze momentum zones: is the game trending strongly in one direction?
    # Split into halves and compare
    mid_point = len(values) // 2
    first_half = values[:mid_point]
    second_half = values[mid_point:]
    first_trend = first_half[-1] - first_half[0] if len(first_half) > 1 else 0
    second_trend = second_half[-1] - second_half[0] if len(second_half) > 1 else 0
    print(f"  First half trend: {first_trend}, Second half trend: {second_trend}")
    
    # Acceleration: is the second derivative positive or negative?
    # If diffs are increasing, momentum is accelerating
    if len(diffs) >= 15:
        early_diffs = diffs[:len(diffs)//2]
        late_diffs = diffs[len(diffs)//2:]
        print(f"  Early step volatility: {early_diffs.std():.2f}, Late step volatility: {late_diffs.std():.2f}")
        print(f"  Early mean diff: {early_diffs.mean():.2f}, Late mean diff: {late_diffs.mean():.2f}")

# --- Now analyze specific games with interesting patterns ---
print("\n\n=== SOFT VS VOLATILE GAMES ===")
# Find most volatile and least volatile games
cur.execute("""
    SELECT gp.match_id, 
           COUNT(*) as n_points,
           MAX(gp.minute) as max_min,
           ROUND(AVG(ABS(gp.value)), 1) as avg_abs,
           ROUND(MAX(ABS(gp.value)), 0) as max_abs,
           m.league, m.home_team, m.away_team
    FROM graph_points gp
    JOIN matches m ON gp.match_id = m.match_id
    WHERE gp.minute <= 30
    GROUP BY gp.match_id
    HAVING n_points >= 20 AND max_min >= 28
    ORDER BY RANDOM()
    LIMIT 10
""")
games = cur.fetchall()
print(f"\nRandom 10 games sample:")
for g in games:
    cur.execute("""
        SELECT value FROM graph_points 
        WHERE match_id=? AND minute <= 30 
        ORDER BY minute
    """, (g[0],))
    vals = [r[0] for r in cur.fetchall()]
    if len(vals) >= 5:
        diffs = np.diff(vals)
        vol = diffs.std()
        sign_changes = sum(1 for i in range(1, len(diffs)) if diffs[i-1] * diffs[i] < 0)
        print(f"  {g[5]:30s} {g[6]:20s} vs {g[7]:20s} | abs_avg={g[3]:5.1f} max_lead={g[4]:3.0f} volatility={vol:.2f} turns={sign_changes} final={vals[-1]}")

# --- PBP event patterns ---
print("\n\n=== PBP EVENT PATTERNS ===")
for mid in sample_mids[:3]:
    cur.execute("""
        SELECT quarter, seq, time, team, points, home_score, away_score
        FROM play_by_play
        WHERE match_id=?
        ORDER BY 
            CASE quarter 
                WHEN 'Q1' THEN 1 WHEN 'Q2' THEN 2 
                WHEN 'Q3' THEN 3 WHEN 'Q4' THEN 4 
                ELSE 5 
            END,
            seq
    """, (mid,))
    events = cur.fetchall()
    print(f"\n--- Match: {mid} ({len(events)} events total) ---")
    
    # Home/away scoring runs
    home_runs = []
    away_runs = []
    current_run_team = None
    current_run_pts = 0
    for ev in events:
        team = ev[3]
        pts = ev[4] or 0
        if pts > 0:
            if team == current_run_team:
                current_run_pts += pts
            else:
                if current_run_team == "home" and current_run_pts > 0:
                    home_runs.append(current_run_pts)
                elif current_run_team == "away" and current_run_pts > 0:
                    away_runs.append(current_run_pts)
                current_run_team = team
                current_run_pts = pts
    if current_run_team == "home" and current_run_pts > 0:
        home_runs.append(current_run_pts)
    elif current_run_team == "away" and current_run_pts > 0:
        away_runs.append(current_run_pts)
    
    print(f"  Home runs: {home_runs}")
    print(f"  Away runs: {away_runs}")
    print(f"  Max home run: {max(home_runs) if home_runs else 0}")
    print(f"  Max away run: {max(away_runs) if away_runs else 0}")
    
    # Event density (how many scoring events per minute)
    # Last 5 minutes of Q3 events
    q3_events = [ev for ev in events if ev[0] == 'Q3']
    if q3_events:
        print(f"  Q3 events: {len(q3_events)}")
        # Show last 5 Q3 events
        last_q3 = q3_events[-5:]
        print(f"  Last 5 Q3 events:")
        for ev in last_q3:
            print(f"    {ev[1]:4s} t={ev[2]:6s} team={ev[3]:6s} pts={ev[4]} score={ev[5]}-{ev[6]}")

# --- Now look specifically at games WHERE Q3->Q4 momentum shifts happen ---
print("\n\n=== Q3->Q4 MOMENTUM SHIFTS ===")
# Find games where Q4 winner was NOT Q3+halftime leader (comebacks)
cur.execute("""
    SELECT q.match_id,
           q2h + q1h as ht_home, q2a + q1a as ht_away,
           q3h, q3a,
           q4h, q4a,
           m.league, m.home_team, m.away_team
    FROM (
        SELECT match_id,
               MAX(CASE WHEN quarter='Q1' THEN home END) as q1h,
               MAX(CASE WHEN quarter='Q1' THEN away END) as q1a,
               MAX(CASE WHEN quarter='Q2' THEN home END) as q2h,
               MAX(CASE WHEN quarter='Q2' THEN away END) as q2a,
               MAX(CASE WHEN quarter='Q3' THEN home END) as q3h,
               MAX(CASE WHEN quarter='Q3' THEN away END) as q3a,
               MAX(CASE WHEN quarter='Q4' THEN home END) as q4h,
               MAX(CASE WHEN quarter='Q4' THEN away END) as q4a
        FROM quarter_scores
        WHERE quarter IN ('Q1','Q2','Q3','Q4')
        GROUP BY match_id
        HAVING COUNT(DISTINCT quarter) = 4
    ) q
    JOIN matches m ON q.match_id = m.match_id
    WHERE (q3h + q2h + q1h) - (q3a + q2a + q1a) > 0  -- home leads after Q3
      AND q4h < q4a  -- away wins Q4
    LIMIT 3
""")
comebacks = cur.fetchall()
for c in comebacks:
    mid = c[0]
    ht_home = c[1]
    ht_away = c[2]
    ht_diff = ht_home - ht_away
    q3h, q3a = c[3], c[4]
    q4h, q4a = c[5], c[6]
    after_q3_diff = (ht_home + q3h) - (ht_away + q3a)
    print(f"\nComeback: {c[7]:30s} {c[8]:20s} vs {c[9]:20s}")
    print(f"  Halftime: {ht_home}-{ht_away} (diff={ht_diff:+d})")
    print(f"  Q3: {q3h}-{q3a}")
    print(f"  After Q3: diff={after_q3_diff:+d}")
    print(f"  Q4: {q4h}-{q4a} (away wins Q4)")
    
    # Show the graph trajectory
    cur.execute("SELECT minute, value FROM graph_points WHERE match_id=? ORDER BY minute", (mid,))
    gp = cur.fetchall()
    if gp:
        vals = [g[1] for g in gp]
        print(f"  Graph trajectory: {vals}")

conn.close()
