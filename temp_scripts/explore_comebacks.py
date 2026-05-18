"""Analyze Q3->Q4 momentum shifts and identify pattern features."""
import sqlite3
import numpy as np

DB = r"C:\Users\borre\OneDrive\OLD\Escritorio\pulpa\match\matches.db"
conn = sqlite3.connect(DB)
cur = conn.cursor()

# --- Q3->Q4 comebacks: games where Q4 winner was NOT the after-Q3 leader ---
print("=== Q3->Q4 MOMENTUM SHIFTS (comebacks) ===")
cur.execute("""
    SELECT qs.match_id,
           qs.q1h + qs.q2h as ht_h, qs.q1a + qs.q2a as ht_a,
           qs.q3h, qs.q3a, qs.q4h, qs.q4a,
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
    ) qs
    JOIN matches m ON qs.match_id = m.match_id
    WHERE (qs.q3h + qs.q1h + qs.q2h) - (qs.q3a + qs.q1a + qs.q2a) > 0
      AND qs.q4h < qs.q4a
    LIMIT 5
""")
rows = cur.fetchall()
print(f"Found {len(rows)} comebacks (home leads after Q3, away wins Q4)")

for r in rows:
    mid, ht_h, ht_a, q3h, q3a, q4h, q4a = r[0], r[1], r[2], r[3], r[4], r[5], r[6]
    ht_diff = ht_h - ht_a
    after_q3_h = ht_h + q3h
    after_q3_a = ht_a + q3a
    after_q3_diff = after_q3_h - after_q3_a
    
    cur.execute("SELECT minute, value FROM graph_points WHERE match_id=? ORDER BY minute", (mid,))
    gp = cur.fetchall()
    
    print(f"\n{r[7]:30s} {r[8]:20s} vs {r[9]:20s}")
    print(f"  Halftime: {ht_h}-{ht_a} (diff={ht_diff:+d})")
    print(f"  Q3: {q3h}-{q3a}")
    print(f"  After Q3: {after_q3_h}-{after_q3_a} (diff={after_q3_diff:+d})")
    print(f"  Q4: {q4h}-{q4a} (away wins)")
    if gp:
        vals = [g[1] for g in gp]
        vals_30 = [v for g, v in zip(gp, vals) if g[0] <= 30]
        print(f"  Graph values (to min30): {vals_30}")


# --- Stable games: leader wins Q4 comfortably ---
print("\n\n=== STABLE GAMES (leader holds Q4) ===")
cur.execute("""
    SELECT qs.match_id,
           qs.q1h + qs.q2h as ht_h, qs.q1a + qs.q2a as ht_a,
           qs.q3h, qs.q3a, qs.q4h, qs.q4a,
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
    ) qs
    JOIN matches m ON qs.match_id = m.match_id
    WHERE ABS((qs.q3h + qs.q1h + qs.q2h) - (qs.q3a + qs.q1a + qs.q2a)) >= 8
      AND (qs.q4h - qs.q4a) * ((qs.q3h + qs.q1h + qs.q2h) - (qs.q3a + qs.q1a + qs.q2a)) > 0
    ORDER BY RANDOM()
    LIMIT 5
""")
rows = cur.fetchall()
for r in rows:
    mid, ht_h, ht_a, q3h, q3a, q4h, q4a = r[0], r[1], r[2], r[3], r[4], r[5], r[6]
    ht_diff = ht_h - ht_a
    after_q3_diff = (ht_h + q3h) - (ht_a + q3a)
    
    cur.execute("SELECT minute, value FROM graph_points WHERE match_id=? ORDER BY minute", (mid,))
    gp = cur.fetchall()
    
    # Also get PBP events to see last Q3 actions
    cur.execute("""
        SELECT quarter, seq, time, team, points, home_score, away_score
        FROM play_by_play WHERE match_id=?
        ORDER BY 
            CASE quarter 
                WHEN 'Q1' THEN 1 WHEN 'Q2' THEN 2 
                WHEN 'Q3' THEN 3 WHEN 'Q4' THEN 4 ELSE 5 
            END, seq
    """, (mid,))
    events = cur.fetchall()
    q3_events = [e for e in events if e[0] == 'Q3']
    
    winner = "home" if q4h > q4a else "away"
    
    print(f"\n{r[7]:30s} {r[8]:20s} vs {r[9]:20s}")
    print(f"  Halftime diff={ht_diff:+d}, After Q3 diff={after_q3_diff:+d}, Q4 winner={winner} ({q4h}-{q4a})")
    if gp:
        vals = [g[1] for g in gp]
        vals_30 = [v for g, v in zip(gp, vals) if g[0] <= 30]
        
        # Key features for this game
        diffs = np.diff(vals_30) if len(vals_30) > 1 else []
        vol = np.std(diffs) if len(diffs) > 0 else 0
        turns = sum(1 for i in range(1, len(diffs)) if diffs[i-1] * diffs[i] < 0) if len(diffs) > 1 else 0
        max_abs = max(abs(v) for v in vals_30) if vals_30 else 0
        last5_slope = vals_30[-1] - vals_30[-6] if len(vals_30) >= 6 else (vals_30[-1] - vals_30[0] if vals_30 else 0)
        last3_slope = vals_30[-1] - vals_30[-4] if len(vals_30) >= 4 else last5_slope
        
        print(f"  Graph stats: max_abs={max_abs}, vol={vol:.2f}, turns={turns}, last3_slope={last3_slope:+d}, last5_slope={last5_slope:+d}")
        print(f"  Graph values (last 10): {vals_30[-10:] if len(vals_30) >= 10 else vals_30}")
    
    if q3_events:
        last3 = q3_events[-3:]
        print(f"  Last 3 Q3 events:")
        for ev in last3:
            t, team, pts = ev[2], ev[3], ev[4]
            print(f"    t={t} team={team} pts={pts}")


# --- Close games at minute 30 (score_diff <= 7) ---
print("\n\n=== CLOSE GAMES AT MINUTE 30 ===")
cur.execute("""
    SELECT gp.match_id, gp.value as score_at_30, 
           qs.q4h, qs.q4a,
           m.league, m.home_team, m.away_team
    FROM graph_points gp
    JOIN (
        SELECT match_id,
               MAX(CASE WHEN quarter='Q4' THEN home END) as q4h,
               MAX(CASE WHEN quarter='Q4' THEN away END) as q4a
        FROM quarter_scores WHERE quarter IN ('Q1','Q2','Q3','Q4')
        GROUP BY match_id HAVING COUNT(DISTINCT quarter) = 4
    ) qs ON gp.match_id = qs.match_id
    JOIN matches m ON gp.match_id = m.match_id
    WHERE gp.minute = 30 AND ABS(gp.value) <= 7
    ORDER BY RANDOM()
    LIMIT 8
""")
rows = cur.fetchall()
print(f"Found {len(rows)} close games at min 30")

for r in rows:
    mid, score30, q4h, q4a = r[0], r[1], r[2], r[3]
    actual_winner = "home" if q4h > q4a else ("away" if q4a > q4h else "tie")
    
    cur.execute("SELECT minute, value FROM graph_points WHERE match_id=? AND minute <= 30 ORDER BY minute", (mid,))
    gp = cur.fetchall()
    
    # Direction at minute 30 (last 5 slope)
    vals = [g[1] for g in gp]
    last5 = vals[-5:]
    # Recent momentum: last 3 step changes
    recent_diffs = np.diff(vals[-4:]) if len(vals) >= 4 else []
    mean_momentum = np.mean(recent_diffs) if len(recent_diffs) > 0 else 0
    
    # Second derivative (acceleration)
    if len(recent_diffs) >= 2:
        accel = recent_diffs[-1] - recent_diffs[-2]
    else:
        accel = 0
    
    print(f"\n{mid[:12]} | {r[4]:30s} {r[5]:20s} vs {r[6]:20s}")
    print(f"  Score@30: {score30:+d}, Q4 actual: {q4h}-{q4a} ({actual_winner})")
    print(f"  Last-5: {last5} | recent_momentum={mean_momentum:+.2f} | accel={accel:+.2f}")


# --- Identify key metrics to predict Q4 winner ---
print("\n\n=== CORRELATION CANDIDATES: momentum metrics vs Q4 winner ===")
# For this, we need to use the training data. Let's use a simpler approach:
# Compute features from graph_points for a sample of games and see which
# ones might predict Q4 direction.

# Find games where we have both graph_points and Q4 outcome
cur.execute("""
    SELECT gp.match_id, gp.minute, gp.value,
           qs.q4h, qs.q4a
    FROM graph_points gp
    JOIN (
        SELECT match_id,
               MAX(CASE WHEN quarter='Q4' THEN home END) as q4h,
               MAX(CASE WHEN quarter='Q4' THEN away END) as q4a
        FROM quarter_scores WHERE quarter IN ('Q1','Q2','Q3','Q4')
        GROUP BY match_id HAVING COUNT(DISTINCT quarter) = 4
    ) qs ON gp.match_id = qs.match_id
    WHERE gp.minute = 30 AND gp.value IS NOT NULL
    LIMIT 500
""".replace("LIMIT 500", ""))  # Remove LIMIT for full analysis
rows = cur.fetchall()
print(f"Total games with minute-30 value: {len(rows)}")

# Now compute features for each game and check correlation
q4_winners = []
score30s = []
for r in rows[:200]:  # Sample for speed
    mid, score30, q4h, q4a = r[0], r[1], r[2], r[3]
    q4_winner = 1 if q4h > q4a else 0  # 1 = home wins Q4
    q4_winners.append(q4_winner)
    score30s.append(score30)

# Simple correlation: score at minute 30 vs Q4 winner
corr = np.corrcoef(score30s, q4_winners)[0, 1]
print(f"Correlation(score@30, Q4 winner) = {corr:.4f}  (n={len(score30s)})")

# Split: how often does leader at 30 win Q4?
leads_win = sum(1 for s, w in zip(score30s, q4_winners) if (s > 0 and w == 1) or (s < 0 and w == 0))
total_nonzero = sum(1 for s in score30s if s != 0)
print(f"Leader@30 wins Q4: {leads_win}/{total_nonzero} = {leads_win/total_nonzero:.3f}")


# --- Let's look at the correlation of graph-derived features ---
print("\n\n=== GRAPH-DERIVED FEATURE CORRELATION SAMPLE ===")
cur.execute("""
    SELECT gp.match_id, 
           MAX(CASE WHEN gp.minute = 30 THEN gp.value END) as score30,
           qs.q4h, qs.q4a
    FROM graph_points gp
    JOIN (
        SELECT match_id,
               MAX(CASE WHEN quarter='Q4' THEN home END) as q4h,
               MAX(CASE WHEN quarter='Q4' THEN away END) as q4a
        FROM quarter_scores WHERE quarter IN ('Q1','Q2','Q3','Q4')
        GROUP BY match_id HAVING COUNT(DISTINCT quarter) = 4
    ) qs ON gp.match_id = qs.match_id
    WHERE gp.minute = 30
    GROUP BY gp.match_id
    HAVING COUNT(*) >= 25  -- enough graph points
    LIMIT 500
""")
rows = cur.fetchall()
print(f"Sample: {len(rows)} games")

# Compute correlations manually
scores30 = []
targets = []
for r in rows:
    scores30.append(r[1])
    targets.append(1 if r[2] > r[3] else 0)

scores30 = np.array(scores30)
targets = np.array(targets)

# Accuracy if betting on leader at minute 30
for threshold in [0, 3, 5, 7, 10]:
    if threshold == 0:
        mask = scores30 != 0
    else:
        mask = np.abs(scores30) >= threshold
    
    sub_scores = scores30[mask]
    sub_targets = targets[mask]
    if len(sub_scores) > 0:
        correct = np.sum((sub_scores > 0) == (sub_targets == 1))
        print(f"  |score@30| >= {threshold:2d}: n={len(sub_scores):4d}, leader wins Q4 = {correct/len(sub_scores):.3f}")

conn.close()
