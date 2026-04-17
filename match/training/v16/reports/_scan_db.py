import sqlite3
from datetime import datetime
from pathlib import Path

DB = Path(__file__).parent.parent.parent.parent / "matches.db"
conn = sqlite3.connect(str(DB))
cur = conn.cursor()

print("== VOLUMEN POR SEMANA (todos los matches) ==")
cur.execute("""
SELECT strftime('%Y-W%W', date) AS week, COUNT(*) AS n
FROM matches
WHERE date IS NOT NULL
GROUP BY week
ORDER BY week
""")
for w, n in cur.fetchall():
    print(f"  {w}  {n:>5}")

cur.execute("SELECT MIN(date), MAX(date) FROM matches")
min_d, max_d = cur.fetchone()
span = (datetime.strptime(max_d, "%Y-%m-%d") - datetime.strptime(min_d, "%Y-%m-%d")).days
print()
print(f"SPAN: {min_d} -> {max_d}   ({span} dias)")

print()
print("== TOP 25 LIGAS por volumen total ==")
header_league = "league"
header_total = "total"
header_30 = "last30"
header_7 = "last7"
print(f"  {header_league:50s} {header_total:>6}  {header_30:>6} {header_7:>5}")

cur.execute(f"""
SELECT league,
       COUNT(*) AS total,
       SUM(CASE WHEN date >= date('{max_d}','-30 day') THEN 1 ELSE 0 END) AS last30,
       SUM(CASE WHEN date >= date('{max_d}','-7 day')  THEN 1 ELSE 0 END) AS last7
FROM matches
WHERE date IS NOT NULL
GROUP BY league
HAVING total >= 50
ORDER BY total DESC
LIMIT 25
""")
for lg, total, l30, l7 in cur.fetchall():
    if not lg:
        continue
    show = lg[:50]
    print(f"  {show:50s} {total:>6}  {l30:>6} {l7:>5}")

print()
print("== MATCHES MEN-ONLY (lo que realmente entrena v16) ==")
cur.execute("""
SELECT COUNT(*) FROM matches
WHERE date IS NOT NULL
  AND (LOWER(league) NOT LIKE '%women%')
  AND (LOWER(league) NOT LIKE '%femenin%')
  AND (LOWER(league) NOT LIKE '%wnba%')
  AND (LOWER(league) NOT LIKE '%wcba%')
""")
men = cur.fetchone()[0]
print(f"  hombres aprox: {men}")

print()
print("== LIGAS QUE V16 ENTRENA HOY (>=300 muestras men en train) ==")
# ~50 dias del min date = jan 17 -> mar 8
cutoff_train = "2026-03-08"
cur.execute(f"""
SELECT league, COUNT(*) AS n
FROM matches
WHERE date IS NOT NULL AND date <= '{cutoff_train}'
  AND LOWER(league) NOT LIKE '%women%'
  AND LOWER(league) NOT LIKE '%femenin%'
GROUP BY league
HAVING n >= 150
ORDER BY n DESC
LIMIT 20
""")
for lg, n in cur.fetchall():
    if not lg:
        continue
    print(f"  {lg[:55]:55s} {n:>5}")

conn.close()
