"""
Script para verificar qué información exacta de jugadores existe por cuarto en matches.db.
Revisa player_stats, play_by_play y match_events.
"""
import sqlite3
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DB_PATH = Path("matches.db")

def main():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    print("1. Columnas de player_stats:")
    cols_ps = [c["name"] for c in cur.execute("PRAGMA table_info(player_stats)").fetchall()]
    print("  ", cols_ps)

    print("\n2. Tipos de incidentes en match_events:")
    events = cur.execute("SELECT incident_type, COUNT(*) as cnt FROM match_events GROUP BY incident_type ORDER BY cnt DESC").fetchall()
    for e in events:
        print(f"   - {e['incident_type']}: {e['cnt']} registros")

    print("\n3. Muestra de eventos de sustitución en match_events (si existen):")
    subs = cur.execute("SELECT * FROM match_events WHERE incident_type = 'substitution' OR subtype LIKE '%sub%' LIMIT 5").fetchall()
    print(f"   Total encontrados en muestra: {len(subs)}")
    for s in subs:
        print("    ", dict(s))

    print("\n4. Jugadores presentes por cuarto en play_by_play:")
    pbp_sample = cur.execute("""
        SELECT match_id, quarter, player, team, COUNT(*) as actions
        FROM play_by_play
        WHERE player IS NOT NULL AND player != ''
        GROUP BY match_id, quarter, player, team
        LIMIT 10
    """).fetchall()
    for p in pbp_sample:
        print(f"   Match {p['match_id']} | {p['quarter']} | {p['team']} | Jugador: {p['player']} ({p['actions']} jugadas)")

    print("\n5. ¿Cómo se calcula realmente el Elo por cuarto?:")
    print("   Opción A: A nivel de EQUIPO por cuarto (quarter_scores: Q1, Q2, Q3, Q4) -> 100% disponible sin necesidad de tracking de jugadores.")
    print("   Opción B: A nivel de JUGADORES por cuarto (play_by_play + match_events) -> se sabe quién anotó/jugó en cada cuarto.")

if __name__ == "__main__":
    main()
