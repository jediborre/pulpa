s = open(r"C:\Users\App\Desktop\pulpa\models\m27_v5_3_3\features.py", encoding="utf-8").read()
# The inserted block is indented 8 spaces; fix to 4 spaces
old = """        def _clean_league_name(self, league: str) -> str:
        # Normaliza variantes (Runda Zasadnicza, PO, etc.) al clean_name.
        if not hasattr(self, "_clean_map"):
            try:
                import sqlite3
                from pathlib import Path
                _ROOT = Path(__file__).resolve().parents[2]
                _conn = sqlite3.connect(str(_ROOT / "matches.db"))
                self._clean_map = {r[0]: (r[1] or r[0]) for r in _conn.execute(
                    "SELECT league, clean_name FROM leagues_classification")}
                _conn.close()
            except Exception:
                self._clean_map = {}
        return self._clean_map.get(league, league)

    def league_info(self, league: str, fallback: dict[str, Any] | None = None) -> dict[str, Any]:
        league = self._clean_league_name(league)
        info = self.lc_map.get(league)
        if info:
            return info
        return fallback or {"q_dur": 10, "is_w": 0, "lg_avg_ft": 158.0, "lg_avg_q4": 39.5}"""
new = """    def _clean_league_name(self, league: str) -> str:
        # Normaliza variantes (Runda Zasadnicza, PO, etc.) al clean_name.
        if not hasattr(self, "_clean_map"):
            try:
                import sqlite3
                from pathlib import Path
                _ROOT = Path(__file__).resolve().parents[2]
                _conn = sqlite3.connect(str(_ROOT / "matches.db"))
                self._clean_map = {r[0]: (r[1] or r[0]) for r in _conn.execute(
                    "SELECT league, clean_name FROM leagues_classification")}
                _conn.close()
            except Exception:
                self._clean_map = {}
        return self._clean_map.get(league, league)

    def league_info(self, league: str, fallback: dict[str, Any] | None = None) -> dict[str, Any]:
        league = self._clean_league_name(league)
        info = self.lc_map.get(league)
        if info:
            return info
        return fallback or {"q_dur": 10, "is_w": 0, "lg_avg_ft": 158.0, "lg_avg_q4": 39.5}"""
assert old in s, "old block not found"
s = s.replace(old, new)
open(r"C:\Users\App\Desktop\pulpa\models\m27_v5_3_3\features.py", "w", encoding="utf-8").write(s)
print("fixed")
