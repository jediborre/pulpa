import asyncio, json, sqlite3, sys, time
from pathlib import Path
ROOT = Path(r"C:\Users\App\Desktop\pulpa")
sys.path.insert(0, str(ROOT))
from monitor_v3.core.mobile_client import get_mobile_client

DB = ROOT / "matches.db"
CHECKPOINT = ROOT / "tmp" / "m27_v5_3_audit" / "fill_countries_state.json"

def load_state():
    if CHECKPOINT.exists():
        return json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    return {"done": {}}

def save_state(st):
    CHECKPOINT.write_text(json.dumps(st, ensure_ascii=False, indent=2), encoding="utf-8")

async def main():
    st = load_state()
    done = st["done"]
    conn = sqlite3.connect(str(DB))
    conn.row_factory = sqlite3.Row
    # missing leagues: in analysis but not in leagues_classification
    D = json.load(open(ROOT / "tmp" / "m27_v5_3_audit" / "league_threshold_analysis_v5_3_3.json", encoding="utf-8"))
    missing = [l["league"] for l in D["leagues"] if l["league"] not in {r["league"] for r in conn.execute("SELECT league FROM leagues_classification")}]
    print("missing:", len(missing))
    async with get_mobile_client() as c:
        for lg in missing:
            if lg in done:
                print("skip (cached):", lg, "->", done[lg])
                continue
            q = lg.split(",")[0].strip() + " basketball"
            await asyncio.sleep(0.5)
            try:
                r = await c.request("GET", "/search/" + q.replace(" ", "%20"))
                data = r.json()
                res = [x for x in data.get("results", []) if x.get("entity", {}).get("category", {}).get("sport", {}).get("name") == "Basketball"]
                country = None
                if res:
                    ent = res[0]["entity"]
                    country = ent.get("category", {}).get("country", {}).get("name") or ent.get("category", {}).get("name")
                done[lg] = country or "?"
                print("%-50s -> %s" % (lg[:50], done[lg]))
            except Exception as e:
                done[lg] = "?"
                print("ERR", lg, e)
            save_state(st)
    # write results to a file for review
    conn.close()
    out = json.dumps(done, ensure_ascii=False, indent=2)
    (ROOT / "tmp" / "m27_v5_3_audit" / "country_fill.json").write_text(out, encoding="utf-8")
    print("\nDONE. Filled", len(done), "leagues.")

asyncio.run(main())
