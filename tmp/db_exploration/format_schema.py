"""
Script auxiliar para generar resumen formateado de columnas y tipos para SCHEMA_DATABASE.md
"""
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def main():
    details = json.loads(Path("tmp/db_exploration/table_details.json").read_text(encoding="utf-8"))
    summary = json.loads(Path("tmp/db_exploration/db_summary.json").read_text(encoding="utf-8"))

    lines = []
    for tname, tdata in details.items():
        row_count = summary["tables"].get(tname, {}).get("count", 0)
        lines.append(f"\n### Tabla `{tname}` ({row_count:,} filas)")
        lines.append("| Columna | Tipo | Nulo | PK | Ejemplo / Descripción |")
        lines.append("| :--- | :--- | :---: | :---: | :--- |")
        sample = tdata.get("sample", {})
        for col in tdata["columns"]:
            cname = col["name"]
            ctype = col["type"]
            notnull = "No" if col["notnull"] == 1 else "Sí"
            pk = "Sí" if col["pk"] >= 1 else "No"
            val = str(sample.get(cname, ""))
            if len(val) > 40:
                val = val[:37] + "..."
            lines.append(f"| `{cname}` | `{ctype}` | {notnull} | {pk} | `{val}` |")

    Path("tmp/db_exploration/schema_preview.md").write_text("\n".join(lines), encoding="utf-8")
    print("schema_preview.md escrito.")

if __name__ == "__main__":
    main()
