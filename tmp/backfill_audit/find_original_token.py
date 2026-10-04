"""
Busca en transcript_full el token original de 6 meses que se capturó desde el teléfono con HTTP Toolkit.
"""
import json
from pathlib import Path

transcript_path = Path("C:/Users/App/.gemini/antigravity-cli/brain/377c4e37-30d5-4458-a9a1-d5f98ec5ed0a/.system_generated/logs/transcript_full.jsonl")

if transcript_path.exists():
    with open(transcript_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            content = data.get("content", "")
            if "eyJ0eXAi" in content and data.get("step_index", 0) < 3000:
                print(f"Step {data.get('step_index')}:")
                start = content.find("eyJ0eXAi")
                while start != -1:
                    end = content.find('"', start)
                    end2 = content.find("'", start)
                    end3 = content.find(" ", start)
                    end4 = content.find("\n", start)
                    end_pos = min(x for x in [end, end2, end3, end4, len(content)] if x > start)
                    tok = content[start:end_pos].strip()
                    if len(tok) > 200:
                        print(f"TOKEN (len={len(tok)}):\n{tok}\n")
                        break
                    start = content.find("eyJ0eXAi", start + 1)
