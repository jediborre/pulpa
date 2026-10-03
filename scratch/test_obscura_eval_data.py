"""Test: usar obscura fetch --eval síncrono para explorar datos"""
import subprocess
import json
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
ROOT = Path(__file__).resolve().parents[1]
OBSCURA_EXE = ROOT / "tools" / "obscura" / "v0.1.5" / "obscura-fixed.exe"
CERT_PATH = ROOT / ".venv" / "Lib" / "site-packages" / "certifi" / "cacert.pem"
os.environ["SSL_CERT_FILE"] = str(CERT_PATH)

def obscura_eval(url, js, wait=10, timeout=30):
    cmd = [str(OBSCURA_EXE), "fetch", url, "--stealth", "--wait", str(wait), "--timeout", str(timeout), "--dump", "text", "--eval", js]
    result = subprocess.run(cmd, env=os.environ, capture_output=True, text=True, encoding='utf-8', errors='ignore', timeout=timeout + 15)
    if result.returncode == 0 and result.stdout.strip():
        return result.stdout.strip()
    return None

yesterday = (datetime.now(timezone.utc) - timedelta(days=3)).strftime("%Y-%m-%d")
print(f"Page: https://www.sofascore.com/basketball/{yesterday}")

t = obscura_eval("https://www.sofascore.com/basketball", "document.title")
print(f"Title: {t}")

js = "(()=>{const s=document.getElementById('__NEXT_DATA__');if(!s)return'none';const d=JSON.parse(s.textContent);return JSON.stringify(Object.keys(d?.props?.pageProps||{}))})()"
print(f"pageProps keys: {obscura_eval('https://www.sofascore.com/basketball/'+yesterday, js)}")

js = "(()=>{const s=document.querySelectorAll('script');return JSON.stringify(Array.from(s).filter(x=>x.textContent.includes('scheduledEvents')||x.textContent.includes('homeScore')).map(x=>({src:x.src?.substring(0,100)||'inline',len:(x.textContent||'').length,id:x.id})))})()"
print(f"Scripts with data: {obscura_eval('https://www.sofascore.com/basketball/'+yesterday, js, wait=15, timeout=40)}")

js = "(()=>JSON.stringify(Object.keys(window).filter(k=>!k.startsWith('$')&&!k.startsWith('on')&&!k.startsWith('_')).slice(0,30)))()"
print(f"Window keys: {obscura_eval('https://www.sofascore.com/basketball/'+yesterday, js)}")
