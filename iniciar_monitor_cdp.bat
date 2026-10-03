@echo off
cd /d "%~dp0"
set SSL_CERT_FILE=%CD%\.venv\Lib\site-packages\certifi\cacert.pem
set OBSCURA_CDP_URL=http://127.0.0.1:9222
set SOFASCORE_SCRAPER_BACKEND=obscura
set SOFASCORE_SCRAPER_BACKEND_PROBE=obscura
set SOFASCORE_SCRAPER_BACKEND_LIVE=obscura
set SOFASCORE_SCRAPER_BACKEND_FT=obscura
call .venv\Scripts\activate && python bet_monitor_v2\main.py
pause
