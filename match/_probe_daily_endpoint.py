from playwright.sync_api import sync_playwright

candidates = [
    'https://api.sofascore.com/api/v1/sport/basketball/scheduled-events/2026-03-20',
    'https://api.sofascore.com/api/v1/sport/basketball/events/2026-03-20',
    'https://api.sofascore.com/api/v1/sport/basketball/events/2026-03-20/inverse',
    'https://api.sofascore.com/api/v1/sport/basketball/events/2026-03-20?only=finished',
]
headers = {
    'Referer': 'https://www.sofascore.com/',
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'en-US,en;q=0.9',
}
ua = ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
      'AppleWebKit/537.36 (KHTML, like Gecko) '
      'Chrome/122.0.0.0 Safari/537.36')

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    ctx = browser.new_context(user_agent=ua)
    page = ctx.new_page()
    page.goto('https://www.sofascore.com/basketball', wait_until='networkidle', timeout=45000)
    for u in candidates:
        r = ctx.request.get(u, headers=headers, timeout=20000)
        print('URL', u)
        print('status', r.status, 'ok', r.ok)
        if r.ok:
            j = r.json()
            if isinstance(j, dict):
                print('keys', list(j.keys())[:8])
                evs = j.get('events') if isinstance(j.get('events'), list) else []
                print('events', len(evs))
                if evs:
                    e0 = evs[0]
                    print('sample id', e0.get('id'), 'status', (e0.get('status') or {}).get('type'))
        print('-' * 60)
    browser.close()
