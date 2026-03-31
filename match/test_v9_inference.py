#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')
from training import infer_match

# Test with fetch_missing=False to avoid Playwright scraping
result = infer_match.run_inference('15879747', metric='f1', fetch_missing=False, force_version='v9')
print(f"OK: {result.get('ok')}")
print(f"Result keys: {list(result.keys())}")

if result.get('ok'):
    for target in ('q3', 'q4'):
        pred = result.get('predictions', {}).get(target, {})
        if pred.get('available'):
            print(f"{target}: {pred.get('predicted_winner')} (conf={pred.get('confidence'):.4f}, signal={pred.get('final_recommendation')})")
        else:
            print(f"{target}: Not available - {pred.get('reason')}")
else:
    print(f"Failed: {result.get('reason')}")
    if 'match' in result:
        print(f"Match info: {result['match']}")
