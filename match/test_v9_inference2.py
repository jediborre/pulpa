#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')
from training import infer_match

# Redirect output to file
with open('test_v9_output.log', 'w') as f:
    # Test with fetch_missing=False to avoid Playwright scraping
    result = infer_match.run_inference('15879747', metric='f1', fetch_missing=False, force_version='v9')
    f.write(f"OK: {result.get('ok')}\n")
    f.write(f"Result keys: {list(result.keys())}\n")
    
    if result.get('ok'):
        for target in ('q3', 'q4'):
            pred = result.get('predictions', {}).get(target, {})
            if pred.get('available'):
                f.write(f"{target}: {pred.get('predicted_winner')} (conf={pred.get('confidence'):.4f}, signal={pred.get('final_recommendation')})\n")
            else:
                f.write(f"{target}: Not available - {pred.get('reason')}\n")
    else:
        f.write(f"Failed: {result.get('reason')}\n")
        if 'match' in result:
            f.write(f"Match info: {result['match']}\n")
    
    f.flush()
    print("Done")
