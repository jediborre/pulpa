#!/usr/bin/env python3
import sys
import traceback
sys.path.insert(0, '.')

try:
    print("Starting test...", flush=True)
    from training import infer_match
    print("Imported infer_match", flush=True)
    
    # Test with fetch_missing=False to avoid Playwright scraping
    print("Calling run_inference...", flush=True)
    result = infer_match.run_inference('15879747', metric='f1', fetch_missing=False, force_version='v9')
    
    print(f"OK: {result.get('ok')}", flush=True)
    print(f"Result keys: {list(result.keys())}", flush=True)
    
    if not result.get('ok'):
        print(f"Not OK - Reason: {result.get('reason')}", flush=True)
    else:
        for target in ('q3', 'q4'):
            pred = result.get('predictions', {}).get(target, {})
            if pred.get('available'):
                print(f"{target}: {pred.get('predicted_winner')} (conf={pred.get('confidence'):.4f})", flush=True)
            else:
                print(f"{target}: Not available - {pred.get('reason')}", flush=True)
                
except Exception as e:
    print(f"ERROR: {e}", flush=True)
    traceback.print_exc()
