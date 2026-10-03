# -*- coding: utf-8 -*-
import io

def main():
    import sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    
    log_path = r"c:\Users\App\Desktop\pulpa\logs\monitor_2026-05-27.log"
    
    print(f"Reading the last 40 lines of {log_path}...\n")
    
    try:
        with io.open(log_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
            
        last_lines = lines[-40:]
        for i, line in enumerate(last_lines):
            line_num = len(lines) - 40 + i + 1
            print(f"Line {line_num}: {line.strip()}")
    except Exception as e:
        print("Error reading log:", e)


if __name__ == "__main__":
    main()
