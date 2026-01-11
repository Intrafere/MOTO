#!/usr/bin/env python3
"""Verify the HTML update."""

import re

with open("WEBSITE STUFF/moto-research-log.html", "r", encoding="utf-8") as f:
    content = f.read()

# Check for new events
if "8:48:54 PM" in content:
    print("[OK] New events ARE present in HTML")
else:
    print("[FAIL] New events NOT found in HTML")

# Check for updated stats
if '"total_submissions":296' in content or '"total_submissions": 296' in content:
    print("[OK] Updated total_submissions found")
else:
    print("[FAIL] Updated total_submissions NOT found")
    # Check what's there
    matches = re.findall(r'"total_submissions":(\d+)', content)
    print(f"  Found values: {matches}")

# Check for paper writing event
if "Paper writing started: Explicit Local Langlands" in content:
    print("[OK] Paper writing event found")
else:
    print("[FAIL] Paper writing event NOT found")

# Check sample1Data line
if 'const sample1Data = ' in content:
    print("[OK] sample1Data declaration found")
    # Extract just first 200 chars after declaration
    idx = content.find('const sample1Data = ')
    snippet = content[idx:idx+300]
    print(f"  First 300 chars: {snippet[:300]}")

