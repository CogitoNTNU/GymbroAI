"""
Add this to BOTH your counter files.
Place counts.json in the same folder as Index.html.
"""

import json
import os

COUNTS_FILE = os.path.join(os.path.dirname(__file__), 'counts.json')

def update_count(exercise_key: str, count: int):
    """Write the current rep count to counts.json so the dashboard can read it."""
    try:
        # Read existing counts
        if os.path.exists(COUNTS_FILE):
            with open(COUNTS_FILE, 'r') as f:
                data = json.load(f)
        else:
            data = {}

        # Update this exercise
        data[exercise_key] = count

        # Write back atomically
        with open(COUNTS_FILE, 'w') as f:
            json.dump(data, f)
    except Exception as e:
        print(f"[dashboard sync] Could not write counts.json: {e}")


# ─────────────────────────────────────────────────────────────────
# In bicep_curl_counter.py — call this wherever you increment reps:
#
#   bicep_count += 1
#   update_count('bicep_curl_counter', bicep_count)
#
# In shoulder_press_counter.py — call this wherever you increment reps:
#
#   shoulder_count += 1
#   update_count('shoulder_press_counter', shoulder_count)
# ─────────────────────────────────────────────────────────────────
