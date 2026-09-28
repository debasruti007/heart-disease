"""
logger.py
Appends one row per metric to experiments/results.csv.
This file IS the experiment log required in the submission checklist.
Every script in this project uses log_result() so the format never changes.
"""

import csv
import os
from datetime import datetime

LOG_PATH = "experiments/results.csv"
FIELDS = ["timestamp", "script", "model", "settings", "seed", "metric", "value"]


def log_result(script, model, settings, seed, metric, value):
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    new_file = not os.path.exists(LOG_PATH)
    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(FIELDS)
        writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
                        script, model, settings, seed, metric, float(f"{float(value):.6g}"),
        ])