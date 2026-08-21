#!/usr/bin/env python3
import os
import shutil
from pathlib import Path

def cleanup():
    # Issue: Unsafe default path
    log_dir = os.getenv("LOG_DIR", "/var/log")

    # Issue: Recursive delete without safeguards
    target = Path(log_dir) / "app"
    print(f"Deleting logs in: {target}")
    shutil.rmtree(target, ignore_errors=True)

if __name__ == "__main__":
    cleanup()
