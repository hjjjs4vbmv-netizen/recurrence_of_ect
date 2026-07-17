#!/usr/bin/env python3
"""Compatibility wrapper. Prefer scripts/collect_schedule_results.py."""

from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    target = Path(__file__).resolve().with_name("collect_schedule_results.py")
    runpy.run_path(str(target), run_name="__main__")
