"""Baselines (spec 8): RGB-only, LWIR-only and the RGB + LWIR baseline the student must beat.

    python train/train_baseline.py --model rgb_lwir --data out/dataset
    python train/train_baseline.py --model rgb_only
    python train/train_baseline.py --model lwir_only
    python train/train_baseline.py --model rgb_lwir --no-p2      # spec 7: document the stride-4 gap
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
from train.engine import cli  # noqa: E402

if __name__ == "__main__":
    cli()
