"""Distilled student (spec 7): RGB + LWIR, trained with L = L_det + lambda * L_KD(teacher).

    python train/train_student.py --data out/dataset --teacher out/runs/teacher/best.pt

Same architecture, data and schedule as the rgb_lwir baseline, so the difference between the two is
the effect of distilling from the SWIR-trained teacher.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
from train.engine import cli  # noqa: E402

if __name__ == "__main__":
    cli(default_model="student", needs_teacher=True)
