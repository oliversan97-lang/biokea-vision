"""Freeze per-image detections (features) next to each labelled photo: <stem>.json.

Run BEFORE re-calibrating / re-saving the background, so older photos keep the reference
they were taken with. cli.py train reads these JSONs when present.
"""
import glob
import json
import sys

import cv2

import vision as V

tv = V.TrayVision(".")
folders = sys.argv[1:] or sorted(glob.glob("data/singles/*"))
for d in folders:
    for f in sorted(glob.glob(f"{d}/*.jpg")):
        res = tv.analyze(cv2.imread(f), frame=f)[0]
        json.dump(res, open(f[:-4] + ".json", "w"))
        good = [o for o in res["objects"] if not o["edge"]]
        print(f, "| shift", (res["align"] or {}).get("shift_mm", 0), "mm | objects", len(good))
