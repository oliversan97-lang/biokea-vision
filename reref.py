"""Re-measure photos against an older reference (calib frame + background) after markers were moved.

usage: python3 reref.py <calib_capture.jpg> <bg.png> <folder> [<folder> ...]
"""
import glob
import json
import os
import shutil
import sys

import cv2

import vision as V

calib_img, bg, folders = sys.argv[1], sys.argv[2], sys.argv[3:]
ref = "ref_tmp"
os.makedirs(ref, exist_ok=True)
for f in ("config.json", "bg_noise.npy"):
    if os.path.exists(f):
        shutil.copy(f, ref)
shutil.copy(bg, os.path.join(ref, "bg.png"))
cal = V.calibrate_marker_size(cv2.imread(calib_img), None, roi="frame")
json.dump(cal, open(os.path.join(ref, "calib.json"), "w"))
tv = V.TrayVision(ref)
for d in folders:
    for f in sorted(glob.glob(f"{d}/*.jpg")):
        res = tv.analyze(cv2.imread(f), frame=f)[0]
        json.dump(res, open(f[:-4] + ".json", "w"))
        good = [o for o in res["objects"] if not o["edge"]]
        print(f, "| shift", (res["align"] or {}).get("shift_mm", 0), "mm | objects", len(good))
