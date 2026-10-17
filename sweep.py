"""Sensitivity sweep on the latest chia photo: detections vs. blobs near walls (false-positive proxy)."""
import glob
import sys

import cv2

import vision as V

tv = V.TrayVision(".")
img = cv2.imread(sorted(glob.glob(f"data/singles/{sys.argv[1] if len(sys.argv) > 1 else 'chia'}/*.jpg"))[-1])
for k, t in ((2, 10), (1, 8), (1, 7), (0.7, 6)):
    tv.cfg.update(noise_k=k, min_threshold=t)
    r, rect, mask, objs = tv.analyze(img)
    good = [o for o in r["objects"] if not o["edge"]]
    walls = [o for o in good if o["centroid_px"][0] < 330 or o["centroid_px"][0] > 1450 or o["area_mm2"] > 6]
    print(f"noise_k={k} min_thr={t}: objects={len(good)}  suspicious(wall/large)={len(walls)}")
    if (k, t) == (1, 7):
        cv2.imwrite("/tmp/sweep_overlay.jpg", V.draw_overlay(rect, objs, r, tv.cfg))
