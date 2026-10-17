"""Re-anchor calibration to the empty-tray background frame (adds marker_ref for auto-alignment)."""
import glob
import json

import cv2

import vision as V

bgf = sorted(glob.glob("captures/*_bg.jpg"))[-1]
cal = V.calibrate_marker_size(cv2.imread(bgf), None, roi="frame")
json.dump(cal, open("calib.json", "w"), indent=1)
print("calib from", bgf, {k: cal[k] for k in ("marker_ids", "native_px_per_mm", "marker_size_spread_pct")})
tv = V.TrayVision(".")
for f in sorted(glob.glob("data/singles/*/*.jpg")):
    r = tv.analyze(cv2.imread(f))[0]
    good = [o for o in r["objects"] if not o["edge"]]
    a = r["align"] or {}
    print(f.split("singles/")[1], "| shift_mm", a.get("shift_mm", 0), "rot", a.get("rot_deg"), "| objects", len(good))
