"""After re-calibrating: find the tray's moulded dot again (template match against the previous
background) and move the dot mask + floor ROI by the same offset.

usage: python3 track_dot.py <previous_bg.png>
"""
import json
import sys

import cv2

old_bg, new_bg = cv2.imread(sys.argv[1]), cv2.imread("bg.png")
cfg = json.load(open("config.json"))
cx, cy, r = [int(v) for v in cfg["ignore_circles_px"][0]]
h = 40
T = cv2.cvtColor(old_bg[cy - h:cy + h, cx - h:cx + h], cv2.COLOR_BGR2GRAY)
S = 300                                                   # search +-300 px around the old position
x0, y0 = max(cx - S, 0), max(cy - S, 0)
G = cv2.cvtColor(new_bg[y0:cy + S, x0:cx + S], cv2.COLOR_BGR2GRAY)
_, score, _, loc = cv2.minMaxLoc(cv2.matchTemplate(G, T, cv2.TM_CCOEFF_NORMED))
nx, ny = x0 + loc[0] + h, y0 + loc[1] + h
if score < 0.6:
    sys.exit(f"dot not found (score {score:.2f}); config unchanged")
dx, dy = nx - cx, ny - cy
cfg["ignore_circles_px"] = [[nx, ny, r]]
if cfg.get("roi_px"):
    a, b, c, d = cfg["roi_px"]
    cfg["roi_px"] = [max(a + dx, 0), max(b + dy, 0), min(c + dx, new_bg.shape[1]), min(d + dy, new_bg.shape[0])]
json.dump(cfg, open("config.json", "w"), indent=1)
print(f"dot ({cx},{cy}) -> ({nx},{ny}) score {score:.2f}; roi {cfg.get('roi_px')}")
