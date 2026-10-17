"""Make a synthetic tray photo (ArUco corners + fake foods) to smoke-test the pipeline."""
import sys
import cv2
import numpy as np

rng = np.random.default_rng(int(sys.argv[2]) if len(sys.argv) > 2 else 0)
W, H = 2592, 1944
img = np.full((H, W, 3), (235, 238, 240), np.uint8)
d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
mk = 160
pos = [(100, 100), (W - 100 - mk, 100), (W - 100 - mk, H - 100 - mk), (100, H - 100 - mk)]
for i, (x, y) in enumerate(pos):
    img[y - 20:y + mk + 20, x - 20:x + mk + 20] = 255
    img[y:y + mk, x:x + mk] = cv2.cvtColor(cv2.aruco.generateImageMarker(d, i, mk), cv2.COLOR_GRAY2BGR)
# inner corners: (260,260) .. (W-260, H-260)  -> choose 9.6 px/mm => W_mm = (2592-520)/9.6
ppm = 9.6
foods = {  # (length_mm, width_mm, BGR)
    "chia": (1.3, 0.9, (60, 60, 70)),
    "couscous": (1.8, 1.6, (120, 200, 230)),
    "white_rice": (6.0, 2.2, (250, 250, 250)),
    "wild_rice": (10.0, 1.9, (30, 30, 40)),
    "broken_noodle": (15.0, 6.0, (90, 190, 235)),
    "whole_noodle": (30.0, 6.0, (90, 190, 235)),
}
truth = []
for name, (L, Wd, col) in foods.items():
    for _ in range(8):
        x, y = rng.uniform(300, W - 300), rng.uniform(300, H - 300)
        ang = rng.uniform(0, 180)
        box = cv2.boxPoints(((x, y), (L * ppm, Wd * ppm), ang)).astype(np.int32)
        if L / Wd < 1.5:
            cv2.ellipse(img, ((x, y), (L * ppm, Wd * ppm), ang), col, -1)
        else:
            cv2.ellipse(img, ((x, y), (L * ppm, Wd * ppm), ang), col, -1) if name != "whole_noodle" and name != "broken_noodle" else cv2.fillPoly(img, [box], col)
        truth.append(name)
img = cv2.GaussianBlur(img, (3, 3), 0)
img = np.clip(img.astype(np.int16) + rng.normal(0, 3, img.shape), 0, 255).astype(np.uint8)
cv2.imwrite(sys.argv[1], img)
print("inner-corner size mm:", (W - 520) / ppm, (H - 520) / ppm, "items:", len(truth))
