"""BioKEA tray vision: calibrate -> segment -> measure -> classify -> pick list.

Everything is computed in a *rectified* top-down image where 1 px = 1/px_per_mm mm,
so all measurements come out directly in millimetres.
"""
import json
import math
import os
import subprocess
import time

import cv2
import numpy as np

CLASSES = ["seed", "couscous", "rice", "noodle"]           # the 4 product categories (by size)
# Photo folders -> category: chia is the seed; white+wild rice are rice; broken+whole are noodle.
LABEL_MERGE = {"chia": "seed", "white_rice": "rice", "wild_rice": "rice",
               "broken_noodle": "noodle", "whole_noodle": "noodle"}
FRACTION = {"seed": "fine", "couscous": "couscous", "rice": "rice", "noodle": "noodle", "unknown": "unknown"}
# BGR overlay colours = the bulb colours
CLASS_COLOR = {
    "seed": (0, 0, 255),        # red
    "couscous": (0, 220, 255),  # yellow
    "rice": (0, 200, 0),        # green
    "noodle": (255, 80, 0),     # blue
    "unknown": (160, 160, 160),
}
# Rough upper bound on the area of ONE item (mm^2); bigger => probably a clump.
CLASS_MAX_AREA = {"seed": 4.5, "couscous": 7.0, "rice": 28.0, "noodle": 500.0}
SIZE_FEATURES = ["length_mm", "width_mm", "thickness_mm", "area_mm2"]
FEATURES = ["length_mm", "width_mm", "thickness_mm", "area_mm2", "aspect",
            "solidity", "circularity", "L", "a", "b", "L_std"]

DEFAULT_CFG = {
    "px_per_mm": 10.0,          # resolution of the rectified image
    "uncal_px_per_mm": 10.0,    # scale used when there is no ArUco calibration
    "threshold": None,          # None => Otsu on the colour-difference image
    "min_threshold": 10,        # low threshold that finds blobs (with a background model)
    "noise_k": 2.0,             # extra threshold per unit of per-pixel background noise
    "half_max": 0.5,            # blob outline at this fraction of the blob's own peak contrast
    "min_area_mm2": 0.4,        # ignore dust
    "nozzle_mm": 1.5,           # suction nozzle diameter (ask Sean)
    "clearance_mm": 3.0,        # min gap to the nearest neighbour to be pickable
    "max_pick_len_mm": 35.0,
    "min_conf": 0.5,
    "use_color": True,          # set False when backlit (everything is a silhouette)
    "plate": 48,                # 12, 48 or 96 wells
}


def load_json(path, default=None):
    if path and os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return default


def git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=os.path.dirname(os.path.abspath(__file__)),
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "nogit"


# --------------------------------------------------------------------------- calibration
# Marker sizes on our printed sheet (cli.py markers): ids 0-7 = 20 mm, ids 8-11 = 12 mm.
SHEET_MARKER_MM = {**{i: 20.0 for i in range(8)}, **{i: 12.0 for i in range(8, 12)}}


def aruco_detect(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    p = cv2.aruco.DetectorParameters()          # tuned: default missed 2 of 4 slightly blurry markers
    p.adaptiveThreshWinSizeMax = 63
    p.adaptiveThreshWinSizeStep = 6
    p.polygonalApproxAccuracyRate = 0.05
    p.perspectiveRemoveIgnoredMarginPerCell = 0.2
    p.errorCorrectionRate = 1.0                 # tolerate bit errors from a slightly soft lens
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    corners, ids, _ = cv2.aruco.ArucoDetector(d, p).detectMarkers(gray)
    if ids is None:
        return {}
    # keep only ids printed on our sheet: the permissive settings can hallucinate others
    return {int(i): c.reshape(4, 2) for c, i in zip(corners, ids.flatten()) if int(i) in SHEET_MARKER_MM}


def _inner_corners(found):
    """For each marker, the corner closest to the centre of the marker ring."""
    ring_c = np.array([c.mean(0) for c in found.values()]).mean(0)
    return {k: c[np.argmin(np.linalg.norm(c - ring_c, axis=1))] for k, c in found.items()}


def _exclusions(found, Hm, marker_mm=None, margin_mm=9.5):
    """Marker + its white paper border (6 mm on our sheet, +slack), mapped into the rectified image."""
    polys = []
    for i, c in found.items():
        mm = marker_mm or SHEET_MARKER_MM.get(i, 20.0)
        grow = (mm + 2 * margin_mm) / mm
        ctr = c.mean(0)
        quad = ((c - ctr) * grow + ctr).astype(np.float32).reshape(-1, 1, 2)
        polys.append(cv2.perspectiveTransform(quad, Hm).reshape(-1, 2).round().astype(int).tolist())
    return polys


def calibrate_marker_size(img, marker_mm=None, px_per_mm=None, roi="frame"):
    """No tape measure: scale comes from the printed marker size.

    marker_mm=None -> size looked up per id from our sheet (20 mm / 12 mm).
    Markers may sit anywhere (corners, edge midpoints, ...); they are masked out.
    roi="frame": analyse the whole camera view (rely on the empty-tray background to ignore
    static stuff). roi="markers": only the box between the markers' inner corners.
    Assumes a roughly top-down camera (check marker_size_spread_pct < ~3).
    """
    found = aruco_detect(img)
    if len(found) < 1:
        raise RuntimeError("no ArUco markers found")
    ratios = np.array([np.linalg.norm(c[(k + 1) % 4] - c[k]) / (marker_mm or SHEET_MARKER_MM.get(i, 20.0))
                       for i, c in found.items() for k in range(4)])          # px per mm, per marker side
    native = float(np.median(ratios))
    ppm = float(px_per_mm or round(native, 3))
    s = ppm / native
    if roi == "markers" and len(found) >= 2:
        pts = np.array(list(_inner_corners(found).values()))
        pad = 7.5 * native                                    # paper border sticks into the box
        (x0, y0), (x1, y1) = pts.min(0) + pad, pts.max(0) - pad
    else:
        x0, y0, x1, y1 = 0, 0, img.shape[1], img.shape[0]
    Hm = np.array([[s, 0, -s * x0], [0, s, -s * y0], [0, 0, 1]], np.float64)
    size = [int(round((x1 - x0) * s)), int(round((y1 - y0) * s))]
    return {
        "method": "marker_size", "roi": roi, "marker_mm": marker_mm or {i: SHEET_MARKER_MM.get(i, 20.0) for i in found},
        "H": Hm.tolist(), "size": size,
        "px_per_mm": ppm, "native_px_per_mm": round(native, 2),
        "width_mm": round(size[0] / ppm, 1), "height_mm": round(size[1] / ppm, 1),
        "marker_ids": sorted(found), "exclude": _exclusions(found, Hm, marker_mm),
        "marker_size_spread_pct": round(100 * float(ratios.std() / ratios.mean()), 2),   # >3% => camera tilted
        "warning": None if len(found) >= 2 else "only 1 marker: scale is less certain",
        "marker_ref": {str(i): c.round(2).tolist() for i, c in found.items()},   # to re-align bumped trays
        "image_size": [img.shape[1], img.shape[0]], "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def calibrate(img, width_mm, height_mm, px_per_mm=10.0):
    """Use 4 ArUco markers (DICT_4X4_50, any ids) at the tray corners.

    width_mm / height_mm = distance between the *inner* corners of the markers
    (the corner of each marker that points at the tray centre).
    """
    found = aruco_detect(img)
    if len(found) < 4:
        raise RuntimeError(f"need 4 ArUco markers, found {sorted(found)}")
    if len(found) > 4:
        found = {k: found[k] for k in sorted(found)[:4]}
    centers = np.array([c.mean(0) for c in found.values()])
    tray_c = centers.mean(0)
    inner = np.array([c[np.argmin(np.linalg.norm(c - tray_c, axis=1))] for c in found.values()])
    s, d = inner.sum(1), np.diff(inner, axis=1).ravel()
    src = np.float32([inner[np.argmin(s)], inner[np.argmin(d)], inner[np.argmax(s)], inner[np.argmax(d)]])
    W, H = width_mm * px_per_mm, height_mm * px_per_mm
    dst = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
    Hm = cv2.getPerspectiveTransform(src, dst)
    native = (np.linalg.norm(src[1] - src[0]) + np.linalg.norm(src[2] - src[3])) / 2 / width_mm
    return {
        "method": "rect_wh",
        "H": Hm.tolist(),
        "size": [int(round(W)), int(round(H))],
        "px_per_mm": px_per_mm,
        "width_mm": width_mm,
        "height_mm": height_mm,
        "src_px": src.tolist(),
        "marker_ids": sorted(found),
        "exclude": _exclusions(found, Hm),
        "native_px_per_mm": round(float(native), 2),
        "image_size": [img.shape[1], img.shape[0]],
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


# --------------------------------------------------------------------------- segmentation
def segment(rect, cfg, bg=None, exclude=None, noise=None):
    """Foreground = pixels whose Lab colour differs from the tray background.

    noise: per-pixel std of the empty tray over several frames (flickery tray walls get a
    higher threshold automatically).
    """
    lab = cv2.cvtColor(cv2.GaussianBlur(rect, (0, 0), 1.0), cv2.COLOR_BGR2LAB).astype(np.float32)
    if bg is not None and bg.shape == rect.shape:
        ref = cv2.cvtColor(cv2.GaussianBlur(bg, (0, 0), 1.0), cv2.COLOR_BGR2LAB).astype(np.float32)
    else:   # no empty-tray photo yet: local background = coarse median (absorbs lighting gradients)
        f = 4
        small = cv2.resize(cv2.cvtColor(rect, cv2.COLOR_BGR2LAB), None, fx=1 / f, fy=1 / f, interpolation=cv2.INTER_AREA)
        k = int(20 * cfg["px_per_mm"] / f) | 1                       # ~20 mm window, odd
        small = cv2.medianBlur(small, min(max(k, 3), 255))
        ref = cv2.resize(small, (rect.shape[1], rect.shape[0]), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    diff = np.linalg.norm(lab - ref, axis=2)
    if noise is not None and noise.shape == diff.shape and bg is not None:
        diff = np.maximum(diff - cfg.get("noise_k", 4.0) * noise, 0)
    if exclude is not None and exclude.shape == diff.shape:
        diff[exclude > 0] = 0                      # markers: keep them out of Otsu and the mask
    d8 = np.clip(diff, 0, 255).astype(np.uint8)
    if cfg.get("threshold"):
        t = float(cfg["threshold"])
    elif bg is not None:                           # background model: a fixed low threshold is safe
        t = float(cfg["min_threshold"])
    else:
        t, _ = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        t = max(t, cfg["min_threshold"])
    k = max(3, int(cfg["px_per_mm"] * 0.25) | 1)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    low = cv2.morphologyEx((d8 > t).astype(np.uint8) * 255, cv2.MORPH_OPEN, ker)
    # Half-max outline per blob: edge where contrast drops to 50% of that blob's own peak, so a
    # bright rice grain and a dark chia seed are both measured at their real edge (soft lens halo).
    hm = cfg.get("half_max", 0.5)
    mask = np.zeros_like(low)
    if hm:
        n, lab_img, stats, _ = cv2.connectedComponentsWithStats(low, connectivity=8)
        pad = int(cfg["px_per_mm"])
        H, W = d8.shape
        for i in range(1, n):
            x, y, w, h, a = stats[i]
            x0, y0, x1, y1 = max(x - pad, 0), max(y - pad, 0), min(x + w + pad, W), min(y + h + pad, H)
            comp = lab_img[y0:y1, x0:x1] == i
            dr = d8[y0:y1, x0:x1]
            peak = float(np.percentile(dr[comp], 95))
            near = cv2.dilate(comp.astype(np.uint8), ker) > 0
            sub = (dr > max(t, hm * peak)) & near
            mask[y0:y1, x0:x1][sub] = 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, ker)
    else:
        mask = low
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ker)
    # fill holes
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, cnts, -1, 255, -1)
    return filled, d8, t


# --------------------------------------------------------------------------- measurement
def measure(rect, mask, cfg):
    ppm = cfg["px_per_mm"]
    lab = cv2.cvtColor(rect, cv2.COLOR_BGR2LAB)
    Hh, Ww = mask.shape
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    min_area_px = cfg["min_area_mm2"] * ppm * ppm
    objs = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area_px:
            continue
        x, y, w, h = cv2.boundingRect(c)
        x0, y0, x1, y1 = max(x - 2, 0), max(y - 2, 0), min(x + w + 2, Ww), min(y + h + 2, Hh)
        m = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.drawContours(m, [c - [x0, y0]], -1, 255, -1)
        dt = cv2.distanceTransform(m, cv2.DIST_L2, 5)
        _, rmax, _, loc = cv2.minMaxLoc(dt)               # trick 3: most interior point
        (cx, cy), (rw, rh), ang = cv2.minAreaRect(c)
        hull_a = cv2.contourArea(cv2.convexHull(c))
        per = cv2.arcLength(c, True)
        pix = lab[y0:y1, x0:x1][m > 0].astype(np.float32)
        Lm, am, bm = pix.mean(0)
        M = cv2.moments(c)
        length, width = max(rw, rh) / ppm, min(rw, rh) / ppm
        objs.append({
            "bbox_px": [int(x), int(y), int(w), int(h)],
            "centroid_px": [M["m10"] / M["m00"], M["m01"] / M["m00"]],
            "pick_px_rect": [int(loc[0] + x0), int(loc[1] + y0)],
            "length_mm": length,                     # ~max Feret
            "width_mm": width,                       # ~min Feret: what a sieve sees (trick 4)
            "thickness_mm": 2 * rmax / ppm,          # max inscribed diameter
            "area_mm2": area / ppm / ppm,
            "aspect": length / max(width, 1e-3),
            "solidity": area / max(hull_a, 1.0),
            "circularity": 4 * math.pi * area / max(per * per, 1.0),
            "L": float(Lm), "a": float(am), "b": float(bm),
            "L_std": float(pix[:, 0].std()),
            "edge": bool(x <= 1 or y <= 1 or x + w >= Ww - 1 or y + h >= Hh - 1),
            "_contour": c,
        })
    return objs


def add_clearance(objs, shape, ppm):
    """Gap (mm) from each object to its nearest neighbour via a Voronoi split of the background."""
    n = len(objs)
    if n == 0:
        return
    labels = np.zeros(shape, np.int32)
    for i, o in enumerate(objs):
        cv2.drawContours(labels, [o["_contour"]], -1, i + 1, -1)
    inv = (labels == 0).astype(np.uint8)
    dist, near = cv2.distanceTransformWithLabels(inv, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    lut = np.zeros(near.max() + 1, np.int32)
    fg = labels > 0
    lut[near[fg]] = labels[fg]
    owner = lut[near]
    gap = np.full(n + 1, np.inf)
    nb = np.full(n + 1, -1)
    for oa, ob, da, db in ((owner[:, :-1], owner[:, 1:], dist[:, :-1], dist[:, 1:]),
                           (owner[:-1, :], owner[1:, :], dist[:-1, :], dist[1:, :])):
        sel = oa != ob
        A, B, g = oa[sel], ob[sel], da[sel] + db[sel]
        np.minimum.at(gap, A, g)
        np.minimum.at(gap, B, g)
    # nearest-neighbour id (cheap second pass on the boundary pixels)
    for oa, ob, da, db in ((owner[:, :-1], owner[:, 1:], dist[:, :-1], dist[:, 1:]),
                           (owner[:-1, :], owner[1:, :], dist[:-1, :], dist[1:, :])):
        sel = oa != ob
        A, B, g = oa[sel], ob[sel], da[sel] + db[sel]
        hitA = g <= gap[A] + 1e-6
        nb[A[hitA]] = B[hitA]
        hitB = g <= gap[B] + 1e-6
        nb[B[hitB]] = A[hitB]
    for i, o in enumerate(objs):
        g = gap[i + 1]
        o["clearance_mm"] = None if not np.isfinite(g) else float(g / ppm)
        o["_nearest"] = int(nb[i + 1]) - 1 if nb[i + 1] > 0 else None


# --------------------------------------------------------------------------- classification
def rule_classify(o, use_color=True):
    """Size-only fallback (a 'virtual sieve'). Replaced by train.py's model when model.joblib exists."""
    L, W = o["length_mm"], o["width_mm"]
    if L >= 9.5 or W >= 4.0:
        return "noodle", 0.6
    if L >= 3.5:
        return "rice", 0.6
    if L >= 1.75 or W >= 1.4:
        return "couscous", 0.55
    return "seed", 0.55


class Classifier:
    def __init__(self, model_path=None):
        self.model = None
        if model_path and os.path.exists(model_path):
            import joblib
            blob = joblib.load(model_path)
            self.model, self.features = blob["model"], blob["features"]
            self.name = blob.get("name")

    @property
    def kind(self):
        return getattr(self, "name", None) or ("model" if self.model is not None else "rules")

    def predict(self, objs, use_color=True):
        if not objs:
            return
        if self.model is None:
            for o in objs:
                o["class"], o["confidence"] = rule_classify(o, use_color)
        else:
            X = np.array([[o[f] for f in self.features] for o in objs], np.float32)
            P = self.model.predict_proba(X)
            for o, p in zip(objs, P):
                k = int(np.argmax(p))
                o["class"], o["confidence"] = str(self.model.classes_[k]), float(p[k])
        for o in objs:   # physical prior: nothing but a noodle is this big (also covers whole noodles)
            if o["area_mm2"] >= 60 or o["length_mm"] >= 14:
                o["class"], o["confidence"] = "noodle", max(o["confidence"], 0.9)


# --------------------------------------------------------------------------- pipeline
class TrayVision:
    def __init__(self, root=None):
        self.root = root or os.path.dirname(os.path.abspath(__file__))
        self.reload()

    def p(self, *a):
        return os.path.join(self.root, *a)

    def reload(self):
        self.cfg = dict(DEFAULT_CFG, **(load_json(self.p("config.json"), {}) or {}))
        self.calib = load_json(self.p("calib.json"))
        bg = self.p("bg.png")
        self.bg = cv2.imread(bg) if os.path.exists(bg) else None
        nz = self.p("bg_noise.npy")
        self.noise = np.load(nz).astype(np.float32) if (self.bg is not None and os.path.exists(nz)) else None
        self.clf = Classifier(self.p("model.joblib"))
        self.exclude = None
        if self.calib and self.calib.get("exclude"):
            w, h = self.calib["size"]
            self.exclude = np.zeros((h, w), np.uint8)
            cv2.fillPoly(self.exclude, [np.array(p, np.int32) for p in self.calib["exclude"]], 255)
        self.roi = None
        if self.calib and self.cfg.get("roi_px"):        # [x0, y0, x1, y1]: only the tray floor
            w, h = self.calib["size"]
            x0, y0, x1, y1 = [int(v) for v in self.cfg["roi_px"]]
            self.roi = [max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)]
            if self.exclude is None:
                self.exclude = np.zeros((h, w), np.uint8)
            keep = np.zeros((h, w), bool)
            keep[self.roi[1]:self.roi[3], self.roi[0]:self.roi[2]] = True
            self.exclude[~keep] = 255
        if self.calib and self.cfg.get("ignore_circles_px"):   # static tray features, e.g. the moulded dot
            w, h = self.calib["size"]
            if self.exclude is None:
                self.exclude = np.zeros((h, w), np.uint8)
            for cx, cy, r in self.cfg["ignore_circles_px"]:
                cv2.circle(self.exclude, (int(cx), int(cy)), int(r), 255, -1)

    @property
    def ppm(self):
        return self.calib["px_per_mm"] if self.calib else self.cfg["uncal_px_per_mm"]

    def align(self, img):
        """Markers are taped to the tray: warp this frame so they sit where they were at calibration.

        Keeps the background model, marker masks and ROI valid when the tray (or camera) gets bumped.
        """
        ref = (self.calib or {}).get("marker_ref")
        if not ref:
            return img, None
        found = aruco_detect(img)
        src, dst = [], []
        for k, c in found.items():
            if str(k) in ref:
                src.extend(c.tolist())
                dst.extend(ref[str(k)])
        if len(src) < 8:
            return img, {"aligned": False, "markers": sorted(found), "why": "need 2 calibration markers in view"}
        M, inl = cv2.estimateAffinePartial2D(np.float32(src), np.float32(dst), method=cv2.RANSAC,
                                             ransacReprojThreshold=3.0)
        if M is None:
            return img, {"aligned": False, "markers": sorted(found), "why": "fit failed"}
        scale = float(np.hypot(M[0, 0], M[1, 0]))
        info = {"aligned": True, "markers": sorted(found), "dx_px": round(float(M[0, 2]), 1),
                "dy_px": round(float(M[1, 2]), 1), "rot_deg": round(float(np.degrees(np.arctan2(M[1, 0], M[0, 0]))), 2),
                "scale": round(scale, 4)}
        moved = np.hypot(M[0, 2], M[1, 2]) > 0.7 or abs(info["rot_deg"]) > 0.05 or abs(scale - 1) > 0.001
        if not moved:
            return img, info
        info["shift_mm"] = round(float(np.hypot(M[0, 2], M[1, 2]) / self.ppm), 1)
        info["M"] = M.round(6).tolist()          # camera -> calibration pose (for drawing back in camera view)
        return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE), info

    def rectify(self, img):
        if not self.calib:
            return img
        if list(self.calib.get("image_size", [img.shape[1], img.shape[0]])) != [img.shape[1], img.shape[0]]:
            raise RuntimeError("calibration was made at a different camera resolution: press Calibrate again")
        return cv2.warpPerspective(img, np.array(self.calib["H"]), tuple(self.calib["size"]))

    def rect_to_cam(self, pts):
        if not self.calib:
            return pts
        Hinv = np.linalg.inv(np.array(self.calib["H"]))
        p = cv2.perspectiveTransform(np.float32(pts).reshape(-1, 1, 2), Hinv)
        return p.reshape(-1, 2).tolist()

    def analyze(self, img, frame="frame", rectified=False):
        cfg = dict(self.cfg, px_per_mm=self.ppm)
        align = None
        if not rectified:
            img, align = self.align(img)
        rect = img if rectified else self.rectify(img)
        mask, diff, thr = segment(rect, cfg, self.bg, self.exclude, self.noise)
        objs = measure(rect, mask, cfg)
        if self.roi:                                     # touching the ROI border = partial item
            x0, y0, x1, y1 = self.roi
            for o in objs:
                x, y, w, h = o["bbox_px"]
                if x <= x0 + 1 or y <= y0 + 1 or x + w >= x1 - 1 or y + h >= y1 - 1:
                    o["edge"] = True
        add_clearance(objs, mask.shape, cfg["px_per_mm"])
        self.clf.predict(objs, cfg["use_color"])
        objs.sort(key=lambda o: (round(o["centroid_px"][1] / (cfg["px_per_mm"] * 10)), o["centroid_px"][0]))
        ppm = cfg["px_per_mm"]
        cam_picks = self.rect_to_cam([o["pick_px_rect"] for o in objs]) if objs else []
        for i, o in enumerate(objs):
            o["id"] = f"obj_{i:04d}"
            o["fraction"] = FRACTION.get(o["class"], "unknown")
            reasons = []
            if o["edge"]:
                reasons.append("edge")
            if o["confidence"] < cfg["min_conf"]:
                reasons.append("uncertain")
            if o["area_mm2"] > CLASS_MAX_AREA.get(o["class"], 1e9) * 1.3:
                reasons.append("clump")
            if o["clearance_mm"] is not None and o["clearance_mm"] < cfg["clearance_mm"]:
                reasons.append("crowded")
            if o["length_mm"] > cfg["max_pick_len_mm"]:
                reasons.append("too_big")
            o["pickable"] = not reasons
            o["reason"] = ",".join(reasons) if reasons else None   # trick 5: when in doubt, don't pick
            o["pick_mm"] = [o["pick_px_rect"][0] / ppm, o["pick_px_rect"][1] / ppm]
            o["centroid_mm"] = [o["centroid_px"][0] / ppm, o["centroid_px"][1] / ppm]
            o["pick_px"] = [round(v, 1) for v in cam_picks[i]]
        for o in objs:
            o["nearest"] = objs[o["_nearest"]]["id"] if o.get("_nearest") is not None else None
        result = {
            "frame": frame,
            "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "calibrated": bool(self.calib),
            "mm_per_px": 1.0 / ppm,
            "classifier": self.clf.kind,
            "threshold": float(thr),
            "roi_px": self.roi,
            "align": align,
            "summary": summarize(objs),
            "objects": [clean(o) for o in objs],
        }
        return result, rect, mask, objs


def clean(o):
    out = {}
    for k, v in o.items():
        if k.startswith("_"):
            continue
        if isinstance(v, float):
            v = round(v, 3)
        elif isinstance(v, list):
            v = [round(x, 2) if isinstance(x, float) else x for x in v]
        out[k] = v
    return out


def summarize(objs):
    by_class, by_frac = {}, {}
    for o in objs:
        by_class[o["class"]] = by_class.get(o["class"], 0) + 1
        by_frac[o["fraction"]] = by_frac.get(o["fraction"], 0) + 1
    n = len(objs)
    dom = max(by_frac, key=by_frac.get) if by_frac else None
    return {
        "count": n,
        "pickable": sum(o["pickable"] for o in objs),
        "by_class": by_class,
        "by_fraction": by_frac,
        "dominant_fraction": dom,
        "purity": round(by_frac[dom] / n, 3) if n else None,
        "isolated_pct": round(100 * sum((o["clearance_mm"] or 1e9) >= 3 for o in objs) / n, 1) if n else None,
    }


def purity(result, fraction, screen_mm=None):
    """Sorter purity for one fraction + 'done' signal (trick 7)."""
    objs = result["objects"]
    n = len(objs)
    good = sum(o["fraction"] == fraction for o in objs)
    out = {"fraction": fraction, "count": n, "in_fraction": good,
           "purity": round(good / n, 3) if n else None,
           "contaminants": {}}
    for o in objs:
        if o["fraction"] != fraction:
            out["contaminants"][o["class"]] = out["contaminants"].get(o["class"], 0) + 1
    if screen_mm:
        out["screen_mm"] = screen_mm
        out["passable_still_on_screen"] = sum(o["width_mm"] < screen_mm for o in objs)
    return out


# --------------------------------------------------------------------------- overlay
def draw_overlay(rect, objs, result, cfg=None, highlight=None, bar=True):
    vis = rect.copy()
    H, W = vis.shape[:2]
    s = max(W, H) / 1500.0
    fs, th = 0.45 * s, max(1, int(round(1.5 * s)))
    ppm = 1.0 / result["mm_per_px"]
    nozzle_r = int((cfg or DEFAULT_CFG)["nozzle_mm"] * ppm / 2)
    if result.get("roi_px"):
        x0, y0, x1, y1 = result["roi_px"]
        cv2.rectangle(vis, (x0, y0), (x1 - 1, y1 - 1), (90, 90, 90), th)
    for o in objs:
        col = CLASS_COLOR.get(o["class"], (160, 160, 160))
        cv2.drawContours(vis, [o["_contour"]], -1, col, th)
        px, py = o["pick_px_rect"]
        if o["pickable"]:
            g = (0, 220, 0)
            cv2.circle(vis, (px, py), max(nozzle_r, 3), g, th)
            cv2.line(vis, (px - 3 * nozzle_r, py), (px + 3 * nozzle_r, py), g, th)
            cv2.line(vis, (px, py - 3 * nozzle_r), (px, py + 3 * nozzle_r), g, th)
        else:
            r = max(nozzle_r, 4)
            cv2.line(vis, (px - r, py - r), (px + r, py + r), (0, 0, 255), th)
            cv2.line(vis, (px - r, py + r), (px + r, py - r), (0, 0, 255), th)
        if highlight and o["id"] == highlight:
            x, y, w, h = o["bbox_px"]
            cv2.rectangle(vis, (x - 6, y - 6), (x + w + 6, y + h + 6), (0, 255, 255), th * 2)
        x, y, w, h = o["bbox_px"]
        label = f"{o['id'][-3:]} {o['class'][:6]} {o['length_mm']:.1f}x{o['width_mm']:.1f}"
        cv2.putText(vis, label, (x, max(y - 4, 10)), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), th + 2)
        cv2.putText(vis, label, (x, max(y - 4, 10)), cv2.FONT_HERSHEY_SIMPLEX, fs, col, th)
    if bar:
        draw_summary(vis, result)
    return vis


def draw_summary(vis, result):
    """Black info bar at the top: counts per category."""
    H, W = vis.shape[:2]
    s = max(W, H) / 1500.0
    th = max(1, int(round(1.5 * s)))
    sm = result["summary"]
    lines = [f"n={sm['count']} pickable={sm['pickable']} purity={sm['purity']} ({sm['dominant_fraction']})  "
             f"[{result['classifier']}{'' if result['calibrated'] else ', UNCALIBRATED'}]",
             "  ".join(f"{k}:{v}" for k, v in sorted(sm["by_class"].items()))]
    bar = int(28 * s) * len(lines) + 8
    cv2.rectangle(vis, (0, 0), (W, bar), (0, 0, 0), -1)
    for i, t in enumerate(lines):
        cv2.putText(vis, t, (6, int(22 * s) + i * int(28 * s)), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * s, (255, 255, 255), th)
    return vis


def save_crops(rect, objs, outdir, pad=10):
    os.makedirs(outdir, exist_ok=True)
    for o in objs:
        x, y, w, h = o["bbox_px"]
        crop = rect[max(y - pad, 0):y + h + pad, max(x - pad, 0):x + w + pad]
        path = os.path.join(outdir, f"{o['id']}.png")
        cv2.imwrite(path, crop)
        o["crop"] = path


# --------------------------------------------------------------------------- verification (trick 6)
def match(before, after, max_mm=2.0):
    """Greedy nearest-centroid matching. Returns {before_id: (after_id, dist_mm)}."""
    pairs = []
    for b in before:
        for a in after:
            d = math.dist(b["centroid_mm"], a["centroid_mm"])
            if d <= max_mm and 0.5 < a["area_mm2"] / max(b["area_mm2"], 1e-3) < 2.0:
                pairs.append((d, b["id"], a["id"]))
    pairs.sort()
    used_a, out = set(), {}
    for d, bi, ai in pairs:
        if bi in out or ai in used_a:
            continue
        out[bi] = (ai, d)
        used_a.add(ai)
    return out


def verify(before_res, after_res, target_id, move_mm=0.8):
    before, after = before_res["objects"], after_res["objects"]
    m = match(before, after)
    removed = [b["id"] for b in before if b["id"] not in m]
    moved = [bi for bi, (ai, d) in m.items() if d > move_mm]
    matched_after = {ai for ai, _ in m.values()}
    appeared = [a["id"] for a in after if a["id"] not in matched_after]
    if target_id not in removed:
        verdict = "MISSED"
    elif len(removed) > 1:
        verdict = "DOUBLE"
    elif moved or appeared:
        verdict = "DISTURBED"
    else:
        verdict = "PICK_OK"
    return {"verdict": verdict, "target": target_id, "removed": removed,
            "moved": moved, "appeared": appeared}


def plate_wells(n):
    rows, cols = {12: (3, 4), 24: (4, 6), 48: (6, 8), 96: (8, 12)}[int(n)]
    return [f"{chr(65 + r)}{c + 1}" for r in range(rows) for c in range(cols)]
