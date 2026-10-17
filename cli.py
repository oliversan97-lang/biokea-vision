#!/usr/bin/env python3
"""Offline CLI (runs on the Pi or a laptop).

  python3 cli.py markers                         # printable Letter sheet -> markers_letter.png/.pdf
  python3 cli.py calibrate img.jpg [--marker-mm 20]   # scale from printed marker size (markers anywhere)
  python3 cli.py calibrate img.jpg --w 150 --h 100   # or: 4 corner markers, inner-corner distances in mm
  python3 cli.py background empty_tray.jpg       # store empty-tray reference (best segmentation)
  python3 cli.py detect img.jpg [--crops]        # -> results/<name>.json + _overlay.jpg
  python3 cli.py train data/singles              # folders named by class -> model.joblib
  python3 cli.py verify before.jpg after.jpg --target obj_0003
  python3 cli.py purity img.jpg --fraction rice --screen-mm 2.0
"""
import argparse
import glob
import json
import math
import os
import sys

import cv2
import numpy as np

import vision as V

ROOT = os.path.dirname(os.path.abspath(__file__))
# Generous physical length limits per category (mm). Blobs outside are artefacts in a single-food photo
# (marker-paper slivers, piles), not label truth: e.g. a 10 mm "chia" is impossible.
SANITY_LEN_MM = {"seed": (0.5, 3.0), "couscous": (0.8, 4.5), "rice": (3.0, 14.0), "noodle": (6.0, 60.0)}


def make_marker_sheet(base, dpi=300):
    """US-Letter sheet of unique ArUco markers at exact physical size + 100 mm scale bar."""
    mm = dpi / 25.4
    W, H = int(8.5 * dpi), int(11 * dpi)
    sheet = np.full((H, W), 255, np.uint8)
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    P = lambda v: int(round(v * mm))

    def text(s, x, y, scale=1.0, th=2):
        cv2.putText(sheet, s, (P(x), P(y)), cv2.FONT_HERSHEY_SIMPLEX, scale * dpi / 300, 0,
                    max(1, int(th * dpi / 300)), cv2.LINE_AA)

    def dashed_rect(x0, y0, x1, y1, dash=2.0):
        for xa, ya, xb, yb in ((x0, y0, x1, y0), (x1, y0, x1, y1), (x1, y1, x0, y1), (x0, y1, x0, y0)):
            n = max(2, int(math.hypot(xb - xa, yb - ya) / dash))
            for k in range(0, n, 2):
                t0, t1 = k / n, (k + 1) / n
                cv2.line(sheet, (P(xa + (xb - xa) * t0), P(ya + (yb - ya) * t0)),
                         (P(xa + (xb - xa) * t1), P(ya + (yb - ya) * t1)), 120, max(1, dpi // 150))

    text("BioKEA tray markers | ArUco DICT_4X4_50 | PRINT AT 100% (no scaling)", 15, 20, 1.15, 3)
    text("Cut on the dashed lines (keep the white border). Tape one marker flat at each corner of", 15, 29, 0.8, 2)
    text("the imaging area, at tray-surface height. Measure inner-corner to inner-corner W x H (mm).", 15, 35, 0.8, 2)
    y, margin, gap = 45, 6, 8
    for title, ids, size in (("A - use these (20 mm)", range(0, 4), 20),
                             ("B - spares (20 mm)", range(4, 8), 20),
                             ("C - small, for tight spaces (12 mm)", range(8, 12), 12)):
        text(title, 15, y + 5, 0.9, 2)
        y += 9
        x = 15
        for i in ids:
            px = P(size)
            X, Y = P(x + margin), P(y + margin)
            sheet[Y:Y + px, X:X + px] = cv2.aruco.generateImageMarker(d, i, px)
            dashed_rect(x, y, x + size + 2 * margin, y + size + 2 * margin)
            x += size + 2 * margin + gap
        y += size + 2 * margin + 12
    # 100 mm scale bar to check that the printer did not rescale
    y += 4
    cv2.line(sheet, (P(15), P(y)), (P(115), P(y)), 0, max(2, dpi // 100))
    for k in range(11):
        cv2.line(sheet, (P(15 + 10 * k), P(y - (4 if k % 5 == 0 else 2.5))), (P(15 + 10 * k), P(y)), 0, max(2, dpi // 150))
    text("100 mm scale check. If it is off, that is OK: just measure the taped markers.", 15, y + 7, 0.75, 2)
    cv2.imwrite(base + ".png", sheet)
    try:
        from PIL import Image
        Image.fromarray(sheet).convert("L").save(base + ".pdf", "PDF", resolution=float(dpi))
    except ImportError:
        print("PIL missing: only PNG written")
    return base


def run_detect(tv, path, crops=False, outdir=None):
    img = cv2.imread(path)
    if img is None:
        sys.exit(f"cannot read {path}")
    name = os.path.splitext(os.path.basename(path))[0]
    res, rect, mask, objs = tv.analyze(img, frame=os.path.basename(path))
    outdir = outdir or os.path.join(ROOT, "results")
    os.makedirs(outdir, exist_ok=True)
    if crops:
        V.save_crops(rect, objs, os.path.join(outdir, name + "_crops"))
        res["objects"] = [V.clean(o) for o in objs]
    cv2.imwrite(os.path.join(outdir, name + "_overlay.jpg"), V.draw_overlay(rect, objs, res, tv.cfg))
    cv2.imwrite(os.path.join(outdir, name + "_mask.png"), mask)
    with open(os.path.join(outdir, name + ".json"), "w") as f:
        json.dump(res, f, indent=1)
    return res, rect, objs


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    m = sp.add_parser("markers"); m.add_argument("--dpi", type=int, default=300)
    c = sp.add_parser("calibrate"); c.add_argument("img"); c.add_argument("--w", type=float); c.add_argument("--h", type=float); c.add_argument("--ppm", type=float, default=10); c.add_argument("--marker-mm", type=float, default=None)
    b = sp.add_parser("background"); b.add_argument("img")
    d = sp.add_parser("detect"); d.add_argument("imgs", nargs="+"); d.add_argument("--crops", action="store_true")
    t = sp.add_parser("train"); t.add_argument("folder")
    v = sp.add_parser("verify"); v.add_argument("before"); v.add_argument("after"); v.add_argument("--target")
    p = sp.add_parser("purity"); p.add_argument("img"); p.add_argument("--fraction", required=True); p.add_argument("--screen-mm", type=float)
    a = ap.parse_args()
    tv = V.TrayVision(ROOT)

    if a.cmd == "markers":
        base = os.path.join(ROOT, "markers_letter")
        make_marker_sheet(base, a.dpi)
        print(f"{base}.png/.pdf: Letter sheet, set A ids 0-3 (20 mm), spares 4-7 (20 mm), small 8-11 (12 mm). "
              "Print at 100%, cut on dashed lines, tape one at each tray corner, measure inner-corner W x H.")

    elif a.cmd == "calibrate":
        img = cv2.imread(a.img)
        cal = V.calibrate(img, a.w, a.h, a.ppm) if a.w and a.h else V.calibrate_marker_size(img, a.marker_mm)
        json.dump(cal, open(os.path.join(ROOT, "calib.json"), "w"), indent=1)
        print(json.dumps({k: cal.get(k) for k in ("method", "size", "marker_ids", "native_px_per_mm", "width_mm", "height_mm")}))
        if cal["native_px_per_mm"] < 10:
            print("WARNING: camera gives <10 px/mm; chia (~1 mm) will be marginal. Move camera closer.")

    elif a.cmd == "background":
        cv2.imwrite(os.path.join(ROOT, "bg.png"), tv.rectify(cv2.imread(a.img)))
        print("saved bg.png (re-save after any calibration or lighting change)")

    elif a.cmd == "detect":
        for path in a.imgs:
            res, _, _ = run_detect(tv, path, a.crops)
            print(path, json.dumps(res["summary"]))

    elif a.cmd == "train":
        X, y = [], []
        for folder_cls in sorted(os.listdir(a.folder)):
            cls = V.LABEL_MERGE.get(folder_cls, folder_cls)      # white/wild rice -> rice
            files = sorted(glob.glob(os.path.join(a.folder, folder_cls, "*.jp*g")) + glob.glob(os.path.join(a.folder, folder_cls, "*.png")))
            n0 = len(y)
            dropped = 0
            for f in files:
                jf = f.rsplit(".", 1)[0] + ".json"      # detections frozen at capture time (cache_feats.py)
                if os.path.exists(jf):
                    objs = json.load(open(jf))["objects"]
                else:
                    res, rect, mask, objs = tv.analyze(cv2.imread(f), frame=f)
                good = [o for o in objs if not o["edge"]]
                if not good:
                    continue
                # dominant item size = area-weighted median (big items win over specks / wall noise)
                areas = np.sort([o["area_mm2"] for o in good])
                med = float(areas[np.searchsorted(np.cumsum(areas), areas.sum() / 2)])
                for o in good:   # trick 1: free labels; drop debris (<1/4 dominant) and clumps (>3x)
                    lo, hi = SANITY_LEN_MM.get(cls, (0, 1e9))
                    if med / 4 <= o["area_mm2"] <= med * 3 and lo <= o["length_mm"] <= hi:
                        X.append([o[k] for k in V.FEATURES]); y.append(cls)
                    else:
                        dropped += 1
            print(f"{cls}: {len(files)} images, {len(y) - n0} blobs kept, {dropped} dropped (debris/clumps)")
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        from sklearn.model_selection import StratifiedKFold, cross_val_predict
        from sklearn.tree import DecisionTreeClassifier, export_text
        import joblib
        X, y = np.array(X, np.float32), np.array(y)
        size_idx = [V.FEATURES.index(f) for f in V.SIZE_FEATURES]
        cv = StratifiedKFold(5, shuffle=True, random_state=0)
        cands = {   # the brief: decide the category from SIZE; colour model only for comparison
            "size_tree": (DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=0), size_idx),
            "size_rf": (RandomForestClassifier(300, min_samples_leaf=2, class_weight="balanced", random_state=0), size_idx),
            "size+colour_rf": (RandomForestClassifier(300, min_samples_leaf=2, class_weight="balanced", random_state=0),
                               list(range(len(V.FEATURES)))),
        }
        labels = [c for c in V.CLASSES if c in set(y)]
        report, scores = [], {}
        for name, (mdl, idx) in cands.items():
            pred = cross_val_predict(mdl, X[:, idx], y, cv=cv)
            scores[name] = accuracy_score(y, pred)
            report.append(f"== {name}: 5-fold accuracy {scores[name]:.3f}\n" + classification_report(y, pred, digits=3)
                          + f"confusion (rows=true, cols=pred) {labels}\n{confusion_matrix(y, pred, labels=labels)}\n")
        best = max(("size_tree", "size_rf"), key=lambda k: scores[k])
        mdl, idx = cands[best]
        mdl.fit(X[:, idx], y)
        tree = cands["size_tree"][0].fit(X[:, size_idx], y)
        rules = export_text(tree, feature_names=V.SIZE_FEATURES, decimals=2)
        report.append(f"== chosen: {best}\n== size rules (depth-3 tree, the 'virtual sieve'):\n{rules}")
        joblib.dump({"model": mdl, "features": [V.FEATURES[i] for i in idx], "name": best},
                    os.path.join(ROOT, "model.joblib"))
        text = f"samples per class: { {c: int((y == c).sum()) for c in labels} }\n" + "\n".join(report)
        open(os.path.join(ROOT, "train_report.txt"), "w").write(text)
        print(text)
        print("saved model.joblib + train_report.txt")

    elif a.cmd == "verify":
        rb, _, _ = run_detect(tv, a.before)
        ra, _, _ = run_detect(tv, a.after)
        target = a.target or next((o["id"] for o in rb["objects"] if o["pickable"]), None)
        print(json.dumps(V.verify(rb, ra, target), indent=1))

    elif a.cmd == "purity":
        res, _, _ = run_detect(tv, a.img)
        print(json.dumps(V.purity(res, a.fraction, a.screen_mm), indent=1))


if __name__ == "__main__":
    main()
