#!/usr/bin/env python3
"""Live web UI on the Pi:  python3 server.py   ->  open http://<pi-ip>:8000

Live preview, detect overlay, ArUco calibration, background capture, labelled
snapshots for training (trick 1), and a before/after pick-verify loop that writes
provenance JSONL (one record per well).
"""
try:
    from picamera2 import Picamera2
except Exception:                                                   # laptop: no camera stack
    Picamera2 = None
import json
import logging
import os
import queue
import threading
import time

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request, send_file

import vision as V
from leds import Bulbs

ROOT = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)
tv = V.TrayVision(ROOT)


class _QuietLog(logging.Filter):      # the page polls /api/state every 2 s; keep server.log readable
    def filter(self, rec):
        msg = rec.getMessage()
        return "/api/state" not in msg and "/stream" not in msg


logging.getLogger("werkzeug").addFilter(_QuietLog())


@app.errorhandler(Exception)
def on_error(e):                      # show the reason in the UI instead of a bare HTTP 500
    import traceback
    traceback.print_exc()
    return jsonify({"error": f"{type(e).__name__}: {e}"}), 500


# --------------------------------------------------------------------------- camera
class Camera:
    """One background thread owns Picamera2; every Flask thread only reads shared frames.

    v1 called Picamera2 from several request threads at once (live stream + lock
    exposure + snapshot) and the camera hung. Now only _loop() touches self.cam.
    """

    def __init__(self, size=(1920, 1080), lores=(640, 360)):
        self.cond = threading.Condition()
        self.lores, self.main, self.md = None, None, {}
        self.seq, self.t_last, self.fps, self.error = 0, 0.0, 0.0, None
        self.main_req, self.main_got = 0, 0
        self.jobs = queue.Queue()
        self.cam, self.fake = None, None
        try:
            if Picamera2 is None:
                raise RuntimeError("picamera2 not installed")
            self.cam = Picamera2()
            # OV5647 full-res mode (2592x1944) failed to stream ~1 start in 3 on this unit (and
            # almost always at low fps). The 1920x1080 mode is a centre crop at full pixel pitch:
            # same px/mm, smaller field, and it streamed 15/15 trials.
            cfg = self.cam.create_video_configuration(
                main={"size": size, "format": "RGB888"},             # numpy array is BGR -> OpenCV-ready
                lores={"size": lores, "format": "YUV420"},
                sensor={"output_size": size, "bit_depth": 10},
                controls={"FrameDurationLimits": (66666, 66666)}, buffer_count=4)   # 15 fps
            self.cam.configure(cfg)
            self.cam.start()
            threading.Thread(target=self._loop, daemon=True).start()
            threading.Thread(target=self._watchdog, daemon=True).start()
        except Exception as e:                                      # laptop / camera busy: serve a file
            print("camera unavailable, using test image:", e, flush=True)
            self.cam = None
            self.fake = cv2.imread(os.environ.get("FAKE_IMG", os.path.join(ROOT, "test.jpg")))

    def _watchdog(self, first_s=6.0, stall_s=5.0):
        """Picamera2 occasionally stops delivering frames. Exit so supervise.sh restarts us (~5 s)."""
        t0 = time.time()
        while True:
            time.sleep(1.0)
            if self.t_last == 0 and time.time() - t0 > first_s:
                why = "no first frame"
            elif self.t_last and time.time() - self.t_last > stall_s:
                why = "frames stopped"
            else:
                continue
            print(f"WATCHDOG: camera {why}; exiting so supervise.sh restarts the server", flush=True)
            os._exit(3)

    def apply_controls(self, ctl):
        c = {k: (tuple(v) if isinstance(v, list) else v) for k, v in ctl.items()}
        return self.run(lambda: self.cam.set_controls(c))

    def _loop(self):
        skip = False
        while True:
            try:
                while not self.jobs.empty():                        # control changes, run in this thread
                    fn, box, ev = self.jobs.get_nowait()
                    try:
                        box["result"] = fn()
                    except Exception as e:
                        box["error"] = repr(e)
                    ev.set()
                with self.cond:
                    target = self.main_req if self.main_req > self.main_got else 0
                req = self.cam.capture_request()
                main = None
                try:
                    lores, md = req.make_array("lores"), req.get_metadata()
                    if target:
                        if not skip:
                            skip = True                             # drop 1 frame: image must postdate request
                        else:
                            main, skip = req.make_array("main"), False
                finally:
                    req.release()
                now = time.time()
                with self.cond:
                    if self.t_last:
                        self.fps = 0.9 * self.fps + 0.1 / max(now - self.t_last, 1e-3)
                    self.t_last, self.lores, self.md = now, lores, md
                    self.seq += 1
                    if main is not None:
                        self.main, self.main_got = main, target
                    self.cond.notify_all()
            except Exception as e:
                self.error = repr(e)
                print("camera loop error:", e, flush=True)
                time.sleep(0.5)

    def frame(self, timeout=10.0):
        """Fresh full-res BGR frame taken after this call started."""
        if self.cam is None:
            return self.fake.copy()
        with self.cond:
            self.main_req += 1
            want = self.main_req
            if not self.cond.wait_for(lambda: self.main_got >= want, timeout):
                raise RuntimeError("camera timeout")
            return self.main

    def preview(self, last_seq=None, timeout=5.0):
        if self.cam is None:
            time.sleep(0.3)
            return cv2.resize(self.fake, (640, 480)), 0
        with self.cond:
            self.cond.wait_for(lambda: self.lores is not None and self.seq != last_seq, timeout)
            yuv, seq = self.lores, self.seq
        return cv2.cvtColor(yuv, cv2.COLOR_YUV420p2BGR), seq

    def run(self, fn, timeout=5.0):
        box, ev = {}, threading.Event()
        self.jobs.put((fn, box, ev))
        if not ev.wait(timeout):
            raise RuntimeError("camera job timeout")
        if "error" in box:
            raise RuntimeError(box["error"])
        return box.get("result")

    def lock_exposure(self, on=True):
        if self.cam is None:
            return {}

        def job():
            if on:
                md = self.md
                ctl = {"AeEnable": False, "AwbEnable": False, "ExposureTime": md["ExposureTime"],
                       "AnalogueGain": md["AnalogueGain"], "ColourGains": md["ColourGains"]}
                self.cam.set_controls(ctl)
                return {k: (list(v) if isinstance(v, tuple) else v) for k, v in ctl.items()}
            self.cam.set_controls({"AeEnable": True, "AwbEnable": True})
            return {"AeEnable": True, "AwbEnable": True}
        return self.run(job)

    def status(self):
        if self.cam is None:
            return {"ok": False, "mode": "test-image"}
        with self.cond:
            yuv, md, t = self.lores, dict(self.md), self.t_last
        age = time.time() - t if t else None
        sharp = None
        if yuv is not None:     # variance of Laplacian on the centre of the Y plane: higher = sharper
            hh, ww = yuv.shape[0] * 2 // 3, yuv.shape[1]
            y = yuv[:hh][hh // 4:3 * hh // 4, ww // 4:3 * ww // 4]
            sharp = round(float(cv2.Laplacian(y, cv2.CV_64F).var()), 1)
        return {"ok": age is not None and age < 3, "fps": round(self.fps, 1),
                "last_frame_age_s": None if age is None else round(age, 2), "sharpness": sharp,
                "exposure_us": md.get("ExposureTime"), "gain": round(md.get("AnalogueGain", 0), 2),
                "error": self.error}


cam = Camera()
state = {"last": None, "before": None, "target": None, "run_id": time.strftime("run_%Y%m%d_%H%M%S"),
         "well_idx": 0, "locked": False, "lock_ctl": None}
STATE_FILE = os.path.join(ROOT, "state.json")
PERSIST = ("run_id", "well_idx", "before", "target", "locked", "lock_ctl")
state.update({k: v for k, v in (V.load_json(STATE_FILE, {}) or {}).items() if k in PERSIST})


def save_state():
    """Survive watchdog restarts: run id, well counter, pending BEFORE and exposure lock."""
    with open(STATE_FILE + ".tmp", "w") as f:
        json.dump({k: state[k] for k in PERSIST}, f)
    os.replace(STATE_FILE + ".tmp", STATE_FILE)


if state["locked"] and state["lock_ctl"] and cam.cam is not None:    # re-apply exposure lock after a restart
    def _relock():
        try:
            cam.apply_controls(state["lock_ctl"])
            print("re-applied exposure lock", flush=True)
        except Exception as e:
            print("could not re-apply exposure lock:", e, flush=True)
    threading.Thread(target=_relock, daemon=True).start()

CAP = os.path.join(ROOT, "captures")
RES = os.path.join(ROOT, "results")
for d in (CAP, RES):
    os.makedirs(d, exist_ok=True)

bulbs = Bulbs(tv.cfg.get("bulb_pins"), active_high=tv.cfg.get("bulb_active_high", True))
live = {"on": bool(tv.cfg.get("live_on_start", False)), "jpg": None, "summary": None, "t": 0.0, "fps": 0.0}


def camera_view(ov, res, frame=None):
    """Draw results where things really are: undo the marker alignment (calibration pose -> camera)."""
    M = (res.get("align") or {}).get("M")
    if not M or tv.calib.get("roi") != "frame":
        return ov
    h, w = ov.shape[:2]
    back = cv2.warpAffine(ov, np.array(M), (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    if frame is not None and frame.shape == ov.shape:    # fill uncovered corners with the live frame
        holes = cv2.warpAffine(np.ones((h, w), np.uint8), np.array(M), (w, h),
                               flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP, borderValue=0) == 0
        back[holes] = frame[holes]
    return back


def live_loop():
    """Live mode: detect continuously, keep the newest overlay in memory, drive the bulbs.

    A bulb turns on only when its category is seen in 2 consecutive frames (no flicker from noise).
    """
    hist = []
    while True:
        if not live["on"]:
            hist = []
            time.sleep(0.2)
            continue
        try:
            t0 = time.time()
            frame = cam.frame()
            res, rect, mask, objs = tv.analyze(frame, frame="live")
            hist = (hist + [present_categories(res)])[-2:]
            cats = set.intersection(*hist) if len(hist) == 2 else set()
            res["bulbs"] = bulbs.show(cats)
            ov = V.draw_summary(camera_view(V.draw_overlay(rect, objs, res, tv.cfg, bar=False), res, frame), res)
            h, w = ov.shape[:2]
            ok, jpg = cv2.imencode(".jpg", cv2.resize(ov, (w * 2 // 3, h * 2 // 3)), [cv2.IMWRITE_JPEG_QUALITY, 75])
            live.update(jpg=jpg.tobytes(), summary=res["summary"], t=time.time(),
                        fps=round(1.0 / max(time.time() - t0, 1e-3), 2), lit=sorted(cats))
        except Exception as e:
            live.update(summary={"error": repr(e)})
            time.sleep(1.0)


threading.Thread(target=live_loop, daemon=True).start()


def capture(tag="frame"):
    img = cam.frame()
    name = f"{time.strftime('%Y%m%d_%H%M%S')}_{int(time.time() * 1000) % 1000:03d}_{tag}"
    path = os.path.join(CAP, name + ".jpg")
    cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return img, name, path


def present_categories(res):
    """Categories to light: confident, whole (non-edge), not-a-speck items on the tray."""
    mc, ma = tv.cfg.get("bulb_min_conf", 0.5), tv.cfg.get("bulb_min_area_mm2", 0.6)
    return {o["class"] for o in res["objects"]
            if not o["edge"] and o["confidence"] >= mc and o["area_mm2"] >= ma and o["class"] in V.CLASSES}


def detect(tag="frame", highlight=None):
    img, name, path = capture(tag)
    t0 = time.time()
    res, rect, mask, objs = tv.analyze(img, frame=os.path.relpath(path, ROOT))
    res["elapsed_s"] = round(time.time() - t0, 2)
    res["bulbs"] = bulbs.show(present_categories(res))
    ov = V.draw_summary(camera_view(V.draw_overlay(rect, objs, res, tv.cfg, highlight=highlight, bar=False), res, img), res)
    cv2.imwrite(os.path.join(RES, name + "_overlay.jpg"), ov, [cv2.IMWRITE_JPEG_QUALITY, 85])
    with open(os.path.join(RES, name + ".json"), "w") as f:
        json.dump(res, f, indent=1)
    res["overlay"] = name + "_overlay.jpg"
    state["last"] = res
    return res


# --------------------------------------------------------------------------- routes
@app.route("/stream")
@app.route("/stream.mjpg")
def stream():
    def gen():
        seq = None
        while True:
            img, seq = cam.preview(seq)
            ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
            yield b"--f\r\nContent-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n"
            time.sleep(0.08)                                        # ~10 fps is plenty for a preview
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=f")


@app.route("/snapshot.jpg")
def snapshot():
    """Full-res raw frame (?w=1296 to downscale). Handy for checking focus/framing."""
    img = cam.frame()
    w = int(request.args.get("w", 0))
    if 0 < w < img.shape[1]:
        img = cv2.resize(img, (w, int(img.shape[0] * w / img.shape[1])), interpolation=cv2.INTER_AREA)
    ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return Response(jpg.tobytes(), mimetype="image/jpeg")


@app.route("/results/<path:name>")
def results_file(name):
    return send_file(os.path.join(RES, name))


@app.route("/api/detect", methods=["POST", "GET"])
def api_detect():
    return jsonify(detect())


@app.route("/api/live", methods=["POST"])
def api_live():
    live["on"] = request.args.get("on", "1") == "1"
    if not live["on"]:
        bulbs.off()
    return jsonify({"live": live["on"]})


@app.route("/live.jpg")
def live_jpg():
    if not live["jpg"]:
        return jsonify({"error": "live mode is off"}), 404
    return Response(live["jpg"], mimetype="image/jpeg")


@app.route("/api/live_state")
def api_live_state():
    return jsonify({k: live.get(k) for k in ("on", "summary", "fps", "lit")} | {"age_s": round(time.time() - live["t"], 1)})


@app.route("/api/bulbs", methods=["POST"])
def api_bulbs():
    """?test=1 runs each bulb in turn; ?on=seed,rice lights those; ?on= turns all off."""
    if request.args.get("test") == "1":
        return jsonify(bulbs.test())
    return jsonify(bulbs.show({c for c in request.args.get("on", "").split(",") if c}))


@app.route("/api/lock", methods=["POST"])
def api_lock():
    on = request.args.get("on", "1") == "1"
    ctl = cam.lock_exposure(on)
    state.update(locked=on, lock_ctl=ctl if on else None)
    save_state()
    return jsonify({"locked": on, "controls": ctl})


@app.route("/api/new_run", methods=["POST"])
def api_new_run():
    state.update(run_id=time.strftime("run_%Y%m%d_%H%M%S"), well_idx=0, before=None, target=None)
    save_state()
    return jsonify({"run_id": state["run_id"], "next_well": next_well()})


@app.route("/api/calibrate", methods=["POST"])
def api_calibrate():
    """No W/H -> scale from the printed marker size (markers anywhere). W+H -> 4-corner homography."""
    w, h = request.args.get("w"), request.args.get("h")
    img, name, _ = capture("calib")
    try:
        if w and h:
            cal = V.calibrate(img, float(w), float(h), float(request.args.get("ppm", 10)))
        else:
            mm = request.args.get("marker_mm")
            cal = V.calibrate_marker_size(img, float(mm) if mm else None, roi=request.args.get("roi", "frame"))
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    json.dump(cal, open(os.path.join(ROOT, "calib.json"), "w"), indent=1)
    if os.path.exists(os.path.join(ROOT, "bg.png")):
        os.rename(os.path.join(ROOT, "bg.png"), os.path.join(ROOT, "bg_old.png"))   # geometry changed
    tv.reload()
    return jsonify({k: cal.get(k) for k in ("method", "marker_ids", "native_px_per_mm", "width_mm", "height_mm",
                                             "size", "marker_size_spread_pct", "roi", "warning")})


@app.route("/api/background", methods=["POST"])
def api_background():
    """Empty tray: average N frames -> bg.png, per-pixel variation -> bg_noise.npy."""
    if not state["locked"]:          # background subtraction needs fixed exposure/white balance
        state.update(locked=True, lock_ctl=cam.lock_exposure(True))
        save_state()
        time.sleep(0.5)
    n = int(request.args.get("n", 8))
    frames = [tv.rectify(capture("bg")[0] if i == 0 else cam.frame()).astype(np.float32) for i in range(n)]
    mean = np.mean(frames, axis=0)
    mean_u8 = np.clip(mean + 0.5, 0, 255).astype(np.uint8)
    ref = cv2.cvtColor(cv2.GaussianBlur(mean_u8, (0, 0), 1.0), cv2.COLOR_BGR2LAB).astype(np.float32)
    var = np.zeros(mean.shape[:2], np.float32)
    for f in frames:
        lab = cv2.cvtColor(cv2.GaussianBlur(f.astype(np.uint8), (0, 0), 1.0), cv2.COLOR_BGR2LAB).astype(np.float32)
        var += np.sum((lab - ref) ** 2, axis=2)
    noise = cv2.dilate(np.sqrt(var / n), np.ones((5, 5), np.uint8))       # a little spatial slack
    painted = 0
    if request.args.get("inpaint") == "1":
        # Tray not empty: find small compact items vs a local median and paint them out.
        cfg = dict(tv.cfg, px_per_mm=tv.ppm, threshold=float(request.args.get("thr", 8)), half_max=0)
        m, _, _ = V.segment(mean_u8, cfg, None, tv.exclude, None)
        k, lab, stats, _ = cv2.connectedComponentsWithStats(m)
        hole = np.zeros_like(m)
        max_px = float(request.args.get("max_mm2", 30)) * tv.ppm ** 2
        for i in range(1, k):
            if stats[i, cv2.CC_STAT_AREA] > max_px:
                continue
            ys, xs = np.nonzero(lab == i)
            (_, _), (rw, rh), _ = cv2.minAreaRect(np.column_stack([xs, ys]).astype(np.float32))
            if max(rw, rh) / max(min(rw, rh), 1.0) < 3.5:                  # compact: seed/grain, not a wall edge
                hole[lab == i] = 255
                painted += 1
        r = int(1.0 * tv.ppm) * 2 + 1
        hole = cv2.dilate(hole, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r, r)))
        mean_u8 = cv2.inpaint(mean_u8, hole, 7, cv2.INPAINT_TELEA)
    cv2.imwrite(os.path.join(ROOT, "bg.png"), mean_u8)
    np.save(os.path.join(ROOT, "bg_noise.npy"), noise.astype(np.float16))
    tv.reload()
    return jsonify({"ok": True, "frames": n, "exposure_locked": state["locked"], "items_painted_out": painted,
                    "noise_median": round(float(np.median(noise)), 2), "noise_p99": round(float(np.percentile(noise, 99)), 2)})


@app.route("/api/clear_background", methods=["POST"])
def api_clear_background():
    p = os.path.join(ROOT, "bg.png")
    if os.path.exists(p):
        os.remove(p)
    tv.reload()
    return jsonify({"ok": True})


@app.route("/api/snap", methods=["POST"])
def api_snap():
    """Save a labelled raw image (data/singles/<food>/ or data/misc/<label>/) + show its detections."""
    label = request.args.get("label", "unlabeled").strip().replace("/", "_")
    sub = "singles" if (label in V.CLASSES or label in V.LABEL_MERGE) else "misc"
    d = os.path.join(ROOT, "data", sub, label)
    os.makedirs(d, exist_ok=True)
    img = cam.frame()
    stem = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(d, stem + ".jpg")
    cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    res, rect, mask, objs = tv.analyze(img, frame=os.path.relpath(path, ROOT))
    with open(path[:-4] + ".json", "w") as f:            # freeze detections with today's reference
        json.dump(res, f)
    ov = V.draw_overlay(rect, objs, res, tv.cfg)
    res["overlay"] = f"{stem}_{label}_overlay.jpg"
    cv2.imwrite(os.path.join(RES, res["overlay"]), ov, [cv2.IMWRITE_JPEG_QUALITY, 85])
    res.update(saved=os.path.relpath(path, ROOT), n_in_folder=len(os.listdir(d)))
    return jsonify(res)


@app.route("/api/before", methods=["POST"])
def api_before():
    res = detect("before")
    target = request.args.get("target") or next((o["id"] for o in res["objects"] if o["pickable"]), None)
    if target:   # redraw with the target highlighted
        res = detect_redraw(res, target)
    state.update(before=res, target=target)
    save_state()
    t = next((o for o in res["objects"] if o["id"] == target), None)
    return jsonify({"target": target, "target_obj": t, "well": next_well(), **res})


def detect_redraw(res, target):
    img = cv2.imread(os.path.join(ROOT, res["frame"]))
    r2, rect, mask, objs = tv.analyze(img, frame=res["frame"])
    ov = V.draw_overlay(rect, objs, r2, tv.cfg, highlight=target)
    cv2.imwrite(os.path.join(RES, res["overlay"]), ov, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return res


def next_well():
    wells = V.plate_wells(tv.cfg["plate"])
    return wells[state["well_idx"] % len(wells)]


@app.route("/api/after", methods=["POST"])
def api_after():
    if not state["before"]:
        return jsonify({"error": "take a BEFORE first"}), 400
    after = detect("after")
    v = V.verify(state["before"], after, state["target"])
    well = next_well()
    tgt = next((o for o in state["before"]["objects"] if o["id"] == state["target"]), {})
    rec = {"run_id": state["run_id"], "object_id": state["target"], "well": well,
           "verdict": v["verdict"], "class": tgt.get("class"), "length_mm": tgt.get("length_mm"),
           "width_mm": tgt.get("width_mm"), "pick_mm": tgt.get("pick_mm"),
           "before": state["before"]["frame"], "after": after["frame"],
           "removed": v["removed"], "moved": v["moved"], "git_sha": V.git_sha(),
           "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
    os.makedirs(os.path.join(ROOT, "runs"), exist_ok=True)
    with open(os.path.join(ROOT, "runs", state["run_id"] + ".jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")
    if v["verdict"] in ("PICK_OK", "DOUBLE", "DISTURBED"):   # something went into the well
        state["well_idx"] += 1
    state["before"] = None
    save_state()
    return jsonify({"record": rec, "verify": v, **after})


@app.route("/api/state")
def api_state():
    return jsonify({"run_id": state["run_id"], "next_well": next_well(), "locked": state["locked"],
                    "calibrated": bool(tv.calib), "background": tv.bg is not None,
                    "classifier": tv.clf.kind, "cfg": tv.cfg, "camera": cam.status(),
                    "native_px_per_mm": (tv.calib or {}).get("native_px_per_mm")})


@app.route("/api/config", methods=["POST"])
def api_config():
    cfg = V.load_json(os.path.join(ROOT, "config.json"), {}) or {}
    for k, v in request.get_json(force=True).items():
        cfg[k] = v
    json.dump(cfg, open(os.path.join(ROOT, "config.json"), "w"), indent=1)
    tv.reload()
    return jsonify(tv.cfg)


@app.route("/")
def index():
    return PAGE


PAGE = """<!doctype html><html><head><meta charset=utf-8><title>BioKEA Tray Vision</title>
<style>body{font-family:system-ui;margin:10px;background:#111;color:#eee}button{margin:2px;padding:6px 10px}
img{max-width:49%;border:1px solid #444}pre{background:#222;padding:8px;max-height:40vh;overflow:auto;font-size:12px}
input{width:70px}.row{margin:4px 0}</style></head><body>
<h3>BioKEA Tray Vision <span id=st style="font-size:13px;color:#aaa"></span></h3>
<div class=row>
 <button onclick="post('/api/lock?on=1')">Lock exposure/WB</button>
 <button onclick="post('/api/lock?on=0')">Auto exposure</button>
 | marker size <input id=mm placeholder=auto> mm
 <button onclick="post('/api/calibrate'+(mm.value?'?marker_mm='+mm.value:''))">Calibrate (no measuring)</button>
 or W <input id=cw placeholder=mm> H <input id=ch placeholder=mm>
 <button onclick="post('/api/calibrate?w='+cw.value+'&h='+ch.value)">Calibrate W&times;H</button>
 <button onclick="post('/api/background')">Save empty-tray background</button>
 <button onclick="post('/api/clear_background')">Clear bg</button>
</div>
<div class=row>
 <button onclick="det('/api/detect')" style="background:#4a4">DETECT</button>
 <label><input type=checkbox id=auto style="width:auto"> auto every 2 s</label>
 | label <select id=lab><option>chia<option>couscous<option>white_rice<option>wild_rice<option>broken_noodle<option>whole_noodle<option>mixed<option>empty</select>
 <button onclick="det('/api/snap?label='+lab.value)">Snap labelled photo</button>
 | <button onclick="det('/api/before')" style="background:#46a">1. BEFORE pick</button>
 <button onclick="det('/api/after')" style="background:#a64">2. AFTER pick &rarr; verify</button>
 <button onclick="post('/api/new_run')">New run (well A1)</button>
</div>
<div class=row>
 <button onclick="liveOn(1)" style="background:#4a4;font-weight:bold">LIVE: detect + bulbs</button>
 <button onclick="liveOn(0)">Stop live</button>
 <button onclick="post('/api/bulbs?test=1')">Test bulbs</button>
 <span id=lit style="font-size:15px;color:#ff6"></span>
</div>
<img src="/stream"> <img id=ov>
<pre id=out></pre>
<script>
async function post(u){const r=await fetch(u,{method:'POST'});const j=await r.json();out.textContent=JSON.stringify(j,null,1);state();return j}
async function det(u){const j=await post(u);if(j.overlay)ov.src='/results/'+j.overlay+'?'+Date.now();
 const s={summary:j.summary,elapsed_s:j.elapsed_s};if(j.saved)s.saved=j.saved,s.n_in_folder=j.n_in_folder;
 if(j.error)s.error=j.error;if(j.target)s.target=j.target_obj,s.next_well=j.well;
 if(j.record)s.record=j.record,s.verify=j.verify;
 s.pick_list=(j.objects||[]).filter(o=>o.pickable).map(o=>[o.id,o.class,o.pick_mm,o.clearance_mm]);
 out.textContent=JSON.stringify(s,null,1)}
async function state(){const j=await (await fetch('/api/state')).json();const c=j.camera||{};
 st.textContent=`camera ${c.ok?'OK':'NOT OK'} ${c.fps||0}fps sharpness ${c.sharpness} | run ${j.run_id} | next well ${j.next_well} | calib ${j.calibrated}(${j.native_px_per_mm} px/mm) | bg ${j.background} | locked ${j.locked} | ${j.classifier}`}
let busy=false;setInterval(async()=>{if(auto.checked&&!busy){busy=true;try{await det('/api/detect')}finally{busy=false}}},2000);setInterval(state,2000);state();
let liveTimer=null;
async function liveOn(on){await post('/api/live?on='+on);if(liveTimer){clearInterval(liveTimer);liveTimer=null}
 if(on)liveTimer=setInterval(async()=>{ov.src='/live.jpg?'+Date.now();const s=await (await fetch('/api/live_state')).json();
  lit.textContent='LIT: '+((s.lit||[]).join(', ')||'none')+'   '+JSON.stringify((s.summary||{}).by_class||{})+'   '+s.fps+' fps'},1000)}
</script></body></html>"""

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), threaded=True)
