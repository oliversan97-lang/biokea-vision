# BioKEA Vision

Tray vision for sorting small dry food items on a Raspberry Pi. A camera looks down on a tray, each
item is measured in millimetres and classified into one of four size categories, and a coloured
bulb per category lights up on the Pi's GPIO.

| Category | Examples in the training set | Bulb | GPIO (BCM) |
|---|---|---|---|
| `seed` | chia | red | 10 |
| `couscous` | couscous | yellow | 22 |
| `rice` | white rice, wild rice | green | 27 |
| `noodle` | broken / whole noodle | blue | 17 |

Pipeline: **calibrate → segment → measure → classify → pick list**. All measurements are taken in a
rectified top-down image (ArUco markers give the mm scale), so lengths and areas come out directly in mm.

## Hardware

- Raspberry Pi 4 Model B, Raspberry Pi OS / Debian 13 (trixie)
- Camera Module v1 (OV5647), used in the 1920×1080 cropped sensor mode
- 4 LEDs/bulbs on GPIO 10, 22, 27, 17 (active-high); override in `config.json` → `bulb_pins`
- Printed ArUco marker sheet (`markers_letter.pdf`, US Letter, print at 100 %) placed around the tray

## Software

Python 3 with the Pi's system packages:

```bash
sudo apt install python3-picamera2 python3-opencv python3-flask python3-sklearn python3-gpiozero python3-joblib
```

Tested with Flask 3.1, OpenCV 4.10, scikit-learn 1.4, NumPy 2.2. `server.py` and `cli.py` also run
on a laptop without the camera stack (no live preview there).

## Web UI (on the Pi)

```bash
./start.sh            # runs server.py under supervise.sh (auto-restart if the camera stalls)
# open http://<pi-ip>:8000
```

Or run it directly in the foreground: `python3 server.py`. Logs go to `server.log`.

The UI provides live preview with a detection overlay, ArUco calibration, empty-tray background
capture, labelled snapshots for training, and a before/after pick-verify loop that writes provenance
JSONL (one record per well). Useful endpoints:

| Endpoint | What it does |
|---|---|
| `GET /stream.mjpg` | Live MJPEG stream |
| `GET /snapshot.jpg` | Single frame |
| `POST /api/detect` | Detect + classify the current frame |
| `POST /api/calibrate` | Calibrate scale from the ArUco markers |
| `POST /api/background` | Store the empty-tray reference |
| `POST /api/snap?label=<class>` | Save a labelled training photo |
| `POST /api/before`, `POST /api/after` | Pick-verify loop |
| `POST /api/bulbs` | Drive the category bulbs |
| `GET /api/state` | Current run state |

`./snap.sh <label>` takes a labelled photo through the API and prints quick quality stats.

## Offline CLI (Pi or laptop)

```bash
python3 cli.py markers                               # printable marker sheet -> markers_letter.png/.pdf
python3 cli.py calibrate img.jpg [--marker-mm 20]    # scale from printed marker size
python3 cli.py calibrate img.jpg --w 150 --h 100     # or 4 corner markers, inner-corner distances in mm
python3 cli.py background empty_tray.jpg             # store empty-tray reference
python3 cli.py detect img.jpg [--crops]              # -> results/<name>.json + _overlay.jpg
python3 cli.py train data/singles                    # folders named by class -> model.joblib
python3 cli.py verify before.jpg after.jpg --target obj_0003
python3 cli.py purity img.jpg --fraction rice --screen-mm 2.0
```

## Training data and model

`data/singles/<label>/` holds labelled tray photos (one product per photo) with cached per-object
features (`<stem>.json`, written by `cache_feats.py`). Folder names are merged into the four
categories (`chia → seed`, `white_rice`/`wild_rice → rice`, `broken_noodle → noodle`).

`model.joblib` is the trained classifier; `train_report.txt` has the cross-validation results
(5-fold, 267 samples: size-only random forest 88.8 %, size + colour random forest 95.9 %) and the
depth-3 "virtual sieve" size rules.

## Files

| File | Purpose |
|---|---|
| `vision.py` | Core library: calibration, segmentation, measurement, classification, overlays, verify |
| `server.py` | Flask web UI + camera loop |
| `cli.py` | Offline command-line tools |
| `leds.py` | Category bulbs on GPIO (gpiozero) |
| `config.json` | ROI and bulb pin mapping |
| `calib.json` | Current camera calibration |
| `start.sh` / `supervise.sh` | Start the UI with auto-restart |
| `snap.sh` | Labelled snapshot + stats |
| `led_identify.py` / `led_pattern.sh` | Find which GPIO drives which bulb |
| `camtest.py` / `modetest.sh` | Camera mode / stream reliability tests |
| `cache_feats.py`, `reref.py`, `fix_calib.py` | Maintenance: freeze features, re-measure against an older reference, re-anchor calibration |
| `sweep.py` | Detection-threshold sensitivity sweep |
| `synth.py` | Synthetic tray image for smoke-testing the pipeline |

Not in the repo (see `.gitignore`): raw `captures/`, `results/`, background references (`bg*.png`,
`*.npy`, `ref_*/`), runtime `state.json` and logs. Recapture the background and recalibrate on a new setup.
