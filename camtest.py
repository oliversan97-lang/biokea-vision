import sys, threading, time
from picamera2 import Picamera2

mode = sys.argv[1] if len(sys.argv) > 1 else "1080"
p = Picamera2()
if mode == "1080":   # cropped 1920x1080 sensor mode: native pixel pitch, centre of the sensor
    cfg = p.create_video_configuration(main={"size": (1920, 1080), "format": "RGB888"},
                                       lores={"size": (640, 360), "format": "YUV420"},
                                       sensor={"output_size": (1920, 1080), "bit_depth": 10},
                                       controls={"FrameDurationLimits": (66666, 66666)}, buffer_count=4)
else:                # full-res mode (flaky on this unit)
    cfg = p.create_still_configuration(main={"size": (2592, 1944), "format": "RGB888"},
                                       lores={"size": (640, 480), "format": "YUV420"}, buffer_count=3)
p.configure(cfg)
p.start()
t0 = time.time()
got = []


def grab():
    for i in range(30):
        r = p.capture_request()
        if i == 29:
            got.append(r.make_array("main").shape)
        r.release()
        got.append(i)


th = threading.Thread(target=grab, daemon=True)
th.start()
th.join(8)
print(f"  {mode}: {'OK' if not th.is_alive() else 'HUNG'} frames={len([g for g in got if isinstance(g, int)])} "
      f"in {time.time() - t0:.1f}s shape={[g for g in got if not isinstance(g, int)]} "
      f"sensor={p.camera_config['sensor']}", flush=True)
import os
os._exit(0)
