#!/bin/bash
# Usage: ./snap.sh <label>   -> labelled photo + quick quality stats (+ copies overlay to /tmp/latest_overlay.jpg)
cd "$(dirname "$0")"
curl -s -X POST "localhost:8000/api/snap?label=$1" > /tmp/snap.json
python3 - "$1" <<'EOF'
import json, statistics as st, sys
r = json.load(open("/tmp/snap.json"))
if "error" in r:
    sys.exit("error: " + r["error"])
objs = [o for o in r["objects"] if not o["edge"]]
a = r.get("align") or {}
print(f"saved {r['saved']} | photos in folder: {r['n_in_folder']} | tray shift {a.get('shift_mm', 0)} mm, markers {a.get('markers')}")
if not objs:
    sys.exit("no objects found")
med = st.median([o["area_mm2"] for o in objs])
sing = [o for o in objs if med / 3 <= o["area_mm2"] <= med * 2.5]
print(f"blobs: {len(objs)} | item-sized: {len(sing)} | clumps: {sum(o['area_mm2'] > med * 2.5 for o in objs)} | "
      f"debris: {sum(o['area_mm2'] < med / 3 for o in objs)} | median L x W: "
      f"{st.median([o['length_mm'] for o in sing]):.2f} x {st.median([o['width_mm'] for o in sing]):.2f} mm")
EOF
cp "results/$(python3 -c 'import json; print(json.load(open("/tmp/snap.json")).get("overlay", ""))')" /tmp/latest_overlay.jpg 2>/dev/null
