#!/bin/bash
# Which sensor modes stream reliably? Each trial: 3 s of MJPEG, count frames written.
for cfg in "2592 1944 15" "2592 1944 3" "1296 972 15" "1920 1080 15"; do
  set -- $cfg
  res=""
  for t in 1 2 3; do
    rm -f /tmp/m.mjpeg
    timeout 9 rpicam-vid -n -t 3000 --codec mjpeg --width $1 --height $2 --framerate $3 -q 85 -o /tmp/m.mjpeg >/dev/null 2>&1
    n=$(python3 -c "import os; print(open('/tmp/m.mjpeg','rb').read().count(b'\xff\xd8\xff') if os.path.exists('/tmp/m.mjpeg') else 0)")
    res="$res $n"
  done
  echo "${1}x${2} @${3}fps: frames per 3 s trial =$res"
done
