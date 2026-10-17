#!/bin/bash
# (Re)start the web UI under supervise.sh (auto-restart if the camera stalls). Logs -> server.log
cd "$(dirname "$0")"
pkill -f "[s]upervise.sh"
pkill -f "[p]ython3 server.py"
sleep 2
: > server.log
nohup ./supervise.sh > /dev/null 2>&1 &
sleep 5
grep -E "WATCHDOG|camera unavailable|Traceback" server.log | tail -3
echo "open http://$(hostname -I | awk '{print $1}'):8000"
