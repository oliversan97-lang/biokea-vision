#!/bin/bash
# Keeps server.py running: if the camera watchdog exits the process, restart it.
cd "$(dirname "$0")"
while true; do
  echo "=== server start $(date +%T)" >> server.log
  python3 server.py >> server.log 2>&1
  echo "=== server exited (code $?) at $(date +%T); restarting" >> server.log
  sleep 2
done
