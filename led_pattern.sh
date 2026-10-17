#!/bin/bash
# Restart the bulb identification pattern for 7 minutes.
cd "$(dirname "$0")"
pkill -f "[l]ed_identify" ; sleep 1
sed -i "s/time.sleep([0-9]*)/time.sleep(420)/" led_identify.py
nohup timeout 430 python3 led_identify.py > /tmp/led_id.log 2>&1 &
sleep 2
pinctrl get 10,17,22,27
