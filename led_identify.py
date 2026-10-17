"""Identify which GPIO drives which bulb: each pin shows a different pattern at the same time."""
import time

from gpiozero import LED

pins = {10: "SOLID ON", 17: "SLOW BLINK", 22: "FAST BLINK", 27: "OFF"}
leds = {p: LED(p) for p in pins}
leds[10].on()
leds[17].blink(on_time=1.0, off_time=1.0)
leds[22].blink(on_time=0.15, off_time=0.15)
leds[27].off()
print({p: s for p, s in pins.items()}, flush=True)
time.sleep(420)
