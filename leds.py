"""Category bulbs: one LED per category on the Pi's GPIO (gpiozero)."""
import threading
import time

# category -> BCM GPIO pin, confirmed with led_identify.py: GPIO10 red, GPIO22 yellow, GPIO27 green,
# GPIO17 blue (active-high). Smallest -> largest = red, yellow, green, blue. Override: "bulb_pins" in config.json.
DEFAULT_PINS = {"seed": 10, "couscous": 22, "rice": 27, "noodle": 17}


class Bulbs:
    def __init__(self, pins=None, active_high=True):
        self.pins = {k: int(v) for k, v in (pins or DEFAULT_PINS).items()}
        self.active_high = active_high
        self.leds, self.state, self.error = {}, {}, None
        self.lock = threading.Lock()
        self._open()

    def _open(self):
        try:
            from gpiozero import LED
            self.leds = {c: LED(p, active_high=self.active_high) for c, p in self.pins.items()}
            self.state, self.error = {}, None
        except Exception as e:                      # pins busy (another script) or not a Pi
            self.leds, self.error = {}, repr(e)

    def show(self, cats):
        """Light exactly the bulbs of the categories in `cats`."""
        with self.lock:
            if not self.leds:
                self._open()
            for c, led in self.leds.items():
                on = c in cats
                if self.state.get(c) != on:
                    led.on() if on else led.off()
                    self.state[c] = on
        return self.status()

    def off(self):
        return self.show(set())

    def status(self):
        return {"pins": self.pins, "on": sorted(c for c, v in self.state.items() if v), "error": self.error}

    def test(self, seconds=0.7):
        for c in self.pins:
            self.show({c})
            time.sleep(seconds)
        self.show(set(self.pins))
        time.sleep(seconds)
        return self.off()
