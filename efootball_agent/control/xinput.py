from __future__ import annotations

import threading
import time

from ..core import Action, FootballAction
from .actions import to_xinput, StickyControlState


class XInputController:
    """120 Hz executor. Action buttons are pulses; Y is never emitted."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.dry_run = cfg.dry_run
        self.lock = threading.Lock()
        self.action = Action()
        self.sticky = StickyControlState()
        self.running = False
        self.thread = None
        self.pad = None
        self.vg = None
        self._pulse = {"pass": 0.0, "cross": 0.0, "shoot": 0.0, "switch": 0.0}
        self._next = {"pass":0.0, "cross":0.0, "shoot":0.0, "switch":0.0}
        if not self.dry_run:
            import vgamepad as vg
            self.vg = vg
            self.pad = vg.VX360Gamepad()

    def set_action(self, action: Action):
        with self.lock:
            self.action = action
            self.sticky.set_desired(action, time.monotonic())

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._loop, name="xinput", daemon=True)
        self.thread.start()

    def _tap(self, name, button, now, duration, cooldown):
        if now >= self._next[name]:
            self.pad.press_button(button=button)
            self._pulse[name] = now + duration
            self._next[name] = now + cooldown

    def _loop(self):
        period = 1.0 / max(1.0, self.cfg.hz)
        print("[XINPUT] worker started; Y/GK-rush disabled")
        try:
            while self.running:
                now = time.monotonic()
                with self.lock:
                    action = self.sticky.snapshot()
                if not self.dry_run:
                    mapped = to_xinput(action)
                    self.pad.left_joystick_float(
                        x_value_float=mapped["lx"],
                        y_value_float=-mapped["ly"],
                    )
                    # Release expired pulses first.
                    for name, button in (
                        ("pass", self.vg.XUSB_BUTTON.XUSB_GAMEPAD_A),
                        ("cross", self.vg.XUSB_BUTTON.XUSB_GAMEPAD_B),
                        ("shoot", self.vg.XUSB_BUTTON.XUSB_GAMEPAD_X),
                        ("switch", self.vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_SHOULDER),
                    ):
                        if self._pulse[name] and now >= self._pulse[name]:
                            self.pad.release_button(button=button)
                            self._pulse[name] = 0.0

                    if mapped["pass"]:
                        self._tap("pass", self.vg.XUSB_BUTTON.XUSB_GAMEPAD_A, now, self.cfg.pass_tap_s, self.cfg.pass_cooldown_s)
                    if mapped["cross"]:
                        self._tap("cross", self.vg.XUSB_BUTTON.XUSB_GAMEPAD_B, now, self.cfg.cross_tap_s, self.cfg.cross_cooldown_s)
                    if mapped["shoot"]:
                        self._tap("shoot", self.vg.XUSB_BUTTON.XUSB_GAMEPAD_X, now, self.cfg.shoot_tap_s, self.cfg.shoot_cooldown_s)
                    if mapped["switch"]:
                        self._tap("switch", self.vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_SHOULDER, now, self.cfg.switch_tap_s, self.cfg.switch_cooldown_s)

                    rb = self.vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_SHOULDER
                    if mapped["press"]:
                        self.pad.press_button(button=rb)
                    else:
                        self.pad.release_button(button=rb)

                    self.pad.right_trigger(255 if mapped["sprint"] else 0)
                    self.pad.update()
                time.sleep(period)
        finally:
            self.stop()

    def stop(self):
        self.running = False
        if self.thread is not None and threading.current_thread() is not self.thread:
            self.thread.join(timeout=1.0)
            self.thread = None
        if self.dry_run or self.pad is None:
            return
        try:
            for button in (
                self.vg.XUSB_BUTTON.XUSB_GAMEPAD_A,
                self.vg.XUSB_BUTTON.XUSB_GAMEPAD_B,
                self.vg.XUSB_BUTTON.XUSB_GAMEPAD_X,
                self.vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_SHOULDER,
                self.vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_SHOULDER,
            ):
                self.pad.release_button(button=button)
            self.pad.right_trigger(0)
            self.pad.left_joystick_float(x_value_float=0.0, y_value_float=0.0)
            self.pad.update()
        except Exception:
            pass
