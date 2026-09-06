from __future__ import annotations

import sys


class WindowCapture:
    """eFootball client-area capture. No fullscreen fallback."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.cam = None
        self.region = None
        self.title = None
        self._grab_region = False

    @staticmethod
    def find_window(title_query: str):
        if sys.platform != "win32":
            return None, None
        import win32gui
        query = (title_query or "eFootball").casefold().strip()
        matches = []
        def visit(hwnd, _):
            if not win32gui.IsWindowVisible(hwnd):
                return
            title = win32gui.GetWindowText(hwnd).strip()
            if not title:
                return
            low = title.casefold()
            if query in low or "efootball" in low:
                matches.append((hwnd, title))
        win32gui.EnumWindows(visit, None)
        if not matches:
            return None, None
        matches.sort(key=lambda x: (0 if query in x[1].casefold() else 1, len(x[1])))
        return matches[0]

    @classmethod
    def client_region(cls, title_query: str):
        hwnd, title = cls.find_window(title_query)
        if hwnd is None:
            return None, None
        import win32gui
        l, t, r, b = win32gui.GetClientRect(hwnd)
        sx, sy = win32gui.ClientToScreen(hwnd, (l, t))
        ex, ey = win32gui.ClientToScreen(hwnd, (r, b))
        if ex <= sx or ey <= sy:
            return None, None
        return (int(sx), int(sy), int(ex), int(ey)), title

    def start(self):
        if not self.cfg.window_only:
            raise RuntimeError("V24 requires window-only capture; fullscreen fallback is disabled.")
        if sys.platform != "win32":
            raise RuntimeError("Window capture requires Windows.")
        self.region, self.title = self.client_region(self.cfg.window_title)
        if self.region is None:
            raise RuntimeError(f"eFootball window not found: {self.cfg.window_title!r}")
        print(f"[CAPTURE] window={self.title!r} client_region={self.region} window_only=True")

        import dxcam
        self.cam = dxcam.create(output_color="BGR")
        # No screen-wide fallback: a DXCam build without region support is a hard failure.
        self.cam.start(
            target_fps=self.cfg.fps,
            video_mode=True,
            region=self.region,
        )

        first = self.read()
        if first is not None:
            print(f"[CAPTURE] frame={first.shape[1]}x{first.shape[0]} window_only=True")

    def read(self):
        if self.cam is None:
            return None
        if self._grab_region:
            frame = self.cam.grab(region=self.region)
        else:
            frame = self.cam.get_latest_frame()
        if frame is None:
            return None
        import cv2
        target = (int(self.cfg.width), int(self.cfg.height))
        if (frame.shape[1], frame.shape[0]) != target:
            frame = cv2.resize(frame, target, interpolation=cv2.INTER_AREA)
        return frame

    def stop(self):
        if self.cam is None:
            return
        try:
            self.cam.stop()
        except Exception:
            pass
        self.cam = None
