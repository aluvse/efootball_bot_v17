from __future__ import annotations

import cv2
import numpy as np


class ActiveMarkerDetector:
    """Detect cyan cursor and associate it with a player track with hysteresis."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.last_marker = None
        self.last_track_id = -1
        self.last_confidence = 0.0
        self.missed = 0
        self.pending_switch_id = -1
        self.pending_switch_count = 0

    def reset(self):
        self.last_marker = None
        self.last_track_id = -1
        self.last_confidence = 0.0
        self.missed = 0
        self.pending_switch_id = -1
        self.pending_switch_count = 0

    @staticmethod
    def _cyan_mask(frame):
        b, g, r = cv2.split(frame)
        blue_dominant = (
            (b.astype(np.int16) - r.astype(np.int16)) > 70
        ) & (
            (g.astype(np.int16) - r.astype(np.int16)) > 70
        )
        mask = (
            (b >= 140)
            & (g >= 130)
            & (r <= 110)
            & blue_dominant
        ).astype(np.uint8) * 255
        return mask

    def detect(self, frame):
        mask = self._cyan_mask(frame)
        kernel = np.ones((2, 2), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
        h, w = frame.shape[:2]
        candidates = []
        for i in range(1, n):
            area = int(stats[i, cv2.CC_STAT_AREA])
            x = float(cents[i][0] / w)
            y = float(cents[i][1] / h)
            bw = int(stats[i, cv2.CC_STAT_WIDTH])
            bh = int(stats[i, cv2.CC_STAT_HEIGHT])
            if not self.cfg.min_pixels <= area <= self.cfg.max_pixels:
                continue
            if y > 0.82:
                continue
            # eFootball may render the active selection as a thin horizontal
            # cyan bar above the player rather than a compact symbol.
            if bw > 42 or bh > 18:
                continue
            aspect = bw / max(1, bh)
            if not 0.35 <= aspect <= 12.0:
                continue
            # Long thin bars are valid only when they really are cyan; the
            # color segmentation above is the primary proof of that.
            candidates.append((x, y, area, bw, bh))

        if self.last_marker is not None and candidates:
            px, py = self.last_marker
            candidates.sort(
                key=lambda c: (
                    (c[0] - px) ** 2 + (c[1] - py) ** 2,
                    -c[2],
                )
            )
        else:
            candidates.sort(key=lambda c: (-c[2], c[1], c[0]))

        clean = [(c[0], c[1], c[2]) for c in candidates]
        if clean:
            self.last_marker = clean[0][:2]
            self.missed = 0
            return self.last_marker, clean

        self.missed += 1
        if self.last_marker is not None and self.missed <= self.cfg.max_missing:
            return self.last_marker, clean
        return None, clean

    @staticmethod
    def _player_points(track):
        x1, y1, x2, y2 = track.bbox
        top_center = ((x1 + x2) * 0.5, y1)
        upper_center = ((x1 + x2) * 0.5, y1 + 0.22 * (y2 - y1))
        center = ((x1 + x2) * 0.5, (y1 + y2) * 0.5)
        return top_center, upper_center, center

    def recover_from_tracks(self, frame, tracks):
        """Recover a weak/fragmented cyan marker locally around track heads.

        The normal detector relies on a connected component large enough to
        survive morphology. On live frames the selector can fragment or become
        thin; this fallback uses cyan pixel density in bounded head/marker
        windows and never considers the whole frame as a substitute.
        """
        if frame is None or not tracks:
            return None, []
        h, w = frame.shape[:2]
        mask = self._cyan_mask(frame)
        candidates = []
        for track in tracks:
            if track.bbox is None:
                continue
            missed_limit = getattr(self.cfg, "active_track_max_missed", self.cfg.max_missing)
            if track.missed > missed_limit:
                continue
            x1, y1, x2, y2 = track.bbox
            cx = 0.5 * (x1 + x2)
            top = y1
            xa = max(0, int((cx - self.cfg.recovery_x_radius) * w))
            xb = min(w, int((cx + self.cfg.recovery_x_radius) * w))
            ya = max(0, int((top + self.cfg.recovery_y_min) * h))
            yb = min(h, int((top + self.cfg.recovery_y_max) * h))
            if xb <= xa or yb <= ya:
                continue
            roi = mask[ya:yb, xa:xb]
            if roi.size == 0:
                continue
            count = int(cv2.countNonZero(roi))
            ratio = float(count / roi.size)
            if count < self.cfg.recovery_min_pixels or ratio < self.cfg.recovery_min_ratio:
                continue
            ys, xs = np.where(roi > 0)
            mx = float((xa + float(xs.mean())) / w)
            my = float((ya + float(ys.mean())) / h)
            # Prefer dense cyan close to the top-center of the track.
            dist = float(np.hypot(mx - cx, my - top))
            score = (0.65 * ratio / max(self.cfg.recovery_target_ratio, 1e-6)) + (0.35 * (1.0 - min(dist / 0.12, 1.0)))
            candidates.append((score, mx, my, track.track_id, ratio))
        if not candidates:
            return None, []
        candidates.sort(key=lambda c: (-c[0], c[3]))
        best = candidates[0]
        conf = float(np.clip(0.45 + 0.30 * min(best[0], 1.0), 0.0, 0.90))
        return (best[1], best[2]), [(best[1], best[2], best[4], best[3], conf)]

    def associate(self, marker, tracks):
        if marker is None:
            if self.last_track_id >= 0:
                # detect() already increments marker-missed frames. Do not increment again here.
                if self.missed <= self.cfg.max_missing:
                    return self.last_track_id, max(0.0, self.last_confidence * 0.97)
                self.last_track_id = -1
                self.last_confidence = 0.0
            return -1, 0.0

        mx, my = marker
        ranked = []
        for track in tracks:
            if track.bbox is None or track.missed > getattr(self.cfg, "active_track_max_missed", self.cfg.max_missing):
                continue
            top_center, upper_center, center = self._player_points(track)
            dx = abs(mx - top_center[0])
            dy = my - top_center[1]
            if dx > self.cfg.x_tol:
                continue
            if not self.cfg.y_above_min <= dy <= self.cfg.y_above_max:
                continue

            top_dist = float(np.hypot(mx - top_center[0], my - top_center[1]))
            upper_dist = float(np.hypot(mx - upper_center[0], my - upper_center[1]))
            center_dist = float(np.hypot(mx - center[0], my - center[1]))
            # Marker is expected near the top-center/head region. Use both top-center
            # and upper-bbox geometry instead of a brittle single y offset.
            geometry = (
                0.46 * min(top_dist / max(self.cfg.x_tol, 1e-6), 2.0)
                + 0.34 * min(upper_dist / max(self.cfg.y_above_max + 0.10, 1e-6), 2.0)
                + 0.20 * min(center_dist / 0.20, 2.0)
            )
            score = float(geometry)
            if track.track_id == self.last_track_id:
                score = max(0.0, score - self.cfg.previous_bonus)
            ranked.append((score, track.track_id))

        if not ranked:
            if self.last_track_id >= 0 and self.missed <= self.cfg.max_missing:
                self.last_confidence *= 0.97
                return self.last_track_id, self.last_confidence
            self.last_track_id = -1
            self.last_confidence = 0.0
            self.pending_switch_id = -1
            self.pending_switch_count = 0
            return -1, 0.0

        ranked.sort(key=lambda v: (v[0], v[1]))
        best_score, best_id = ranked[0]
        candidate_conf = float(np.clip(1.0 - best_score, 0.0, 1.0))

        if self.last_track_id >= 0 and best_id != self.last_track_id:
            current = next((x for x in ranked if x[1] == self.last_track_id), None)
            if current is not None:
                # Switching active player is stateful: require the new winner to
                # remain clearly better for two consecutive marker observations.
                clear_win = best_score + self.cfg.switch_margin < current[0]
                if not clear_win:
                    self.pending_switch_id = -1
                    self.pending_switch_count = 0
                    kept_conf = float(np.clip(1.0 - current[0], 0.0, 1.0))
                    self.last_confidence = max(kept_conf, self.last_confidence * 0.92)
                    self.missed = 0
                    return self.last_track_id, self.last_confidence
                if self.pending_switch_id == int(best_id):
                    self.pending_switch_count += 1
                else:
                    self.pending_switch_id = int(best_id)
                    self.pending_switch_count = 1
                if self.pending_switch_count < 2:
                    kept_conf = float(np.clip(1.0 - current[0], 0.0, 1.0))
                    self.last_confidence = max(kept_conf * 0.98, self.last_confidence * 0.94)
                    self.missed = 0
                    return self.last_track_id, self.last_confidence
            self.pending_switch_id = -1
            self.pending_switch_count = 0

        self.last_track_id = int(best_id)
        self.last_confidence = candidate_conf
        self.pending_switch_id = -1
        self.pending_switch_count = 0
        self.missed = 0
        return self.last_track_id, candidate_conf
