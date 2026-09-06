from __future__ import annotations

import time

import cv2
import numpy as np

from ..core import Ball
from .ui_mask import UIMask


class BallManager:
    """Confidence-aware ball fusion with strict detector priority.

    Sources are intentionally explicit:
      DETECTOR   = validated RT-DETR proposal
      CV         = compact field-grounded CV candidate
      PREDICTED  = temporal extrapolation after a real detection
      UNKNOWN    = no trustworthy measurement
    """

    def __init__(self, cfg, ui: UIMask):
        self.cfg = cfg
        self.ui = ui
        self.previous: Ball | None = None
        self.previous_time: float | None = None
        self.missed = 0
        self.cv_pending_point: tuple[float, float] | None = None
        self.cv_pending_count = 0
        self.last_debug = self._empty_debug()
        self.frame_counter = 0
        self.confirmed_streak = 0
        self.last_real_measurement_time: float | None = None

    @staticmethod
    def _empty_debug():
        return {
            "rtdetr_confidence": 0.0,
            "cv_confidence": 0.0,
            "selected_source": "UNKNOWN",
            "final_confidence": 0.0,
            "field_valid": False,
            "ui_rejected": False,
            "temporal_consistency": False,
            "cv_candidates": 0,
            "detector_candidates": 0,
            "fusion_corroborated": False,
            "fusion_distance": None,
            "fusion_confidence_bonus": 0.0,
            "footpoint_candidates": 0,
            "near_foot": 0.0,
            "foot_distance": None,
            "prediction_age_s": None,
        }

    def reset(self):
        self.previous = None
        self.previous_time = None
        self.missed = 0
        self.cv_pending_point = None
        self.cv_pending_count = 0
        self.last_debug = self._empty_debug()
        self.frame_counter = 0
        self.confirmed_streak = 0
        self.last_real_measurement_time = None

    @staticmethod
    def _circle_score(contour, area, bw, bh):
        perimeter = float(cv2.arcLength(contour, True))
        circularity = (4.0 * np.pi * area / (perimeter * perimeter)) if perimeter else 0.0
        aspect = min(float(bw / max(bh, 1)), float(bh / max(bw, 1)))
        compactness = float(area / max(1, bw * bh))
        return float(np.clip(
            0.50 * min(circularity, 1.0)
            + 0.30 * aspect
            + 0.20 * compactness,
            0.0,
            1.0,
        ))

    def _local_field_metrics(self, frame, point):
        x, y = float(point[0]), float(point[1])
        h, w = frame.shape[:2]
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            return 0.0, True
        if self.ui.is_ui(point):
            return 0.0, True
        xi = int(np.clip(x * w, 0, w - 1))
        yi = int(np.clip(y * h, 0, h - 1))
        radius = max(7, int(min(h, w) * 0.014))
        xa, xb = max(0, xi - radius), min(w, xi + radius + 1)
        ya, yb = max(0, yi - radius), min(h, yi + radius + 1)
        patch = frame[ya:yb, xa:xb]
        if patch.size == 0:
            return 0.0, False
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        H, S, V = cv2.split(hsv)
        green = (H >= 25) & (H <= 100) & (S >= 30) & (V >= 40)
        return float(green.mean()), False

    def _candidate_confidence(self, *, field, shape, brightness, background_contrast, area, bw, bh, mask):
        # HSV is only one feature. A trustworthy white-ball candidate on the
        # supplied live frame is small, compact/circular, bright against the
        # green pitch, and field-grounded. Orange/yellow candidates remain
        # intentionally capped by their shape penalty.
        size_target = np.exp(-abs(np.log(max(1.0, area) / 34.0)) / 0.95)
        aspect = min(float(bw / max(bh, 1)), float(bh / max(bw, 1)))
        aspect_score = float(np.clip((aspect - 0.55) / 0.45, 0.0, 1.0))
        score = (
            0.12 * field
            + 0.34 * shape
            + 0.18 * brightness
            + 0.20 * np.clip(background_contrast / 80.0, 0.0, 1.0)
            + 0.10 * size_target
            + 0.06 * aspect_score
        )
        if mask == "orange":
            score *= self.cfg.cv_orange_confidence_cap
        return float(np.clip(score, 0.0, 0.96))

    def _cv_candidates(self, frame):
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Real ball on the supplied live frame is achromatic/white rather than orange.
        # Keep the old orange mask only as a secondary channel, never as sole evidence.
        masks = [
            ((hsv[:, :, 2] >= self.cfg.cv_white_v_min) &
             (hsv[:, :, 1] <= self.cfg.cv_white_s_max)).astype(np.uint8) * 255,
            cv2.inRange(
                hsv,
                np.array([self.cfg.cv_h_low, self.cfg.cv_s_min, self.cfg.cv_v_min], np.uint8),
                np.array([self.cfg.cv_h_high, 255, 255], np.uint8),
            ),
        ]

        out = []
        seen = set()
        for mask_index, mask in enumerate(masks):
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = float(cv2.contourArea(contour))
                if not self.cfg.cv_min_area <= area <= self.cfg.cv_max_area:
                    continue
                x, y, bw, bh = cv2.boundingRect(contour)
                if not (self.cfg.cv_min_width <= bw <= self.cfg.cv_max_width and
                        self.cfg.cv_min_height <= bh <= self.cfg.cv_max_height):
                    continue
                ratio = bw / max(1.0, bh)
                if ratio < self.cfg.cv_aspect_min or ratio > self.cfg.cv_aspect_max:
                    continue
                moments = cv2.moments(contour)
                if moments["m00"] == 0:
                    continue
                point = (
                    float((moments["m10"] / moments["m00"]) / w),
                    float((moments["m01"] / moments["m00"]) / h),
                )
                if point[1] >= self.cfg.cv_y_max or point[1] <= self.cfg.cv_y_min:
                    continue
                field, ui_rejected = self._local_field_metrics(frame, point)
                if ui_rejected or field < self.cfg.min_field_green:
                    continue

                key = (round(point[0], 4), round(point[1], 4))
                if key in seen:
                    continue
                seen.add(key)

                patch_radius = max(5, int(min(h, w) * 0.008))
                xi, yi = int(point[0] * w), int(point[1] * h)
                xa, xb = max(0, xi - patch_radius), min(w, xi + patch_radius + 1)
                ya, yb = max(0, yi - patch_radius), min(h, yi + patch_radius + 1)
                patch = frame[ya:yb, xa:xb]
                phsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
                center_v = float(phsv[:, :, 2].max()) if phsv.size else 0.0
                background_v = float(np.median(phsv[:, :, 2])) if phsv.size else 0.0
                brightness = float(np.clip((center_v - 140.0) / 90.0, 0.0, 1.0))
                contrast = max(0.0, center_v - background_v)
                shape = self._circle_score(contour, area, bw, bh)

                if mask_index == 0:
                    # White-ball candidate must really be bright and compact.
                    if brightness < self.cfg.cv_min_brightness:
                        continue
                else:
                    # Orange/yellow candidate has a lower ceiling; this protects against ads.
                    shape *= self.cfg.cv_orange_shape_penalty

                conf = self._candidate_confidence(
                    field=field,
                    shape=shape,
                    brightness=brightness,
                    background_contrast=contrast,
                    area=area,
                    bw=bw,
                    bh=bh,
                    mask="white" if mask_index == 0 else "orange",
                )
                out.append({
                    "point": point,
                    "confidence": conf,
                    "field_valid": True,
                    "ui_rejected": False,
                    "temporal_consistency": False,
                    "mask": "white" if mask_index == 0 else "orange",
                    "area": area,
                    "bbox": (x, y, bw, bh),
                    "circularity_shape": shape,
                    "background_contrast": contrast,
                })
        return out

    @staticmethod
    def _track_footpoints(players):
        points = []
        for p in players or []:
            bbox = getattr(p, "bbox", None)
            if bbox is not None:
                x1, y1, x2, y2 = [float(v) for v in bbox]
                points.append(((x1 + x2) * 0.5, y2))
            else:
                pos = getattr(p, "pos", None)
                if pos is not None:
                    points.append((float(pos[0]), float(pos[1]) + 0.035))
        return points

    def _near_foot_score(self, point, footpoints, max_dist=0.065):
        if not footpoints:
            return 0.0, None
        pt = np.asarray(point, np.float32)
        best = min(
            (float(np.linalg.norm(pt - np.asarray(fp, np.float32))), fp)
            for fp in footpoints
        )
        dist, fp = best
        return float(np.clip(1.0 - dist / max_dist, 0.0, 1.0)), dist

    def _footpoint_candidates(self, frame, players):
        """Find compact white ball-like blobs immediately around player feet."""
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        white = ((hsv[:, :, 2] >= self.cfg.cv_white_v_min) &
                 (hsv[:, :, 1] <= self.cfg.cv_white_s_max)).astype(np.uint8) * 255
        out = []
        for p in players or []:
            bbox = getattr(p, "bbox", None)
            if bbox is None:
                continue
            x1, y1, x2, y2 = [float(v) for v in bbox]
            fx, fy = (x1 + x2) * 0.5, y2
            cx, cy = int(fx * w), int(fy * h)
            rx = max(18, int(0.055 * w))
            ry = max(14, int(0.075 * h))
            xa, xb = max(0, cx - rx), min(w, cx + rx)
            ya, yb = max(0, cy - ry), min(h, cy + ry)
            roi = white[ya:yb, xa:xb]
            if roi.size == 0:
                continue
            # Ignore the body bbox itself; the ball can overlap the lower edge.
            mask = roi.copy()
            body_xa = max(0, int(x1*w) - xa)
            body_xb = min(mask.shape[1], int(x2*w) - xa)
            body_ya = max(0, int(y1*h) - ya)
            body_yb = min(mask.shape[0], int(y2*h) - ya)
            if body_xb > body_xa and body_yb > body_ya:
                # Keep a narrow 8px band around the bottom edge for an overlapping ball.
                keep_ya = max(body_ya, body_yb - 8)
                mask[body_ya:keep_ya, body_xa:body_xb] = 0
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2,2), np.uint8))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                xx, yy, bw, bh = cv2.boundingRect(c)
                area = float(cv2.contourArea(c))
                if not (self.cfg.cv_min_area <= area <= self.cfg.cv_max_area):
                    continue
                if not (self.cfg.cv_min_width <= bw <= self.cfg.cv_max_width and self.cfg.cv_min_height <= bh <= self.cfg.cv_max_height):
                    continue
                per = float(cv2.arcLength(c, True))
                circularity = float(4*np.pi*area/(per*per)) if per > 1e-6 else 0.0
                if circularity < 0.28:
                    continue
                px = (xa + xx + bw*0.5) / w
                py = (ya + yy + bh*0.5) / h
                # A white boot/shin can look more ball-like than the actual ball.
                # Require the candidate center to be outside the player's torso bbox;
                # the real eFootball ball in the supplied frame sits just left/below it.
                inside_body = (x1 <= px <= x2 and y1 <= py <= y2)
                if inside_body:
                    continue
                foot_dist = float(np.linalg.norm(np.asarray([px, py], np.float32) - np.asarray([fx, fy], np.float32)))
                near = float(np.clip(1.0 - foot_dist / 0.065, 0.0, 1.0))
                score = float(np.clip(0.45*near + 0.30*circularity + 0.25*min(1.0, area/8.0), 0, 1))
                out.append({
                    "point": (px, py),
                    "confidence": score,
                    "field_valid": True,
                    "ui_rejected": False,
                    "temporal_consistency": self._temporal_consistent((px, py)),
                    "mask": "foot_white",
                    "area": area,
                    "circularity_shape": circularity,
                    "near_foot": near,
                    "foot_distance": foot_dist,
                })
        return out

    def _local_candidates(self, frame, players=None):
        """Fast local ball reacquisition around the last trusted measurement."""
        if self.previous is None:
            return []
        h, w = frame.shape[:2]
        cx = int(np.clip(self.previous.x * w, 0, w - 1))
        cy = int(np.clip(self.previous.y * h, 0, h - 1))
        rx = max(24, int(w * 0.055))
        ry = max(20, int(h * 0.065))
        xa, xb = max(0, cx-rx), min(w, cx+rx+1)
        ya, yb = max(0, cy-ry), min(h, cy+ry+1)
        roi = frame[ya:yb, xa:xb]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = ((hsv[:, :, 2] >= self.cfg.cv_white_v_min) &
                (hsv[:, :, 1] <= self.cfg.cv_white_s_max)).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2,2), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        feet = self._track_footpoints(players)
        out = []
        for c in contours:
            area = float(cv2.contourArea(c))
            x, y, bw, bh = cv2.boundingRect(c)
            if not (self.cfg.cv_min_area <= area <= self.cfg.cv_max_area):
                continue
            if not (self.cfg.cv_min_width <= bw <= self.cfg.cv_max_width and
                    self.cfg.cv_min_height <= bh <= self.cfg.cv_max_height):
                continue
            per = float(cv2.arcLength(c, True))
            circularity = float(4*np.pi*area/(per*per)) if per > 1e-6 else 0.0
            if circularity < 0.25:
                continue
            px, py = (xa+x+bw*0.5)/w, (ya+y+bh*0.5)/h
            dist_prev = float(np.linalg.norm(np.asarray([px,py],np.float32) - np.asarray([self.previous.x,self.previous.y],np.float32)))
            if dist_prev > self.cfg.cv_local_max_dist:
                continue
            near, foot_dist = self._near_foot_score((px,py), feet, max_dist=0.065)
            field, ui_rejected = self._local_field_metrics(frame, (px,py))
            if ui_rejected or field < self.cfg.min_field_green:
                continue
            conf = float(np.clip(0.42*field + 0.33*circularity + 0.15*near + 0.10*np.exp(-dist_prev/0.025), 0, 0.95))
            out.append({
                "point": (px,py), "confidence": conf, "field_valid": True,
                "ui_rejected": False, "temporal_consistency": True,
                "mask": "local_white", "area": area,
                "circularity_shape": circularity, "near_foot": near,
                "foot_distance": foot_dist, "dist_prev": dist_prev,
            })
        return out

    def _model_candidates(self, frame, model_balls, players=None, active_track_id=-1):
        out = []
        for mb in model_balls:
            point = (float(mb.pos[0]), float(mb.pos[1]))
            field, ui_rejected = self._local_field_metrics(frame, point)
            if ui_rejected or field < self.cfg.model_min_field_green:
                continue

            # RT-DETR class-32 proposals are still proposals: reject implausible
            # ball geometry before allowing detector evidence to dominate CV.
            # This is especially important when the model confuses a player's
            # torso with the ball class.
            if mb.bbox is not None:
                x1, y1, x2, y2 = map(float, mb.bbox)
                bw = max(0.0, x2 - x1)
                bh = max(0.0, y2 - y1)
                aspect = bw / max(bh, 1e-6)
                if not (
                    self.cfg.detector_min_width <= bw <= self.cfg.detector_max_width
                    and self.cfg.detector_min_height <= bh <= self.cfg.detector_max_height
                    and self.cfg.detector_aspect_min <= aspect <= self.cfg.detector_aspect_max
                ):
                    continue

            score = float(mb.confidence * (0.72 + 0.28 * field))
            footpoints = self._track_footpoints(players)
            near_foot, foot_distance = self._near_foot_score(point, footpoints)
            # A detector ball proposal is much more plausible when it is at a player's feet.
            # This is a ranking/validation feature, never a replacement for detector evidence.
            if foot_distance is not None and foot_distance <= 0.075:
                score = float(min(0.95, score + 0.22 * near_foot))
            elif foot_distance is not None and foot_distance > 0.090:
                score *= 0.52
            temporal = False
            if self.previous is not None:
                prev = np.array([self.previous.x, self.previous.y], np.float32)
                jump = float(np.linalg.norm(np.asarray(point, np.float32) - prev))
                temporal = jump <= self.cfg.max_jump
                if not temporal:
                    score *= self.cfg.detector_jump_penalty
            out.append({
                "point": point,
                "confidence": score,
                "field_valid": True,
                "ui_rejected": False,
                "temporal_consistency": temporal,
                "near_foot": near_foot,
                "foot_distance": foot_distance,
            })
        return out


    def _select_local_cv(self, cvs):
        if self.previous is None or not cvs:
            return None
        prev = np.array([self.previous.x, self.previous.y], np.float32)
        ranked = []
        for c in cvs:
            pt = np.asarray(c["point"], np.float32)
            dist = float(np.linalg.norm(pt - prev))
            near_foot = float(c.get("near_foot", 0.0))
            temporal = dist <= self.cfg.max_jump
            if dist > min(0.10, self.cfg.max_jump * 0.70) and near_foot < 0.45:
                continue
            score = float(c.get("confidence", 0.0))
            score += 0.18 * near_foot
            score += 0.16 if temporal else 0.0
            score -= min(dist / 0.12, 1.0) * 0.10
            ranked.append((score, c))
        return max(ranked, key=lambda x: (x[0], x[1].get("confidence", 0.0)), default=(0.0, None))[1]
    def _temporal_consistent(self, point):
        if self.previous is None:
            return False
        prev = np.array([self.previous.x, self.previous.y], np.float32)
        return float(np.linalg.norm(np.asarray(point, np.float32) - prev)) <= self.cfg.max_jump

    def _commit(self, chosen, source, confidence, now, *, field_valid=True, temporal_consistency=False):
        point = (float(chosen[0]), float(chosen[1]))
        raw_x, raw_y = point
        vx = vy = 0.0
        filtered_x, filtered_y = raw_x, raw_y
        if self.previous is not None and self.previous_time is not None:
            dt = max(1e-3, now - self.previous_time)
            # Lightweight alpha-beta filter: constant-velocity prediction plus
            # bounded residual correction. This is cheaper than a full Kalman
            # filter and well suited to 30 FPS visual tracking.
            pred = np.array([
                self.previous.x + self.previous.vx * dt,
                self.previous.y + self.previous.vy * dt,
            ], np.float32)
            meas = np.array([raw_x, raw_y], np.float32)
            residual = meas - pred
            alpha = float(np.clip(getattr(self.cfg, "alpha_beta_alpha", 0.72), 0.1, 1.0))
            beta = float(np.clip(getattr(self.cfg, "alpha_beta_beta", 0.18), 0.01, 1.0))
            filtered = pred + alpha * residual
            filtered_x, filtered_y = [float(np.clip(v, 0.0, 1.0)) for v in filtered]
            vx = float(np.clip(self.previous.vx + (beta / dt) * float(residual[0]), -4, 4))
            vy = float(np.clip(self.previous.vy + (beta / dt) * float(residual[1]), -4, 4))
        ball = Ball(
            x=filtered_x,
            y=filtered_y,
            vx=vx,
            vy=vy,
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            source=source,
            predicted=False,
            valid=confidence >= self.cfg.accept_conf,
        )
        self.previous = ball
        self.previous_time = now
        self.last_real_measurement_time = now
        self.missed = 0
        self.confirmed_streak += 1
        self.last_debug.update({
            "selected_source": source,
            "final_confidence": float(ball.confidence),
            "field_valid": bool(field_valid),
            "temporal_consistency": bool(temporal_consistency),
        })
        return ball

    def update(self, frame, model_balls, players=None, active_track_id=-1):
        now = time.monotonic()
        self.frame_counter += 1
        self.last_debug = self._empty_debug()

        detector = self._model_candidates(frame, model_balls, players, active_track_id)
        local = self._local_candidates(frame, players)
        # Global CV search is intentionally throttled; local tracking remains every frame.
        global_cvs = self._cv_candidates(frame) if (self.previous is None or self.frame_counter % max(1, int(getattr(self.cfg, "global_cv_every_n_frames", 3))) == 0) else []
        cvs = global_cvs + self._footpoint_candidates(frame, players) + local
        local_cv = None
        if local:
            local_cv = max(local, key=lambda c: (c.get("confidence",0.0), -c.get("dist_prev",1.0)))
        if local_cv is None:
            local_cv = self._select_local_cv(cvs)
        if local_cv is not None:
            local_cv["temporal_consistency"] = True
            local_cv["confidence"] = float(min(0.95, local_cv.get("confidence", 0.0) + 0.05))
        self.last_debug["detector_candidates"] = len(detector)
        self.last_debug["cv_candidates"] = len(cvs)
        self.last_debug["rtdetr_confidence"] = max((c["confidence"] for c in detector), default=0.0)
        self.last_debug["cv_confidence"] = max((c["confidence"] for c in cvs), default=0.0)
        self.last_debug["footpoint_candidates"] = sum(1 for c in cvs if c.get("mask") == "foot_white")

        # Fast local re-acquisition: once a ball track exists, a close, compact
        # candidate around its last measurement can be accepted on the same frame.
        if (
            not detector
            and local_cv is not None
            and local_cv.get("mask") == "local_white"
            and float(local_cv.get("confidence",0.0)) >= self.cfg.cv_local_accept_conf
            and float(local_cv.get("dist_prev",1.0)) <= self.cfg.cv_local_max_dist
        ):
            self.cv_pending_point = None
            self.cv_pending_count = 0
            self.last_debug.update({
                "selected_source": "CV",
                "final_confidence": float(local_cv["confidence"]),
                "field_valid": True,
                "temporal_consistency": True,
                "near_foot": float(local_cv.get("near_foot",0.0)),
                "foot_distance": local_cv.get("foot_distance"),
            })
            return self._commit(local_cv["point"], "CV", local_cv["confidence"], now, field_valid=True, temporal_consistency=True)

        # P0 rule: detector evidence wins when it is itself plausible. A weak
        # detector proposal that lands on an isolated field marking must not beat
        # a strong player-foot corroborated CV candidate.
        if detector:
            best = max(detector, key=lambda c: (c.get("near_foot", 0.0), c["confidence"], c["temporal_consistency"]))
            foot_cv = [c for c in cvs if c.get("mask") == "foot_white" and c.get("confidence", 0.0) >= 0.58]
            strong_foot_cv = max(foot_cv, key=lambda c: c.get("confidence", 0.0), default=None)

            # When the only detector proposal is far from every player's feet
            # but a compact white candidate is tightly coupled to a player's foot,
            # prefer the foot candidate. This specifically rejects the white
            # field/penalty spot misread visible in the supplied frame.
            if (
                strong_foot_cv is not None
                and best.get("near_foot", 0.0) < 0.25
                and strong_foot_cv.get("near_foot", 0.0) >= 0.55
                and float(strong_foot_cv.get("foot_distance", 1.0)) <= 0.032
                and float(strong_foot_cv["confidence"]) > float(best["confidence"]) + 0.08
            ):
                self.cv_pending_point = None
                self.cv_pending_count = 0
                self.last_debug.update({
                    "selected_source": "CV",
                    "final_confidence": float(strong_foot_cv["confidence"]),
                    "field_valid": True,
                    "temporal_consistency": bool(strong_foot_cv.get("temporal_consistency", False)),
                    "near_foot": float(strong_foot_cv.get("near_foot", 0.0)),
                    "foot_distance": float(strong_foot_cv.get("foot_distance", 1.0)),
                })
                return self._commit(
                    strong_foot_cv["point"],
                    "CV",
                    strong_foot_cv["confidence"],
                    now,
                    field_valid=True,
                    temporal_consistency=bool(strong_foot_cv.get("temporal_consistency", False)),
                )

            # Detector remains authoritative. When an independent CV candidate
            # lands essentially on the same point and passes the stronger field
            # / compactness checks, it is corroborating evidence rather than a
            # replacement source.
            corroboration = None
            for c in cvs:
                distance = float(np.linalg.norm(
                    np.asarray(c["point"], np.float32)
                    - np.asarray(best["point"], np.float32)
                ))
                if distance <= self.cfg.detector_cv_match_distance:
                    if corroboration is None or c["confidence"] > corroboration[0]:
                        corroboration = (c["confidence"], distance, c)

            final_conf = float(best["confidence"])
            if corroboration is not None:
                cv_conf, distance, cv_best = corroboration
                proximity = float(np.clip(
                    1.0 - distance / max(self.cfg.detector_cv_match_distance, 1e-6),
                    0.0, 1.0,
                ))
                bonus = float(self.cfg.detector_cv_bonus * cv_conf * proximity)
                final_conf = float(min(
                    self.cfg.detector_cv_max_conf,
                    final_conf + bonus,
                ))
                self.last_debug.update({
                    "fusion_corroborated": True,
                    "fusion_distance": distance,
                    "fusion_confidence_bonus": bonus,
                })

            self.cv_pending_point = None
            self.cv_pending_count = 0
            self.last_debug.update({
                "near_foot": float(best.get("near_foot", 0.0)),
                "foot_distance": best.get("foot_distance"),
            })
            return self._commit(
                best["point"],
                "DETECTOR",
                final_conf,
                now,
                field_valid=best["field_valid"],
                temporal_consistency=best["temporal_consistency"],
            )

        # CV may recover only when it has strong field/shape evidence and either
        # temporal continuity or a very strong first observation.
        cv_pool = []
        if local_cv is not None and local_cv.get("confidence", 0.0) >= self.cfg.cv_accept_conf:
            cv_pool.append(local_cv)
        for c in cvs:
            if local_cv is not None and c is local_cv:
                continue
            temporal = self._temporal_consistent(c["point"])
            c["temporal_consistency"] = temporal
            score = float(c["confidence"])
            if not temporal and self.previous is not None:
                score *= self.cfg.cv_non_temporal_penalty
            c["confidence"] = score
            if (
                score >= self.cfg.cv_accept_conf
                and (
                    temporal
                    or (
                        score >= self.cfg.cv_first_frame_conf
                        and c.get("mask") == "white"
                        and c.get("area", 0.0) >= self.cfg.cv_first_frame_min_area
                    )
                )
            ):
                cv_pool.append(c)
        if cv_pool:
            # A very strong achromatic, compact candidate on the pitch is allowed
            # as a first-frame acquisition. This is intentionally narrower than
            # ordinary CV recovery and is aimed at the small white eFootball ball.
            if self.previous is None:
                first_frame_pool = [
                    c for c in cv_pool
                    if c.get("mask") == "white"
                    and self.cfg.cv_first_frame_global_min_area <= c.get("area", 0.0) <= self.cfg.cv_first_frame_global_max_area
                    and c.get("circularity_shape", 0.0) >= self.cfg.cv_first_frame_global_min_shape
                    and c.get("background_contrast", 0.0) >= self.cfg.cv_first_frame_global_min_contrast
                ]
                if first_frame_pool:
                    best = max(first_frame_pool, key=lambda c: (c["confidence"], c.get("background_contrast", 0.0), c.get("area", 0.0)))
                    self.cv_pending_point = None
                    self.cv_pending_count = 0
                    self.last_debug.update({
                        "selected_source": "CV",
                        "final_confidence": float(best["confidence"]),
                        "field_valid": True,
                        "temporal_consistency": False,
                        "near_foot": float(best.get("near_foot", 0.0)),
                        "foot_distance": best.get("foot_distance"),
                    })
                    return self._commit(
                        best["point"],
                        "CV",
                        best["confidence"],
                        now,
                        field_valid=True,
                        temporal_consistency=False,
                    )

            # CV acquisition is temporal, not single-frame truth. Keep a pending
            # candidate until a second frame corroborates essentially the same
            # point. This avoids turning pitch highlights/player fragments into
            # a false ball on an isolated probe frame.
            best = max(
                cv_pool,
                key=lambda c: (
                    c["temporal_consistency"],
                    c.get("near_foot", 0.0),
                    c["confidence"],
                    -abs(c["point"][0] - 0.5),
                ),
            )
            point = np.asarray(best["point"], np.float32)
            if self.cv_pending_point is not None:
                pending = np.asarray(self.cv_pending_point, np.float32)
                pending_distance = float(np.linalg.norm(point - pending))
                if pending_distance <= self.cfg.detector_cv_match_distance:
                    self.cv_pending_count += 1
                else:
                    self.cv_pending_point = tuple(best["point"])
                    self.cv_pending_count = 1
            else:
                self.cv_pending_point = tuple(best["point"])
                self.cv_pending_count = 1

            self.last_debug["temporal_consistency"] = self.cv_pending_count >= 2
            if self.cv_pending_count >= 2:
                self.cv_pending_point = None
                self.cv_pending_count = 0
                return self._commit(
                    best["point"],
                    "CV",
                    best["confidence"],
                    now,
                    field_valid=best["field_valid"],
                    temporal_consistency=True,
                )

        # Prediction is strictly downstream of a previously committed measurement.
        if (
            self.previous is not None
            and self.previous_time is not None
            and self.last_real_measurement_time is not None
            and self.confirmed_streak >= 2
            and self.missed < self.cfg.prediction_max_missed
        ):
            age = max(0.0, now - self.last_real_measurement_time)
            bridge_limit = min(float(self.cfg.max_prediction_age_s), 0.20)
            if age <= bridge_limit:
                self.missed += 1
                dt = min(age, 0.10)
                confidence = float(self.previous.confidence * (self.cfg.prediction_decay ** self.missed))
                predicted = Ball(
                    x=float(np.clip(self.previous.x + self.previous.vx * dt, 0, 1)),
                    y=float(np.clip(self.previous.y + self.previous.vy * dt, 0, 1)),
                    vx=self.previous.vx,
                    vy=self.previous.vy,
                    confidence=confidence,
                    source="PREDICTED",
                    predicted=True,
                    valid=confidence >= min(self.cfg.accept_conf, 0.60),
                )
                self.last_debug.update({
                    "selected_source": "PREDICTED",
                    "final_confidence": confidence,
                    "field_valid": True,
                    "prediction_age_s": age,
                })
                return predicted

        self.missed += 1
        self.confirmed_streak = 0
        self.last_debug.update({
            "selected_source": "UNKNOWN",
            "final_confidence": 0.0,
        })
        return Ball(source="UNKNOWN", confidence=0.0, predicted=False, valid=False)
