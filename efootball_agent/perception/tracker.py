from __future__ import annotations

from collections import defaultdict
import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment

from .schema import PlayerDetection, Track, TrackLifecycle


class PlayerTracker:
    def __init__(self, cfg):
        self.cfg = cfg
        self.tracks: dict[int, Track] = {}
        self.next_id = 1
        self.prev_gray = None
        self.prev_timestamp = None
        self.last_flow_count = 0
        self.last_flow_tracks = 0

    def reset(self):
        self.tracks.clear()
        self.next_id = 1
        self.prev_gray = None
        self.prev_timestamp = None
        self.last_flow_count = 0
        self.last_flow_tracks = 0

    @staticmethod
    def _appearance_distance(a, b):
        if a is None or b is None:
            return 0.0
        aa = np.asarray(a, np.float32).reshape(-1)
        bb = np.asarray(b, np.float32).reshape(-1)
        if aa.size == 0 or bb.size == 0:
            return 0.0
        if aa.size != bb.size:
            n = min(aa.size, bb.size)
            aa, bb = aa[:n], bb[:n]
        return float(np.linalg.norm(aa - bb))

    @staticmethod
    def _iou(a, b):
        if a is None or b is None:
            return 0.0
        ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1,bx1), max(ay1,by1)
        ix2, iy2 = min(ax2,bx2), min(ay2,by2)
        inter = max(0, ix2-ix1) * max(0, iy2-iy1)
        aa = max(0, ax2-ax1)*max(0, ay2-ay1)
        bb = max(0, bx2-bx1)*max(0, by2-by1)
        union = aa+bb-inter
        return inter/union if union else 0.0

    def _cost(self, track, det, dt):
        pred = track.predict(dt)
        pos = np.asarray(det.center, np.float32)
        dist = float(np.linalg.norm(pred-pos))
        if dist > self.cfg.match_radius:
            return 999.0
        meas_vel = (pos-track.pos)/max(dt,1e-3)
        vel_err = float(np.linalg.norm(meas_vel-track.vel))
        app = self._appearance_distance(track.appearance, det.appearance)
        team_penalty = self.cfg.team_mismatch_cost if track.team in {"OUR","OPP"} and det.team in {"OUR","OPP"} and track.team != det.team else 0.0
        return float(
            dist
            + self.cfg.velocity_weight*min(vel_err,1.0)
            + self.cfg.appearance_weight*min(app,1.0)
            + team_penalty
            + 0.02*(1-self._iou(track.bbox,det.bbox))
        )

    def _update_team(self, track, det):
        track.team_history.append((det.team, float(det.team_confidence)))
        track.team_history = track.team_history[-9:]
        scores = defaultdict(float)
        for team, conf in track.team_history:
            scores[team] += conf
        best, best_score = max(scores.items(), key=lambda item: (item[1], item[0]))
        total = sum(scores.values()) or 1.0
        best_conf = float(np.clip(best_score / total, 0, 1))
        # UNKNOWN evidence must not erase a previously stable semantic identity.
        if best == "UNKNOWN" and track.team != "UNKNOWN" and track.team_conf >= 0.55:
            return
        track.team = best
        track.team_conf = best_conf

    def _flow_predict(self, frame_gray, track):
        if frame_gray is None or self.prev_gray is None or track.bbox is None:
            return track.predict(1.0 / 30.0), 0.0
        try:
            gray = frame_gray
            x1, y1, x2, y2 = track.bbox
            h, w = gray.shape[:2]
            xa = int(max(0, min(w - 1, x1 * w - 4)))
            xb = int(max(xa + 2, min(w, x2 * w + 4)))
            ya = int(max(0, min(h - 1, y1 * h - 4)))
            yb = int(max(ya + 2, min(h, y2 * h + 4)))
            roi = self.prev_gray[ya:yb, xa:xb]
            if roi.size < 25:
                return track.predict(1.0 / 30.0), 0.0
            pts = cv2.goodFeaturesToTrack(
                roi, maxCorners=8, qualityLevel=0.03, minDistance=3, blockSize=3
            )
            if pts is None or len(pts) < 2:
                return track.predict(1.0 / 30.0), 0.0
            pts[:, 0, 0] += xa
            pts[:, 0, 1] += ya
            nxt, status, err = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, gray, pts, None, winSize=(15, 15), maxLevel=2,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, 0.03)
            )
            if nxt is None or status is None:
                return track.predict(1.0 / 30.0), 0.0
            good = status.reshape(-1) > 0
            if int(good.sum()) < 2:
                return track.predict(1.0 / 30.0), 0.0
            delta = np.median(nxt[good, 0] - pts[good, 0], axis=0)
            dx, dy = float(delta[0]) / w, float(delta[1]) / h
            predicted = np.clip(track.pos + np.array([dx, dy], np.float32), 0.0, 1.0)
            flow_conf = float(np.clip(good.mean(), 0.0, 1.0))
            return predicted, flow_conf
        except Exception:
            return track.predict(1.0 / 30.0), 0.0

    def _cost(self, track, det, dt, pred_override=None):
        pred = np.asarray(pred_override if pred_override is not None else track.predict(dt), np.float32)
        pos = np.asarray(det.center, np.float32)
        dist = float(np.linalg.norm(pred-pos))
        if dist > self.cfg.match_radius:
            return 999.0
        meas_vel = (pos-track.pos)/max(dt,1e-3)
        vel_err = float(np.linalg.norm(meas_vel-track.vel))
        app = self._appearance_distance(track.appearance, det.appearance)
        team_penalty = self.cfg.team_mismatch_cost if track.team in {"OUR","OPP"} and det.team in {"OUR","OPP"} and track.team != det.team else 0.0
        source_bonus = -0.015 if det.source == "RTDETR" else 0.0
        return float(
            dist
            + self.cfg.velocity_weight*min(vel_err,1.0)
            + self.cfg.appearance_weight*min(app,1.0)
            + team_penalty
            + 0.02*(1-self._iou(track.bbox,det.bbox))
            + source_bonus
        )

    def update(self, detections, dt, timestamp, frame=None):
        track_rows = sorted(self.tracks.values(), key=lambda t: t.track_id)
        flow_preds = {}
        flow_conf = {}
        current_gray = None
        if frame is not None:
            current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if self.prev_gray is not None:
                for t in track_rows:
                    pred, conf = self._flow_predict(current_gray, t)
                    flow_preds[t.track_id] = pred
                    flow_conf[t.track_id] = conf
        self.last_flow_tracks = len(flow_preds)
        self.last_flow_count = sum(1 for v in flow_conf.values() if v >= self.cfg.flow_min_conf)

        matched_tracks = set()
        matched_dets = set()
        if track_rows and detections:
            cost = np.array(
                [[self._cost(t,d,dt,flow_preds.get(t.track_id)) for d in detections] for t in track_rows],
                dtype=np.float32,
            )
            rows, cols = linear_sum_assignment(cost)
            for r,c in zip(rows,cols):
                if cost[r,c] >= 999:
                    continue
                track, det = track_rows[r], detections[c]
                raw_pos = np.asarray(det.center, np.float32)
                reference = np.asarray(flow_preds.get(track.track_id, track.predict(dt)), np.float32)
                if det.source == "RTDETR_STALE":
                    # Old async detector geometry is identity evidence, not a
                    # current position measurement. Prefer current-frame flow
                    # and use the stale detector only as a bounded correction.
                    pos = (0.78 * reference + 0.22 * raw_pos).astype(np.float32)
                else:
                    pos = raw_pos
                measured = (pos-reference)/max(dt,1e-3)
                flow_w = min(0.35, 0.35 * flow_conf.get(track.track_id, 0.0))
                track.vel = ((1.0-flow_w)*track.vel + 0.35*measured).astype(np.float32)
                track.pos = pos
                track.bbox = det.bbox
                track.confidence = float(np.clip(0.68*track.confidence+0.32*det.confidence,0,.99))
                if det.appearance:
                    track.appearance = (0.78*track.appearance+0.22*np.asarray(det.appearance,np.float32)).astype(np.float32)
                track.age += 1
                track.hits += 1
                track.missed = 0
                track.lifecycle = TrackLifecycle.CONFIRMED if track.hits >= self.cfg.tentative_hits else TrackLifecycle.TENTATIVE
                track.last_timestamp = timestamp
                self._update_team(track,det)
                matched_tracks.add(track.track_id)
                matched_dets.add(c)

        for track_id, track in list(self.tracks.items()):
            if track_id in matched_tracks:
                continue
            if track_id in flow_preds and flow_conf.get(track_id, 0.0) >= self.cfg.flow_min_conf:
                target = flow_preds[track_id]
                old = track.pos.copy()
                track.pos = target
                delta = (target-old).astype(np.float32)
                track.vel = (0.72*track.vel + 0.28*(delta/max(dt,1e-3))).astype(np.float32)
                if track.bbox is not None:
                    x1, y1, x2, y2 = track.bbox
                    track.bbox = (
                        float(np.clip(x1 + delta[0], 0.0, 1.0)),
                        float(np.clip(y1 + delta[1], 0.0, 1.0)),
                        float(np.clip(x2 + delta[0], 0.0, 1.0)),
                        float(np.clip(y2 + delta[1], 0.0, 1.0)),
                    )
            else:
                track.pos = np.clip(track.pos + track.vel*min(dt,.12)*.45,0,1)
            track.missed += 1
            track.age += 1
            track.confidence *= 0.94 if flow_conf.get(track_id, 0.0) >= 0.30 else 0.91
            if track.missed > self.cfg.max_missed:
                track.lifecycle = TrackLifecycle.DEAD
                del self.tracks[track_id]
            else:
                track.lifecycle = TrackLifecycle.LOST

        detector_present = any(d.source == "RTDETR" for d in detections)
        cv_spawned = 0
        max_tracks = max(11, int(getattr(self.cfg, "max_tracks", 22)))
        current_tracks = sorted(self.tracks.values(), key=lambda t: t.track_id)
        for j, det in enumerate(detections):
            if j in matched_dets:
                continue
            if len(self.tracks) >= max_tracks:
                break
            # Never spawn a duplicate identity just because the detector bbox
            # moved slightly. Compare against the predicted position of every
            # existing track before allocating a new ID.
            duplicate = False
            det_pos = np.asarray(det.center, np.float32)
            for t in current_tracks:
                pred = np.asarray(t.predict(dt), np.float32)
                if float(np.linalg.norm(pred - det_pos)) <= 0.045:
                    duplicate = True
                    break
            if duplicate:
                continue
            if det.source == "CV_RECOVERY":
                # CV recovery fills gaps but must not create a second roster from
                # field highlights. Prefer at most two short-lived recovery tracks
                # only when the global detector is absent.
                if detector_present or cv_spawned >= 2 or len(self.tracks) >= max_tracks - 3:
                    continue
                cv_spawned += 1
            self.tracks[self.next_id] = Track(
                track_id=self.next_id,
                pos=np.asarray(det.center,np.float32),
                vel=np.zeros(2,np.float32),
                bbox=det.bbox,
                team=det.team,
                team_conf=float(det.team_confidence),
                confidence=float(det.confidence),
                appearance=np.asarray(det.appearance if det.appearance else np.zeros(6),np.float32),
                role=det.role,
                last_timestamp=timestamp,
                team_history=[(det.team,float(det.team_confidence))],
            )
            self.next_id += 1
            current_tracks.append(self.tracks[self.next_id - 1])

        self.prev_gray = current_gray
        self.prev_timestamp = timestamp
        return self.snapshot()

    def snapshot(self):
        return [
            Track(
                track_id=t.track_id,
                pos=t.pos.copy(),
                vel=t.vel.copy(),
                bbox=t.bbox,
                team=t.team,
                team_conf=t.team_conf,
                role=t.role,
                confidence=t.confidence,
                appearance=t.appearance.copy(),
                age=t.age,
                hits=t.hits,
                missed=t.missed,
                lifecycle=t.lifecycle,
                last_timestamp=t.last_timestamp,
                team_history=list(t.team_history),
            )
            for t in sorted(self.tracks.values(), key=lambda t:t.track_id)
        ]
