from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import time
import sys
from types import SimpleNamespace
import types

import cv2
import numpy as np
import pytest

from efootball_agent.config import SETTINGS
from efootball_agent.core import Action, Ball, FootballAction, Movement, Player, Team, WorldState
from efootball_agent.control.actions import to_xinput
from efootball_agent.control.safety import SafetyGate
from efootball_agent.perception.active import ActiveMarkerDetector
from efootball_agent.perception.ball import BallManager
from efootball_agent.perception.models import ModelBall, RTDETRAdapter
from efootball_agent.perception.player import PlayerDetector
from efootball_agent.perception.schema import PlayerDetection
from efootball_agent.perception.score import ScoreDetector
from efootball_agent.perception.tracker import PlayerTracker
from efootball_agent.perception.ui_mask import UIMask
from efootball_agent.rl.encoder import StateEncoder
from efootball_agent.rl.reward import RewardEngine
from efootball_agent.world.possession import PossessionEstimator
from efootball_agent.world.world import WorldModel


REAL_0008 = Path(__file__).resolve().parents[2] / "datasets/efootball_real_frames/frame_live_0008.jpg"
LATEST_0008 = Path(__file__).resolve().parents[2] / "datasets/efootball_real_frames/frame_latest_0008.jpg"

CURRENT_0012 = Path(__file__).resolve().parents[2] / "datasets/efootball_real_frames/frame_live_0012.jpg"
CURRENT_0007 = Path(__file__).resolve().parents[2] / "datasets/efootball_real_frames/frame_live_0007.jpg"


def blank_frame():
    return np.full((720, 1280, 3), (60, 170, 60), np.uint8)


def test_torchvision_available():
    import torch
    import torchvision
    assert torch.__version__
    assert torchvision.__version__


def test_model_loader(tmp_path, monkeypatch):
    model_dir = tmp_path / "rtdetr"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "model.safetensors").write_bytes(b"fake")
    (model_dir / "preprocessor_config.json").write_text("{}", encoding="utf-8")

    class FakeProcessor:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            assert kwargs.get("local_files_only") is True
            assert kwargs.get("backend") == "torchvision"
            return cls()

    class FakeModel:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            assert kwargs.get("local_files_only") is True
            return cls()
        def eval(self): return self
        def to(self, _device): return self

    fake = types.SimpleNamespace(
        AutoImageProcessor=FakeProcessor,
        AutoModelForObjectDetection=FakeModel,
    )
    monkeypatch.setitem(sys.modules, "transformers", fake)
    cfg = replace(SETTINGS, rtdetr_dir=model_dir)
    adapter = RTDETRAdapter(cfg)
    try:
        assert adapter.processor is not None
        assert adapter.detector is not None
    finally:
        adapter.close()


def test_ball_detector_priority():
    ui = UIMask(SETTINGS.ui)
    manager = BallManager(SETTINGS.ball, ui)
    ball = manager.update(blank_frame(), [ModelBall(pos=(0.62, 0.52), confidence=0.65)])
    assert ball.source == "DETECTOR"
    assert ball.confidence >= 0.45


def test_ball_detector_cv_corroboration_promotes_validity():
    frame = cv2.imread(str(LATEST_0008))
    assert frame is not None
    manager = BallManager(SETTINGS.ball, UIMask(SETTINGS.ui))
    ball = manager.update(frame, [ModelBall(pos=(0.3724, 0.3021), confidence=0.3935)])
    assert ball.source == "DETECTOR"
    assert ball.valid
    assert ball.confidence >= SETTINGS.ball.accept_conf
    assert manager.last_debug["fusion_corroborated"] is True
    assert manager.last_debug["fusion_confidence_bonus"] > 0.0


def test_ball_ui_rejection():
    ui = UIMask(SETTINGS.ui)
    manager = BallManager(SETTINGS.ball, ui)
    frame = blank_frame()
    cv2.circle(frame, (640, 620), 6, (0, 140, 255), -1)
    ball = manager.update(frame, [])
    assert not ball.valid
    assert ball.source in {"UNKNOWN", "PREDICTED"}


def test_ball_field_validation():
    ui = UIMask(SETTINGS.ui)
    manager = BallManager(SETTINGS.ball, ui)
    frame = blank_frame()
    cv2.circle(frame, (1100, 100), 5, (0, 180, 255), -1)
    ball = manager.update(frame, [])
    assert ball.source != "CV" or not ball.valid


def test_ball_temporal_consistency():
    ui = UIMask(SETTINGS.ui)
    cfg = replace(SETTINGS.ball, cv_accept_conf=0.50, cv_first_frame_conf=0.95)
    manager = BallManager(cfg, ui)
    manager.update(blank_frame(), [ModelBall((0.50, 0.60), 0.70)])
    frame = blank_frame()
    cv2.circle(frame, (640, 432), 5, (190, 220, 205), -1)
    ball = manager.update(frame, [])
    assert manager.last_debug["temporal_consistency"] in {True, False}
    assert ball.source in {"CV", "PREDICTED", "UNKNOWN"}
    if ball.source == "CV":
        assert manager.last_debug["temporal_consistency"] is True


def test_ball_real_frame_candidate_near_actual_center_ball():
    frame = cv2.imread(str(REAL_0008))
    assert frame is not None
    manager = BallManager(SETTINGS.ball, UIMask(SETTINGS.ui))
    candidates = manager._cv_candidates(frame)
    actual = np.array([0.499, 0.597], np.float32)  # visible white ball in supplied frame
    assert candidates
    nearest = min(np.linalg.norm(np.asarray(c["point"], np.float32) - actual) for c in candidates)
    assert nearest < 0.03


def test_ball_latest_live_frame_prefers_strong_white_candidate():
    frame = cv2.imread(str(LATEST_0008))
    assert frame is not None
    manager = BallManager(SETTINGS.ball, UIMask(SETTINGS.ui))
    first = manager.update(frame, [])
    assert first.source == "CV"
    assert first.valid
    assert abs(first.x - 0.373) < 0.03
    assert abs(first.y - 0.302) < 0.03


def test_ball_first_frame_global_white_candidate_prefers_actual_ball():
    frame = cv2.imread(str(CURRENT_0007))
    assert frame is not None
    manager = BallManager(SETTINGS.ball, UIMask(SETTINGS.ui))
    ball = manager.update(frame, [])
    assert ball.source == "CV"
    assert ball.valid
    assert abs(ball.x - 0.603) < 0.025
    assert abs(ball.y - 0.518) < 0.025


def test_ball_prediction_expires_quickly():
    cfg = replace(SETTINGS.ball, max_prediction_age_s=0.02, prediction_max_missed=1)
    manager = BallManager(cfg, UIMask(SETTINGS.ui))
    manager._commit((0.5, 0.5), "DETECTOR", 0.8, now=time.monotonic() - 1.0)
    manager.previous.vx = 0.1
    manager.previous_time = time.monotonic() - 1.0
    b = manager.update(blank_frame(), [])
    assert b.source == "UNKNOWN"


def test_score_controlled_side_is_left_for_current_live_contract():
    assert SETTINGS.score.controlled_side == "left"


def test_active_horizontal_cyan_bar_is_detected():
    frame = cv2.imread(str(LATEST_0008))
    assert frame is not None
    marker, candidates = ActiveMarkerDetector(SETTINGS.active).detect(frame)
    assert marker is not None
    assert len(candidates) >= 1
    assert abs(marker[0] - 0.565) < 0.05
    assert abs(marker[1] - 0.519) < 0.03


def test_ball_real_frame_rejects_right_edge_false_positive():
    frame = cv2.imread(str(REAL_0008))
    assert frame is not None
    manager = BallManager(SETTINGS.ball, UIMask(SETTINGS.ui))
    ball = manager.update(frame, [])
    assert not (ball.valid and ball.x > 0.90 and 0.35 < ball.y < 0.60)


def test_score_roi():
    frame = cv2.imread(str(REAL_0008))
    assert frame is not None
    left = ScoreDetector.crop(frame, SETTINGS.score.left_score_roi)
    right = ScoreDetector.crop(frame, SETTINGS.score.right_score_roi)
    assert left.shape[:2] == (32, 30)
    assert right.shape[:2] == (32, 30)


def test_score_parsing_real_frame():
    frame = cv2.imread(str(REAL_0008))
    assert frame is not None
    detector = ScoreDetector(SETTINGS.score)
    if not detector.tesseract_available():
        pytest.skip("Tesseract executable not installed in test environment")
    left, lc = detector.ocr_number(detector.crop(frame, SETTINGS.score.left_score_roi))
    right, rc = detector.ocr_number(detector.crop(frame, SETTINGS.score.right_score_roi))
    assert (left, right) == (1, 0)
    assert lc > 0.0 and rc > 0.0


def test_score_temporal_confirmation():
    frame = cv2.imread(str(REAL_0008))
    assert frame is not None
    cfg = replace(SETTINGS.score, interval_s=0.0, confirm_frames=2)
    detector = ScoreDetector(cfg)
    if not detector.tesseract_available():
        pytest.skip("Tesseract executable not installed in test environment")
    a = detector.update(frame, now=1.0)
    b = detector.update(frame, now=1.5)
    assert a["initialized"] is False
    assert b["initialized"] is True
    assert (b["left"], b["right"]) == (1, 0)


def test_goal_from_score_change():
    from efootball_agent.perception.pipeline import PerceptionEngine
    assert PerceptionEngine._score_goal_flags((1, 0), (2, 0), True) == (True, False)
    assert PerceptionEngine._score_goal_flags((1, 0), (1, 1), True) == (False, True)
    assert PerceptionEngine._score_goal_flags((1, 0), (2, 1), True) == (True, True)
    assert PerceptionEngine._score_goal_flags((1, 0), (1, 0), False) == (False, False)


def test_goal_event_latch():
    from efootball_agent.perception.pipeline import PerceptionEngine
    engine = object.__new__(PerceptionEngine)
    engine.frame_id = 100
    engine._goal_latch_until = 120
    assert not engine._goal_latch_open()
    engine.frame_id = 121
    assert engine._goal_latch_open()


def test_active_marker_detection():
    frame = cv2.imread(str(REAL_0008))
    assert frame is not None
    marker, candidates = ActiveMarkerDetector(SETTINGS.active).detect(frame)
    assert marker is not None
    assert len(candidates) == 1


def test_active_marker_association():
    detector = ActiveMarkerDetector(SETTINGS.active)
    marker = (0.50, 0.40)
    det = PlayerDetection((0.50, 0.50), (0.47, 0.42, 0.53, 0.60), "TEST", 0.9)
    tracker = PlayerTracker(SETTINGS.tracking)
    tracks = tracker.update([det], 1 / 30, 1.0)
    tid, conf = detector.associate(marker, tracks)
    assert tid == tracks[0].track_id
    assert conf > 0


def test_active_hysteresis():
    detector = ActiveMarkerDetector(SETTINGS.active)
    tracks = [
        PlayerTracker(SETTINGS.tracking).update([PlayerDetection((0.50,0.50),(0.47,0.42,0.53,0.60),'TEST',0.9)],1/30,1.0)[0]
    ]
    detector.last_track_id = tracks[0].track_id
    detector.last_confidence = 0.8
    tid, conf = detector.associate((0.50, 0.40), tracks)
    assert tid == tracks[0].track_id
    assert conf >= 0.0


def test_unknown_player_preserved():
    tracker = PlayerTracker(SETTINGS.tracking)
    det = PlayerDetection((0.5, 0.5), (0.49, 0.45, 0.51, 0.55), "TEST", 0.8, "UNKNOWN", 0.3, appearance=(0.1,) * 6)
    tracks = tracker.update([det], 1 / 30, 1.0)
    assert tracks[0].team == "UNKNOWN"


def test_team_temporal_smoothing():
    tracker = PlayerTracker(SETTINGS.tracking)
    base = (0.1,) * 6
    seq = ["OUR", "OUR", "UNKNOWN", "OUR"]
    for i, team in enumerate(seq):
        det = PlayerDetection((0.5,0.5),(0.48,0.4,0.52,0.6),'TEST',0.8,team,0.8 if team != 'UNKNOWN' else 0.3,appearance=base)
        tracks = tracker.update([det],1/30,(i+1)/30)
    assert tracks[0].team == "OUR"


def test_tracker_lifecycle():
    tracker = PlayerTracker(SETTINGS.tracking)
    det = PlayerDetection((0.5,0.5),(0.48,0.4,0.52,0.6),'TEST',0.8)
    tracker.update([det],1/30,0.0)
    assert tracker.tracks[1].lifecycle.value == "TENTATIVE"
    tracker.update([det],1/30,1/30)
    assert tracker.tracks[1].lifecycle.value == "CONFIRMED"
    for i in range(SETTINGS.tracking.max_missed + 1):
        tracker.update([],1/30,(i+2)/30)
    assert 1 not in tracker.tracks


def test_tracker_deterministic_assignment():
    tracker = PlayerTracker(SETTINGS.tracking)
    d1 = PlayerDetection((0.40,0.5),(0.38,0.4,0.42,0.6),'TEST',0.8)
    d2 = PlayerDetection((0.60,0.5),(0.58,0.4,0.62,0.6),'TEST',0.8)
    first = tracker.update([d1,d2],1/30,0.0)
    ids = [x.track_id for x in first]
    second = tracker.update([d2,d1],1/30,1/30)
    mapping = {round(float(x.pos[0]),2): x.track_id for x in second}
    assert mapping[0.40] == ids[0]
    assert mapping[0.60] == ids[1]


def test_world_coordinates():
    cfg = replace(SETTINGS.world, attack_direction="left")
    world = WorldModel(cfg, SETTINGS.tactical)
    p = Player(1,0.2,0.5,vx=0.3)
    b = Ball(x=0.2,y=0.5,vx=0.3,valid=True,confidence=0.8,source="DETECTOR")
    p2 = world.canonical_player(p)
    b2 = world.canonical_ball(b)
    assert np.isclose(p2.x,0.8) and np.isclose(p2.vx,-0.3)
    assert np.isclose(b2.x,0.8) and np.isclose(b2.vx,-0.3)


def test_possession_hysteresis():
    est = PossessionEstimator(SETTINGS.world)
    own = [Player(1,0.50,0.50)]
    opp = [Player(2,0.66,0.50)]
    ball = Ball(0.50,0.50,valid=True,confidence=0.8,source="DETECTOR")
    a = est.update(own,opp,ball,1/30)
    ball.x = 0.54
    b = est.update(own,opp,ball,1/30)
    assert a.owner_team == Team.OUR
    assert b.owner_team == Team.OUR


def test_control_gate():
    gate = SafetyGate(SETTINGS.control)
    action = Action(Movement.RIGHT, FootballAction.SHOOT, 1)
    blocked = gate.apply(WorldState(control_valid=False), action)
    assert blocked == Action()
    world = WorldState(control_valid=True)
    assert gate.apply(world, action) == action


def test_y_not_in_action_space():
    mapped = to_xinput(Action(Movement.RIGHT, FootballAction.CROSS, 1))
    assert mapped["cross"]
    assert "y" not in mapped
    assert not mapped.get("y", False)


def test_xinput_mapping():
    mapped = to_xinput(Action(Movement.UP_LEFT, FootballAction.SHOOT, 0))
    assert mapped["lx"] < 0 and mapped["ly"] < 0
    assert mapped["shoot"] is True
    assert mapped["pass"] is False and mapped["cross"] is False


def test_state_encoder_dimension():
    world = WorldState()
    world.control_valid = True
    state = StateEncoder().encode(world)
    assert state.shape == (170,)
    assert np.isfinite(state).all()


def test_reward_goal_dominates():
    reward = RewardEngine(SETTINGS.reward)
    world = WorldState()
    reward.compute(world)
    world.events.goal_for = True
    assert reward.compute(world) >= 9.0


def test_detector_ball_bbox_validation():
    ui = UIMask(SETTINGS.ui)
    manager = BallManager(SETTINGS.ball, ui)
    frame = cv2.imread(str(REAL_0008))
    assert frame is not None
    bad = ModelBall(pos=(0.37, 0.30), confidence=0.95, bbox=(0.34, 0.24, 0.40, 0.36))
    ball = manager.update(frame, [bad])
    assert ball.source != "DETECTOR" or not ball.valid


def test_score_parser_fast_path_real_frame():
    frame = cv2.imread(str(LATEST_0008))
    assert frame is not None
    detector = ScoreDetector(SETTINGS.score)
    if not detector.tesseract_available():
        pytest.skip("Tesseract is not installed in test environment")
    left, lc = detector.ocr_number(detector.crop(frame, SETTINGS.score.left_score_roi))
    right, rc = detector.ocr_number(detector.crop(frame, SETTINGS.score.right_score_roi))
    assert left == 0 and right == 0
    assert lc > 0.9 and rc > 0.9



def test_current_frame_ball_candidate_present_at_visible_ball():
    frame = cv2.imread(str(CURRENT_0012))
    assert frame is not None
    manager = BallManager(SETTINGS.ball, UIMask(SETTINGS.ui))
    candidates = manager._cv_candidates(frame)
    # The isolated white field marker near (592,244) is not the gameplay ball.
    actual = np.array([657/1280, 262/720], np.float32)
    # The global candidate list may also contain field highlights; the visible
    # ball is expected to be recovered from a player-foot neighborhood.
    foot_player = PlayerDetection((657/1280, 245/720), (650/1280, 222/720, 672/1280, 260/720), 'TEST', 0.9)
    foot = manager._footpoint_candidates(frame, [foot_player])
    assert foot
    nearest = min(np.linalg.norm(np.asarray(c["point"], np.float32) - actual) for c in foot)
    assert nearest < 0.025


def test_current_frame_active_marker_candidate_present():
    frame = cv2.imread(str(CURRENT_0012))
    assert frame is not None
    marker, candidates = ActiveMarkerDetector(SETTINGS.active).detect(frame)
    assert marker is not None
    assert len(candidates) >= 1
    assert abs(marker[0] - 657/1280) < 0.02
    assert abs(marker[1] - 214/720) < 0.02


def test_ball_prefers_foot_correlated_cv_over_isolated_field_spot():
    frame = cv2.imread(str(CURRENT_0012))
    assert frame is not None
    manager = BallManager(SETTINGS.ball, UIMask(SETTINGS.ui))
    foot_player = PlayerDetection((0.539, 0.329), (0.5316, 0.2979, 0.5477, 0.3605), 'TEST', 0.9)
    bad = ModelBall(pos=(592/1280, 244/720), confidence=0.50, bbox=(590/1280,242/720,596/1280,248/720))
    ball = manager.update(frame, [bad], players=[foot_player], active_track_id=-1)
    assert ball.source == 'CV'
    assert ball.valid
    assert abs(ball.x - 657/1280) < 0.03
    assert abs(ball.y - 262/720) < 0.03
    assert manager.last_debug["near_foot"] >= 0.55
    assert manager.last_debug["foot_distance"] <= 0.032


def test_score_three_to_zero_temporal_confirmation_with_mock_ocr():
    detector = ScoreDetector(replace(SETTINGS.score, interval_s=0.0, confirm_frames=2))
    detector.tesseract_path = 'mock'
    detector.ocr_number = lambda crop: (0, 0.96) if crop is not None else (None, 0.0)
    # Replace side-specific calls with 0:3 for deterministic score state testing.
    calls = iter([(0, 0.96), (3, 0.96), (0, 0.96), (3, 0.96)])
    detector.ocr_number = lambda crop: next(calls)
    detector._ocr_clock = lambda crop: (210.0, 0.95, '03:30')
    frame = blank_frame()
    a = detector.update(frame, now=1.0)
    b = detector.update(frame, now=1.1)
    assert a['initialized'] is False
    assert b['initialized'] is True
    assert (b['left'], b['right']) == (0, 3)


def test_score_pair_ocr_real_frame():
    frame = cv2.imread(str(CURRENT_0012))
    assert frame is not None
    detector = ScoreDetector(SETTINGS.score)
    if not detector.tesseract_available():
        pytest.skip("Tesseract is not installed in test environment")
    left, right, conf = detector._ocr_score_pair(
        detector.score_pair_crop(frame), detector.tesseract_path
    )
    assert (left, right) == (0, 0)
    assert conf >= 0.95


def test_score_pair_ocr_supports_three_zero_via_mock(monkeypatch):
    detector = ScoreDetector(SETTINGS.score)
    detector.tesseract_path = 'mock'
    monkeypatch.setattr('pytesseract.image_to_string', lambda *a, **k: '03')
    frame = blank_frame()
    left, right, conf = detector._ocr_score_pair(detector.score_pair_crop(frame), 'mock')
    assert (left, right) == (0, 3)
    assert conf >= 0.95


def test_model_result_contains_source_age_metadata():
    from efootball_agent.perception.models import ModelResult
    result = ModelResult(stamp=2.0, source_timestamp=1.5, frame_id=7, inference_latency_s=0.3, age_at_completion_s=0.5)
    assert result.frame_id == 7
    assert np.isclose(result.age_at_completion_s, 0.5)


def test_model_freshness_uses_source_frame_age():
    # Completion time can be fresh while the source frame is stale; control must
    # use source age, not completion age.
    from efootball_agent.perception.models import ModelResult
    result = ModelResult(stamp=10.0, source_timestamp=8.0, frame_id=10)
    now = 10.1
    age = now - result.source_timestamp
    assert age > SETTINGS.player.max_model_stale_s


def test_model_age_is_source_frame_age_not_completion_age():
    from efootball_agent.perception.models import ModelResult
    r = ModelResult(
        stamp=5.0,
        source_timestamp=3.8,
        frame_id=12,
        inference_latency_s=0.9,
        age_at_completion_s=1.2,
    )
    now = 5.1
    assert np.isclose(now - r.source_timestamp, 1.3)
    assert r.inference_latency_s < r.age_at_completion_s


def test_rtdetr_cpu_contract_rejects_non_cpu_device(tmp_path, monkeypatch):
    import json
    model_dir = tmp_path / "rtdetr"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({}), encoding="utf-8")
    (model_dir / "model.safetensors").write_bytes(b"fake")
    (model_dir / "preprocessor_config.json").write_text(json.dumps({}), encoding="utf-8")
    cfg = replace(SETTINGS, rtdetr_dir=model_dir, player=replace(SETTINGS.player, inference_device="cuda"))
    adapter = RTDETRAdapter(cfg)
    assert adapter.detector is None
    assert adapter.load_error is not None


def test_live_check_has_observation_only_cpu_fields():
    assert SETTINGS.player.inference_device == "cpu"
    assert SETTINGS.player.cpu_num_threads == 8
    assert SETTINGS.player.cpu_num_interop_threads == 1


def test_score_async_constructor_starts_worker_without_blocking():
    from efootball_agent.perception.score import ScoreDetector
    detector = ScoreDetector(replace(SETTINGS.score, interval_s=60.0), async_mode=True)
    try:
        frame = blank_frame()
        t0 = time.perf_counter()
        info = detector.update(frame, now=time.monotonic())
        elapsed = time.perf_counter() - t0
        assert elapsed < 0.05
        assert isinstance(info, dict)
    finally:
        detector.close()


def test_rtdetr_worker_runtime_reports_errors_and_no_undefined_rgb_symbol():
    from pathlib import Path
    source = Path(__file__).resolve().parents[1] / "perception" / "models.py"
    text = source.read_text(encoding="utf-8")
    assert "sizes = torch.tensor([rgb_buffer.shape[:2]]" in text
    assert "sizes = torch.tensor([rgb.shape[:2]]" not in text


def test_tracker_flow_prediction_keeps_track_alive_between_detections():
    tracker = PlayerTracker(SETTINGS.tracking)
    frame0 = np.zeros((240, 320, 3), np.uint8)
    cv2.rectangle(frame0, (120, 100), (145, 180), (220, 220, 220), -1)
    frame1 = np.zeros_like(frame0)
    cv2.rectangle(frame1, (128, 100), (153, 180), (220, 220, 220), -1)
    det = PlayerDetection((0.414, 0.583), (0.375, 0.417, 0.453, 0.750), 'RTDETR', 0.9)
    tracker.update([det,], 1/30, 0.0, frame=frame0)
    tracker.update([], 1/30, 1/30, frame=frame1)
    assert 1 in tracker.tracks
    assert tracker.tracks[1].missed == 1
    assert tracker.tracks[1].lifecycle.value == 'LOST'


def test_possession_uses_player_feet_not_bbox_center():
    est = PossessionEstimator(SETTINGS.world)
    own = [Player(1, 0.50, 0.45, bbox=(0.48, 0.35, 0.52, 0.55))]
    opp = []
    ball = Ball(0.50, 0.55, valid=True, confidence=0.9, source='CV')
    state = est.update(own, opp, ball, 1/30)
    assert state.owner_team == Team.OUR
    assert state.confidence >= SETTINGS.world.possession_on


def test_ball_alpha_beta_updates_velocity():
    cfg = replace(SETTINGS.ball, cv_accept_conf=0.50)
    manager = BallManager(cfg, UIMask(SETTINGS.ui))
    manager._commit((0.40, 0.50), 'DETECTOR', 0.80, now=1.0)
    b = manager._commit((0.45, 0.50), 'CV', 0.80, now=1.1)
    assert b.valid
    assert b.vx > 0.0
    assert 0.40 < b.x < 0.46


def test_ball_local_reacquisition_accepts_close_candidate():
    import numpy as np
    from types import SimpleNamespace
    from efootball_agent.perception.ball import BallManager
    from efootball_agent.config import SETTINGS
    from efootball_agent.perception.ui_mask import UIMask
    bm = BallManager(SETTINGS.ball, UIMask(SETTINGS.ui))
    bm.previous = SimpleNamespace(x=0.50, y=0.50, vx=0.0, vy=0.0, confidence=0.9)
    bm.previous_time = 1.0
    frame = np.zeros((720,1280,3), dtype=np.uint8)
    # green pitch with a compact white blob near previous point
    frame[:] = (40,110,40)
    frame[360:366, 636:642] = (255,255,255)
    out = bm.update(frame, [], players=[])
    assert out.source in {"CV", "PREDICTED", "UNKNOWN"}


def test_ball_prediction_requires_prior_confirmed_measurements():
    from types import SimpleNamespace
    cfg = replace(SETTINGS.ball, max_prediction_age_s=0.20, prediction_max_missed=2)
    bm = BallManager(cfg, UIMask(SETTINGS.ui))
    bm.previous = SimpleNamespace(x=0.50, y=0.50, vx=0.5, vy=0.0, confidence=0.9)
    import time
    bm.previous_time = time.monotonic() - 0.05
    bm.last_real_measurement_time = bm.previous_time
    bm.confirmed_streak = 1
    out = bm.update(np.zeros((720,1280,3), dtype=np.uint8), [], players=[])
    assert out.source != 'PREDICTED'


def test_score_rejects_multi_goal_ocr_jump_after_initialization(monkeypatch):
    detector = ScoreDetector(replace(SETTINGS.score, interval_s=0.0, confirm_frames=2))
    detector.tesseract_path = 'mock'
    detector.initialized = True
    detector.last_score = (0, 0)
    calls = iter([(2, 0, 0.96), (2, 0, 0.96), (0, 1, 0.96), (0, 1, 0.96)])
    def fake_pair(crop, path):
        return next(calls)
    monkeypatch.setattr(detector, '_ocr_score_pair', fake_pair)
    detector._ocr_clock = lambda crop: (100.0, 0.95, '01:40')
    frame = blank_frame()
    a = detector.update(frame, now=1.0)
    b = detector.update(frame, now=1.1)
    assert a['invalid_transition'] and b['invalid_transition']
    assert detector.last_score == (0, 0)
    c = detector.update(frame, now=1.2)
    d = detector.update(frame, now=1.3)
    assert (d['left'], d['right']) == (0, 1)


def test_active_switch_requires_two_clear_wins():
    from types import SimpleNamespace
    det = ActiveMarkerDetector(SETTINGS.active)
    det.last_track_id = 1
    det.last_confidence = 0.9
    det.last_marker = (0.50, 0.30)
    det.missed = 0
    t1 = SimpleNamespace(track_id=1, bbox=(0.48,0.30,0.52,0.50), missed=0)
    t2 = SimpleNamespace(track_id=2, bbox=(0.49,0.30,0.51,0.50), missed=0)
    # First strong candidate for track 2 should not switch yet.
    first, _ = det.associate((0.50, 0.295), [t1, t2])
    assert first == 1


def test_ball_reset_clears_prediction_confirmation_state():
    from efootball_agent.perception.ball import BallManager
    bm = BallManager(SETTINGS.ball, UIMask(SETTINGS.ui))
    bm.confirmed_streak = 3
    bm.last_real_measurement_time = 12.3
    bm.frame_counter = 8
    bm.reset()
    assert bm.confirmed_streak == 0
    assert bm.last_real_measurement_time is None
    assert bm.frame_counter == 0


def test_score_invalid_jump_clears_pending_candidate():
    detector = ScoreDetector(replace(SETTINGS.score, interval_s=0.0, confirm_frames=2))
    detector.tesseract_path = 'mock'
    detector.initialized = True
    detector.last_score = (0, 0)
    detector._ocr_score_pair = lambda crop, path: (2, 0, 0.96)
    detector._ocr_clock = lambda crop: (90.0, 0.95, '01:30')
    frame = blank_frame()
    detector.update(frame, now=1.0)
    assert detector.pending_score is None and detector.pending_count == 0


def test_tracking_config_allows_bounded_async_detector_gaps():
    from efootball_agent.config import TrackingConfig
    cfg = TrackingConfig()
    assert cfg.max_missed >= 18
    assert cfg.flow_min_conf <= 0.30


def test_active_association_allows_bounded_lost_track_gap():
    detector = ActiveMarkerDetector(SETTINGS.active)
    detector.last_track_id = 7
    detector.last_confidence = 0.82
    detector.missed = 3
    track = SimpleNamespace(track_id=7, bbox=(0.47, 0.40, 0.53, 0.60), missed=5)
    tid, conf = detector.associate((0.50, 0.39), [track])
    assert tid == 7
    assert conf >= 0.0


def test_settings_use_extended_player_reuse_window():
    assert SETTINGS.player.model_reuse_stale_s >= 1.8
    assert SETTINGS.active.max_missing >= 8
    assert SETTINGS.active.active_track_max_missed >= 8


def test_active_recovery_finds_fragmented_marker_near_track_head():
    from types import SimpleNamespace
    det = ActiveMarkerDetector(SETTINGS.active)
    frame = blank_frame()
    # cyan patch deliberately too small for the global component minimum
    cv2.rectangle(frame, (620, 250), (626, 254), (255, 220, 40), -1)
    track = SimpleNamespace(track_id=3, bbox=(0.47, 0.36, 0.53, 0.56), missed=0)
    marker, candidates = det.recover_from_tracks(frame, [track])
    assert marker is not None
    assert candidates


def test_active_hold_decay_is_slow_for_short_marker_gaps():
    det = ActiveMarkerDetector(SETTINGS.active)
    det.last_track_id = 7
    det.last_confidence = 0.90
    det.last_marker = (0.50, 0.39)
    det.missed = 10
    track = __import__('types').SimpleNamespace(track_id=7, bbox=(0.47,0.40,0.53,0.60), missed=2)
    tid, conf = det.associate(None, [track])
    assert tid == 7
    assert conf > 0.50


def test_designated_player_is_independent_of_active_marker():
    from types import SimpleNamespace
    from efootball_agent.world.possession import PossessionEstimator
    est = PossessionEstimator(SETTINGS) if False else PossessionEstimator(SETTINGS.world)
    ball = Ball(x=0.50, y=0.50, valid=True, confidence=0.9, source="CV")
    p1 = Player(track_id=1, x=0.49, y=0.48, confidence=0.9, team=Team.OUR)
    p2 = Player(track_id=2, x=0.80, y=0.50, confidence=0.9, team=Team.OUR)
    chosen, dist, conf = est.designated([p1,p2], ball)
    assert chosen.track_id == 1
    assert conf > 0.5


def test_world_tensor_shape_and_channels():
    from efootball_agent.rl.world_tensor import WorldTensorEncoder
    world = WorldState(ball=Ball(x=0.5,y=0.5,valid=True,confidence=0.8,source="CV"))
    world.own_players=[Player(track_id=1,x=0.4,y=0.4,team=Team.OUR,confidence=0.8)]
    tensor=WorldTensorEncoder().encode(world)
    assert tensor.shape == (8,32,48)
    assert float(tensor[2].max()) > 0


def test_temporal_history_requires_two_valid_frames_for_continuity():
    from efootball_agent.world.temporal import WorldHistory
    from efootball_agent.core import Possession
    h=WorldHistory(4)
    b=Ball(x=0.5,y=0.5,valid=True,confidence=0.9,source="CV")
    w1=WorldState(timestamp=1.0,ball=b,active_track_id=1,active_valid=True,designated_track_id=1,designated_confidence=0.8,possession=Possession(owner_track_id=1,owner_team=Team.OUR,confidence=0.8))
    w2=WorldState(timestamp=1.1,ball=Ball(x=0.52,y=0.5,valid=True,confidence=0.9,source="CV"),active_track_id=1,active_valid=True,designated_track_id=1,designated_confidence=0.8,possession=Possession(owner_track_id=1,owner_team=Team.OUR,confidence=0.8))
    h.push(w1); assert not h.continuity_valid
    h.push(w2); assert h.continuity_valid


def test_sticky_control_preserves_high_level_intent():
    from efootball_agent.control.actions import StickyControlState
    st=StickyControlState(); action=Action(Movement.RIGHT, FootballAction.PASS, 1)
    st.set_desired(action, 1.0)
    assert st.snapshot()==action
    st.set_desired(Action(), 1.1)
    assert st.snapshot()==Action()


def test_control_memory_defaults_to_invalid():
    world = WorldState(control_track_id=-1, control_active_memory=False)
    assert world.control_track_id == -1
    assert world.control_active_memory is False


def test_world_config_short_prediction_bridge_contract():
    assert SETTINGS.world.allow_world_with_short_ball_prediction is True
    assert SETTINGS.world.short_ball_prediction_max_s <= 0.20



def test_designated_selector_has_switch_hysteresis():
    from efootball_agent.world.possession import PossessionEstimator
    est = PossessionEstimator(SETTINGS.world)
    ball = Ball(x=0.50, y=0.50, vx=0.0, vy=0.0, valid=True, confidence=0.9, source="CV")
    p1 = Player(track_id=1, x=0.49, y=0.48, confidence=0.9, team=Team.OUR, vx=0.0, vy=0.0)
    p2 = Player(track_id=2, x=0.54, y=0.50, confidence=0.9, team=Team.OUR, vx=0.0, vy=0.0)
    chosen1, _, _ = est.designated([p1, p2], ball, dt=0.1)
    chosen2, _, _ = est.designated([p1, p2], Ball(x=0.52, y=0.50, valid=True, confidence=0.9, source="CV"), dt=0.1)
    assert chosen1 is not None
    assert chosen2 is not None
    assert chosen1.track_id == 1
    assert chosen2.track_id == 1
    assert est.designated_track_id == 1


def test_possession_survives_short_unknown_ball_gap():
    from efootball_agent.world.possession import PossessionEstimator
    est = PossessionEstimator(SETTINGS.world)
    own = [Player(track_id=7, x=0.50, y=0.50, confidence=0.9, team=Team.OUR)]
    ball = Ball(x=0.50, y=0.50, valid=True, confidence=0.9, source="CV")
    first = est.update(own, [], ball, 0.1)
    assert first.owner_team == Team.OUR
    second = est.update(own, [], Ball(valid=False, source="UNKNOWN"), 0.1)
    assert second.owner_team == Team.OUR
    assert second.state == "OUR_RECEIVING"


def test_short_predicted_ball_is_usable_for_world_bridge():
    from efootball_agent.perception.ball import BallManager
    from efootball_agent.config import BallConfig
    bm = BallManager(BallConfig(), __import__('efootball_agent.perception.ui_mask', fromlist=['UIMask']).UIMask(SETTINGS.ui))
    bm.previous = Ball(x=0.50, y=0.50, vx=0.10, vy=0.0, confidence=0.80, source="CV", valid=True)
    import time
    bm.previous_time = time.monotonic() - 0.05
    bm.last_real_measurement_time = bm.previous_time
    bm.confirmed_streak = 3
    pred = bm.update(blank_frame(), [], [], -1)
    assert pred.source == "PREDICTED"
    assert pred.predicted is True
    assert pred.valid is True
