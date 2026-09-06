from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class CaptureConfig:
    window_title: str = "eFootball"
    window_only: bool = True
    fps: int = 60
    width: int = 1280
    height: int = 720


@dataclass(frozen=True)
class UIConfig:
    # Normalized client-frame coordinates.
    scoreboard: tuple[float, float, float, float] = (0.035, 0.025, 0.32, 0.105)
    minimap: tuple[float, float, float, float] = (0.42, 0.78, 0.58, 0.96)
    bottom_left_card: tuple[float, float, float, float] = (0.03, 0.875, 0.235, 0.995)
    bottom_right_card: tuple[float, float, float, float] = (0.74, 0.875, 0.995, 0.995)
    top_left_icon: tuple[float, float, float, float] = (0.0, 0.0, 0.09, 0.12)


@dataclass(frozen=True)
class PlayerConfig:
    person_label: int = 0
    min_model_conf: float = 0.18
    # RT-DETR runs asynchronously; the main perception loop never waits for it.
    model_hz: float = 2.5
    # Detector measurements may be consumed for perception until this age,
    # but control has a stricter bound below.
    max_model_stale_s: float = 0.65
    # RT-DETR is a periodic re-detector on CPU. Its last result may continue
    # correcting the tracker for a longer bounded window, while control uses
    # only fast local perception and does not require a fresh detector frame.
    model_reuse_stale_s: float = 1.80
    control_model_stale_s: float = 0.45
    control_active_grace_s: float = 0.85
    control_active_min_conf: float = 0.28
    inference_device: str = "cpu"   # V24.1 AMD/Windows contract: CPU only; no CUDA dependency.
    cpu_num_threads: int = 8
    cpu_num_interop_threads: int = 1
    cpu_affinity_cores: str = "auto"
    model_input_size: int = 640
    max_players: int = 22
    cv_recovery_max: int = 10
    max_tracks: int = 22
    cv_recovery_every_n_frames: int = 2

    field_y_max: float = 0.93
    footpoint_green_min: float = 0.28
    min_bbox_h: float = 0.025
    max_bbox_h: float = 0.55
    min_bbox_w: float = 0.006
    max_bbox_w: float = 0.12
    max_player_aspect: float = 0.72

    black_v_max: int = 120
    black_dark_fraction: float = 0.40
    white_v_min: int = 150
    white_sat_max: int = 105
    white_bright_fraction: float = 0.32
    team_margin: float = 0.15

    enable_cv_recovery: bool = True
    require_model_for_control: bool = False


@dataclass(frozen=True)
class ActiveConfig:
    hue_low: int = 72
    hue_high: int = 112
    sat_min: int = 75
    val_min: int = 105
    min_pixels: int = 30
    max_pixels: int = 500
    max_missing: int = 15
    x_tol: float = 0.08
    y_above_min: float = -0.04
    y_above_max: float = 0.16
    min_association_conf: float = 0.45
    switch_margin: float = 0.12
    active_track_max_missed: int = 8
    previous_bonus: float = 0.12
    recovery_x_radius: float = 0.065
    recovery_y_min: float = -0.06
    recovery_y_max: float = 0.12
    recovery_min_pixels: int = 10
    recovery_min_ratio: float = 0.0010
    recovery_target_ratio: float = 0.012


@dataclass(frozen=True)
class BallConfig:
    ball_label: int = 32
    min_model_conf: float = 0.18
    accept_conf: float = 0.45
    control_conf: float = 0.60
    model_min_field_green: float = 0.42
    min_field_green: float = 0.72
    cv_min_area: int = 2
    cv_max_area: int = 120
    cv_min_width: int = 2
    cv_max_width: int = 16
    cv_min_height: int = 2
    cv_max_height: int = 12
    cv_aspect_min: float = 0.35
    cv_aspect_max: float = 2.8
    cv_y_min: float = 0.10
    cv_y_max: float = 0.86
    cv_min_brightness: float = 0.22
    cv_white_v_min: int = 175
    cv_white_s_max: int = 100
    cv_h_low: int = 8
    cv_h_high: int = 40
    cv_s_min: int = 110
    cv_v_min: int = 135
    cv_orange_shape_penalty: float = 0.65
    cv_accept_conf: float = 0.62
    cv_first_frame_conf: float = 0.72
    cv_first_frame_min_area: float = 2.0
    cv_first_frame_min_shape: float = 0.50
    cv_first_frame_min_contrast: float = 25.0
    cv_first_frame_global_min_area: float = 10.0
    cv_first_frame_global_max_area: float = 70.0
    cv_first_frame_global_min_shape: float = 0.55
    cv_first_frame_global_min_contrast: float = 45.0
    cv_orange_confidence_cap: float = 0.72
    cv_non_temporal_penalty: float = 0.82
    cv_local_accept_conf: float = 0.62
    cv_local_max_dist: float = 0.050
    global_cv_every_n_frames: int = 3
    detector_jump_penalty: float = 0.50
    detector_cv_match_distance: float = 0.04
    detector_cv_bonus: float = 0.40
    detector_cv_max_conf: float = 0.90
    detector_min_width: float = 0.002
    detector_max_width: float = 0.035
    detector_min_height: float = 0.002
    detector_max_height: float = 0.030
    detector_aspect_min: float = 0.35
    detector_aspect_max: float = 2.8
    prediction_decay: float = 0.78
    max_jump: float = 0.16
    max_missed: int = 6
    max_prediction_age_s: float = 0.30
    prediction_max_missed: int = 3
    local_search_radius: float = 0.095
    local_min_conf: float = 0.58
    max_ball_hold_s: float = 0.24
    alpha_beta_alpha: float = 0.72
    alpha_beta_beta: float = 0.18

    # Trajectory-based goal fallback. Score OCR remains the preferred signal.
    goal_cross_x_for: float = 0.995
    goal_cross_x_against: float = 0.005
    goal_y_min: float = 0.34
    goal_y_max: float = 0.66
    goal_min_vx: float = 0.20
    goal_candidate_timeout_frames: int = 8


@dataclass(frozen=True)
class TrackingConfig:
    max_missed: int = 20
    tentative_hits: int = 2
    match_radius: float = 0.10
    appearance_weight: float = 0.12
    velocity_weight: float = 0.18
    team_mismatch_cost: float = 0.05
    flow_min_conf: float = 0.30


@dataclass(frozen=True)
class ScoreConfig:
    enabled: bool = True
    # Calibrated from the supplied 1280x720 live eFootball HUD:
    # left score cell ~= x[148:178], y[48:80]; right ~= x[182:212], y[48:80].
    left_score_roi: tuple[float, float, float, float] = (148/1280, 48/720, 178/1280, 80/720)
    right_score_roi: tuple[float, float, float, float] = (182/1280, 48/720, 212/1280, 80/720)
    clock_roi: tuple[float, float, float, float] = (296/1280, 48/720, 376/1280, 80/720)
    controlled_side: str = "left"
    interval_s: float = 0.35
    confirm_frames: int = 2


@dataclass(frozen=True)
class WorldConfig:
    # Canonical attacking coordinate system: OUR goal at x=0, opponent goal at x=1.
    attack_direction: str = "right"
    possession_scale: float = 0.07
    possession_on: float = 0.66
    possession_off: float = 0.42
    prediction_horizons: tuple[float, float, float] = (0.25, 0.50, 1.00)
    temporal_history: int = 4
    world_temporal_min_depth: int = 2
    world_temporal_ball_max_jump: float = 0.18
    designated_min_confidence: float = 0.45
    designated_previous_bonus: float = 0.12
    designated_team_bonus: float = 0.04
    designated_switch_margin: float = 0.10
    designated_decay: float = 0.92
    designated_hold_s: float = 0.55
    possession_switch_margin: float = 0.10
    possession_unknown_grace_s: float = 0.30
    possession_unknown_decay: float = 0.92
    allow_world_without_possession_owner: bool = True
    allow_world_with_short_ball_prediction: bool = True
    short_ball_prediction_max_s: float = 0.20
    short_ball_prediction_min_conf: float = 0.48


@dataclass(frozen=True)
class TacticalConfig:
    shot_x: float = 0.72
    attack_x: float = 0.50
    cross_x: float = 0.80
    pass_space_min: float = 0.25
    pressure_high: float = 0.70
    counter_threshold: float = 0.35


@dataclass(frozen=True)
class PPOConfig:
    # GRF-inspired semantic state + structured MultiCategorical action.
    state_dim: int = 170
    movement_actions: int = 9
    football_actions: int = 6   # NONE/PASS/CROSS/SHOOT/PRESS/SWITCH
    sprint_actions: int = 2

    hidden: int = 384
    lr: float = 2.5e-4
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip: float = 0.20
    value_coef: float = 0.50
    entropy_coef: float = 0.006
    max_grad_norm: float = 0.50

    decision_hz: float = 10.0
    rollout_steps: int = 256
    minibatch: int = 64
    epochs: int = 4
    target_kl: float = 0.03

    # Planner bootstrap / behavior cloning before PPO.
    teacher_steps: int = 3000
    teacher_batch: int = 64

    deterministic: bool = False
    checkpoint: Path = Path("checkpoints/v24_ppo.pt")
    resume: bool = True
    seed: int = 42


@dataclass(frozen=True)
class RewardConfig:
    goal_for: float = 10.0
    goal_against: float = -10.0
    shot_on_target: float = 1.0
    ball_won: float = 0.50
    ball_lost: float = -0.60
    possession: float = 0.01
    x_progress: float = 1.50
    x_retreat: float = 0.35
    shaping_clip: float = 0.50
    pressing_spam: float = -0.012
    pressing_effort: float = 0.008


@dataclass(frozen=True)
class ControlConfig:
    hz: float = 120.0
    require_focus: bool = True
    dry_run: bool = False
    y_invert: bool = True

    pass_tap_s: float = 0.075
    cross_tap_s: float = 0.075
    shoot_tap_s: float = 0.085
    switch_tap_s: float = 0.055

    pass_cooldown_s: float = 0.14
    cross_cooldown_s: float = 0.16
    shoot_cooldown_s: float = 0.22
    switch_cooldown_s: float = 0.25
    stick_smoothing: float = 0.28
    sticky_action_release_s: float = 0.08


@dataclass(frozen=True)
class RuntimeConfig:
    perception_hz: float = 30.0
    log_interval_s: float = 2.0


@dataclass(frozen=True)
class Settings:
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    player: PlayerConfig = field(default_factory=PlayerConfig)
    active: ActiveConfig = field(default_factory=ActiveConfig)
    ball: BallConfig = field(default_factory=BallConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    score: ScoreConfig = field(default_factory=ScoreConfig)
    world: WorldConfig = field(default_factory=WorldConfig)
    tactical: TacticalConfig = field(default_factory=TacticalConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    rtdetr_dir: Path = Path("models/rtdetr_r50vd")


SETTINGS = Settings()
