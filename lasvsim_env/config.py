from dataclasses import dataclass, field
from pathlib import Path
import textwrap
from typing import Any, Dict, Optional, Tuple, Union, List

from omegaconf import OmegaConf

FAKE_PATH = Path('/')
Color = Any  # Tuple[int, int, int, int]
PolyOption = Tuple[Color, int, float, bool]  # color, layer, width, filled
ObsNumSur = Any  # Union[int, Dict[str, int]]

@dataclass(frozen=False)
class Config:
    """Configuration for the Lasvsim environment."""

    seed: Optional[int] = None
    debug: bool = False
    dense_ref_mode: str = "bezier"
    # dense_ref_param set to None
    dense_ref_param: Optional[Any] = None

    # ===== Env =====
    dt: float = 0.1  # Do not change this value.
    actuator: str = "ExternalActuator"
    max_steps: int = 1000
    use_pose_reward: bool = False
    penalize_collision: bool = True
    no_done_at_collision: bool = False
    takeover_bias: bool = False
    takeover_bias_prob: float = 0.0
    random_ref_v: bool = False
    ref_v_range: Tuple[float, float] = (2,12)
    nonimal_acc: bool = False
    # [mean, std]
    takeover_bias_x: Tuple[float, float] = (0.0, 0.1)
    takeover_bias_y: Tuple[float, float] = (0.0, 0.1)
    takeover_bias_phi: Tuple[float, float] = (0.0, 0.05)
    takeover_bias_vx: Tuple[float, float] = (0.6, 0.2)
    takeover_bias_ax: Tuple[float, float] = (0.0, 0.1)
    takeover_bias_steer: Tuple[float, float] = (0.0, 0.01)
    add_sur_bias: bool = False
    sur_bias_range: Tuple[float, float] = (0.0, 0.0) # (a_min, a_max)
    sur_bias_prob: float = 0.0
    # model free reward config
    punish_sur_mode: str = "sum"
    enable_slow_reward: bool = False
    R_step: float = 5.0
    P_lat: float = 5.0
    P_long: float = 2.5
    P_phi: float = 20.0
    P_yaw: float = 10.0
    P_vel: float = 3.0
    P_front: float = 5.0
    P_side: float = 5.0
    P_space: float = 5.0
    P_rear: float = 5.0
    P_steer: float = 50.0
    P_acc: float = 0.2
    P_delta_steer: float = 50.0
    P_jerk: float = 0.1
    P_boundary: float = 0.0
    P_done: float = 2000.0


    safety_lat_margin_front: float = 0.0
    safety_long_margin_front: float = 0.0
    safety_long_margin_side: float = 0.0
    front_dist_thd: float = 50.0
    space_dist_thd: float = 8.0
    rel_v_thd: float = 1.0
    rel_v_rear_thd: float = 0.0
    time_dist: float = 2.0
    # Since libsumo/traci only allows a single running instance,
    # we need to use a singleton mode to avoid multiple instances.
    # The following modes are available:
    #   "raise": raise an error if multiple instances are created (default).
    #   "reuse": reuse the previous instance.
    #   "invalidate": invalidate the previous instance.
    singleton_mode: str = "raise"

    # ===== Ego veh dynamics =====
    ego_id: Optional[str] = None
    action_repeat: int = 1
    incremental_action: bool = False
    action_lower_bound: Tuple[float, float] = (-2.0, -0.35)
    action_upper_bound: Tuple[float, float] = ( 1.0,  0.35)
    real_action_lower_bound: Tuple[float, float] = (-3.0, -0.065)
    real_action_upper_bound: Tuple[float, float] = ( 0.8,  0.065)
    vehicle_spec:Tuple[float, float, float, float, float, float, float, float] = \
        (1412.0, 1536.7, 1.06, 1.85, -128915.5, -85943.6, 20.0, 0.0)
    # m: float  # mass
    # Iz: float  # moment of inertia
    # lf: float  # distance from front axle to center of gravity
    # lr: float  # distance from rear axle to center of gravity
    # Cf: float  # cornering stiffness of front tires (negative)
    # Cr: float  # cornering stiffness of rear tires (negative)
    # vx_max: float  # maximum longitudinal velocity
    # vx_min: float  # minimum longitudinal velocity

    # ===== Observation =====
    obs_components: Tuple[str, ...] = (
        "EgoState", "Waypoint", "TrafficLight", "DrivingArea", "SurState"
    )
    obs_flatten: bool = True
    obs_normalize: bool = False
    # For Waypoint
    obs_num_ref_points: int = 5
    obs_ref_interval: float = 5.0
    obs_ref_candidate_set: bool = False
    random_ref_cooldown: int = 30
    choose_closest: bool = False
    mid_line_obs: bool = False
    # For DrivingArea
    obs_driving_area_lidar_num_rays: int = 16
    obs_driving_area_lidar_range: float = 20.0
    # For SurState
    obs_num_surrounding_vehicles: ObsNumSur = 0
    # For SurLidar
    obs_surrounding_lidar_num_rays: int = 16
    # For SurEncoding
    obs_surrounding_encoding_model_path: Path = FAKE_PATH

    # ===== Scenario =====
    # NOTE: consider using idscene config?
    scenario_root: Path = FAKE_PATH
    scenario_name_template: str = "{id}"
    num_scenarios: int = 10
    multilane_scenarios: Tuple[int, ...] = tuple(range(0, 22))
    scenario_selector: Optional[str] = None  # should be in `:a,b,c:d,e:` format
    scenario_filter_surrounding_selector: Optional[str] = None  # should be a list
    scenario_filter_surrounding_range: Tuple[int, ...] = ()


    # ===== Traffic =====
    step_length: float = dt
    extra_sumo_args: Tuple[str, ...] = ()  # e.g. ("--start", "--quit-on-end", "--delay", "100")
    warmup_time: float = 5.0  # Select ego vehicle after `warmup_time` seconds for fresh (un-reused) reset
    scenario_reuse: int = 1
    detect_range: float = 5.0
    ignore_traffic_lights: bool = False
    persist_traffic_lights: bool = False
    native_collision_check: bool = False
    skip_waiting: bool = False
    skip_waiting_after_episode_steps: int = 20
    v_class_selector: str = "passenger"
    grab_vehicle_in_junction: bool = False # if true, vehicle in junction will be preferentially selected as ego vehicle in a probablity of 0.5
    direction_selector: Optional[str] = None
    choose_vehicle_retries: int = 3
    choose_vehicle_step_time: float = 5.0
    ignore_surrounding: bool = False
    ignore_opposite_direction: bool = False
    minimum_clearance_when_takeover: float = -1.0
    keep_route_mode: int = 0b011  # See https://sumo.dlr.de/docs/TraCI/Change_Vehicle_State.html#move_to_xy_0xb4
    traffic_seed: Optional[int] = None
    assigned_traffic_seed: Optional[int] = None

    # ===== Navigation =====
    ref_length: float = 20.0  # May be less than expected if cut
    ref_v: float = 8.0
    reference_selector: int = 0  # NOTE: will add more options in the future
    use_left_turn_waiting_area: bool = False

    use_multiple_path_for_multilane: bool = False
    random_ref_probability: float = 0.0
    random_ref_cooldown: int = 30

    use_random_acc: bool = False # if false, ref_v will be calculated by default.
    random_acc_cooldown: Tuple[int] = (0, 50, 50) # cooldown for acceleration, deceleration and ref_v, respectively
    random_acc_prob: Tuple[float] = (0.0, 0.5) # probability to accelerate and decelerate, respectively
    random_acc_range: Tuple[float] = (0.0, 0.0) # (m/s^2), used for acceleration (now useless)
    random_dec_range: Tuple[float] = (-3.0, -1.0) # (m/s^2), used for deceleration

    # ===== Trajectory =====
    use_trajectory: bool = False
    trajectory_deque_capacity: int = 20

    # ===== Rendering =====
    use_render: bool = False  # False - sumo + libsumo; True - sumo-gui + libtraci
    gui_setting_file: Optional[Path] = FAKE_PATH
    ego_color: Color = (255, 0, 255, 255)
    ref_poly_option: PolyOption = ((0, 145, 247, 255), 15, 0.2, False)
    show_detect_range: bool = True
    detection_poly_option: PolyOption = ((255, 255, 255, 255), 10, 0.2, False)
    detection_detail: int = 8
    show_driving_area: bool = True
    driving_area_poly_option: PolyOption = ((56, 255, 65, 255), 15, 0.4, False)
    vehicle_info_position: Tuple[float, float] = (-50.0, 50.0)  # Left-bottom corner of the vehicle info
    vehicle_info_template: str = textwrap.dedent("""
    step: {context.episode_step}
    time: {context.simulation_time:.2f}
    minor: {minor}

    x: {vehicle.state[0]:.2f}
    y: {vehicle.state[1]:.2f}
    vx: {vehicle.state[2]:.2f}
    vy: {vehicle.state[3]:.2f}
    phi: {vehicle.state[4]:.2f}
    omega: {vehicle.state[5]:.2f}

    ax: {vehicle.action[0]:5.2f}
    steer: {vehicle.action[1]:5.2f}

    route: {vehicle.route}
    edge: {vehicle.edge}
    lane: {vehicle.lane}

    on_lane: {vehicle.on_lane}
    waiting_well_positioned: {vehicle.waiting_well_positioned}
    waiting_ill_positioned: {vehicle.waiting_ill_positioned}
    speed_limit: {vehicle.speed_limit:5.2f}

    traffic_light: {vehicle.traffic_light}
    ahead_lane_length: {vehicle.ahead_lane_length:5.2f}
    remain_phase_time: {vehicle.remain_phase_time:5.2f}
    """)
    use_screenpeek: bool = False
    video_root: Path = Path("video")
    video_name_template: str = "{context.created_at:%Y-%m-%d_%H-%M-%S}_{context.id}/{context.scenario_count:03d}/{context.episode_count:04d}.mp4"
    video_width: int = 960
    video_height: int = 540
    video_zoom: float = 300.0
    video_output: bool = True  # Only when use_screenpeek is True, requires ffmpeg
    video_cleanup: bool = True

    # ===== Logging =====
    use_logging: bool = False
    logging_root: Path = Path("logs")
    logging_name_template: str = "{context.created_at:%Y-%m-%d_%H-%M-%S}_{context.id}/{context.scenario_count:03d}/{context.episode_count:04d}.pkl"
    logging_context: bool = False
    output_fcd: bool = True  # Also requires use_logging=True
    fcd_name_template: str = "{context.created_at:%Y-%m-%d_%H-%M-%S}_{context.id}/{context.scenario_count:03d}/fcd.xml"


    N: int = 30
    num_ref_points: int = 31
    full_horizon_sur_obs: bool = False
    sur_obs_padding: str = "zero" # "zero or "rule"
    ego_feat_dim: int = 7 # vx, vy, r, last_last_acc, last_last_steer, last_acc, last_steer
    add_boundary_obs: bool = False
    ego_bound_dim: int = 2 # left, right
    per_sur_state_dim: int = 6 # x, y, phi, speed, length, width
    per_sur_state_withinfo_dim: int = 7 # x, y, phi, speed, length, width, mask
    per_sur_feat_dim: int = 5 # x, y, cos(phi), sin(phi), speed
    per_ref_feat_dim: int = 5 # x, y, cos(phi), sin(phi), speed
    ref_v_lane: float = 8.0 # default for urban
    v_discount_in_junction_straight: float = 0.75
    v_discount_in_junction_left_turn: float = 0.5
    v_discount_in_junction_right_turn: float = 0.5
    num_ref_lines: int = 3
    downsample_ref_point_index: Tuple[int] = tuple([i for i in range(31)])
    filter_num: int = 0  # only for extra filter
    ahead_lane_length_min: float = 6.0
    ahead_lane_length_max: float = 60.0
    dec_before_junction: float = 0.8
    dec_before_junction_green: float = 0.8
    dec_before_junction_red: float = 1.3
    ego_length: float = 5.0
    ego_width: float = 1.8
    padding_veh_shape: Tuple[float, float] = (5.0, 1.8)
    padding_bike_shape: Tuple[float, float] = (0.0, 0.0)  # TODO: need to be modified
    padding_ped_shape: Tuple[float, float] = (0.0, 0.0)
    safe_dist_incremental: float = 1.2 # same with IDC problem
    min_dist_sur_length: float = 1.8
    min_dist_sur_width: float = 1.8

    real_action_upper: Tuple[float] = (0.8, 0.065) # [acc, steer]
    real_action_lower: Tuple[float] = (-3.0, -0.065) # [acc, steer]

    steer_rate_2_min: float = -0.2
    steer_rate_2_max: float = 0.2

    vx_min: float = 0.0
    vx_max: float = 20.0 # (m/s)
    vy_min: float = -4.0
    vy_max: float = 4.0

    max_dist_from_ref: float = 1.8 # (self added)

    Q: Tuple[float] = (0.4, 0.4, 500., 1., 1., 300.0)
    R: Tuple[float] = (1.0, 20.0,)

    track_closest_ref_point: bool = False
    use_nominal_action: bool = False
    ref_v_slow_focus: float = 0. # focus more on low speed tracking when ref_v < ref_v_slow_focus
    Q_slow_incre: Tuple[float] = (0., 0., 0., 0., 0., 0.) # when ref_v < ref_v_slow_focus, increment Q
    R_slow_incre: Tuple[float] = (0., 0.) # when ref_v < ref_v_slow_focus, increment R
    clear_nonessential_cost_safe: bool = False # clear the safe cost of some nonessential obstacles

    C_acc_rate_1: float = 0.0
    C_steer_rate_1: float = 0.0
    C_steer_rate_2: Tuple[float] = (100., 100.) # C_steer_rate_2_min, C_steer_rate_2_max
    C_v: Tuple[float] = (100., 100., 100., 100.) # C_vx_min, C_vx_max, C_vy_min, C_vy_max

    gamma: float = 0.99 # should be equal to discount factor of algorithm
    lambda_c: float = 0.99  # discount of lat penalty
    lambda_p: float = 0.99  # discount of lon penalty
    C_lat: float = 3.0
    C_obs: float = 300.0
    C_back: Tuple[float] = (0.1, 1.0)
    C_road: float = 300.0

    reward_scale: float = 0.01
    reward_comps: Tuple[str] = ()
    
    critic_dict: Dict[str, Any] = field(default_factory=lambda: {
        "sur_reward": [
            "env_scaled_reward_done",
            "env_scaled_reward_collision",
            "env_reward_collision_risk",
            "env_scaled_reward_boundary"
        ],
        "ego_reward": [
            "env_scaled_reward_step",
            "env_scaled_reward_vel_long",
            "env_scaled_reward_steering",
            "env_scaled_reward_acc_long",
            "env_scaled_reward_delta_steer",
            "env_scaled_reward_jerk",
            "env_scaled_reward_nominal_acc",
            "env_scaled_reward_overspeed",
        ],
        "tracking_reward": [
            "env_scaled_reward_dist_lat",
            "env_scaled_reward_head_ang",
            "env_scaled_reward_yaw_rate"
        ]
    })

    @staticmethod
    def from_partial_dict(partial_dict) -> "Config":
        base = OmegaConf.structured(Config)
        merged = OmegaConf.merge(base, partial_dict)
        return OmegaConf.to_object(merged)  # type: ignore
