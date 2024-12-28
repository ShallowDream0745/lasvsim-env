"""
Simulator model module for the lasvsim API.
"""
from typing import Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class Point:
    """Point information."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


@dataclass
class ObjBaseInfo:
    """Object base information."""
    width: float = 0.0
    height: float = 0.0
    length: float = 0.0
    weight: float = 0.0


@dataclass
class DynamicInfo:
    """Dynamic information."""
    front_wheel_stiffness: float = 0.0
    rear_wheel_stiffness: float = 0.0
    front_axle_to_center: float = 0.0
    rear_axle_to_center: float = 0.0
    yaw_moment_of_inertia: float = 0.0


@dataclass
class Position:
    """Position information."""
    point: Optional[Point] = None
    phi: float = 0.0
    lane_id: str = ""
    link_id: str = ""
    junction_id: str = ""
    segment_id: str = ""
    dis_to_lane_end: Optional[float] = None
    position_type: int = 0

    def __init__(self, data: dict = None):
        if data is None:
            return
        point = data.pop("point", None)
        self.__dict__.update(data)
        self.point = None if point is None else Point(**point)


@dataclass
class ObjMovingInfo:
    """Object moving information."""
    u: float = 0.0
    u_acc: float = 0.0
    v: float = 0.0
    v_acc: float = 0.0
    w: float = 0.0
    w_acc: float = 0.0


@dataclass
class ControlInfo:
    """Control information."""
    ste_wheel: float = 0.0
    lon_acc: float = 0.0
    fl_torque: float = 0.0
    fr_torque: float = 0.0
    rl_torque: float = 0.0
    rr_torque: float = 0.0


@dataclass
class ReferenceLine:
    """Reference line information."""
    lane_ids: List[str] = field(default_factory=list)
    lane_types: List[str] = field(default_factory=list)
    points: List[Point] = field(default_factory=list)
    lane_idxes: List[int] = field(default_factory=list)
    opposite: bool = False

    def __init__(self, data: dict = None):
        if data is None:
            self.lane_ids = []
            self.lane_types = []
            self.points = []
            self.lane_idxes = []
            return
        self.lane_ids = data.get("lane_ids", [])
        self.lane_types = data.get("lane_types", [])
        self.points = [Point(**point) for point in data.get("points", [])]
        self.lane_idxes = data.get("lane_idxes", [])
        self.opposite = data.get("opposite", False)


@dataclass
class NavigationInfo:
    """Navigation information."""
    link_nav: List[str] = field(default_factory=list)
    destination: Optional[Position] = None

    def __init__(self, data: dict = None):
        if data is None:
            self.link_nav = []
            return
        destination = data.pop("destination", None)
        self.__dict__.update(data)
        self.destination = None if destination is None else Position(destination)


@dataclass
class Movement:
    """Movement information."""
    map_id: int = 0
    movement_id: str = ""
    upstream_link_id: str = ""
    downstream_link_id: str = ""
    junction_id: str = ""
    flow_direction: str = ""


@dataclass
class LaneNav:
    """Lane navigation information."""
    nav: Dict[int, str] = field(default_factory=dict)


@dataclass
class SimulatorConfig:
    """Simulator configuration."""
    scen_id: str = ""
    scen_ver: str = ""
    sim_record_id: str = ""


@dataclass
class InitReq:
    """Request for initializing simulator."""
    scen_id: str = ""
    scen_ver: str = ""
    sim_record_id: str = ""


@dataclass
class InitRes:
    """Response for initializing simulator."""
    simulation_id: str = ""
    simulation_addr: str = ""

    def __init__(self, data: dict = None):
        if data is None:
            return
        self.__dict__.update(data)


@dataclass
class StopReq:
    """Request for stopping simulator."""
    simulation_id: str = ""


@dataclass
class StopRes:
    """Response for stopping simulator."""
    pass


@dataclass
class StepReq:
    """Request for stepping simulator."""
    simulation_id: str = ""


@dataclass
class StepRes:
    """Response for stepping simulator."""
    pass


@dataclass
class GetCurrentStageReq:
    """Request for getting current stage."""
    simulation_id: str = ""  # 仿真ID
    junction_id: str = ""  # movementId


@dataclass
class GetCurrentStageRes:
    """Response for getting current stage."""
    movement_ids: List[str] = field(default_factory=list)  # 当前阶段包含的绿灯流向
    countdown: int = 0  # 倒计时(s)

    def __init__(self, data: dict = None):
        if data is None:
            self.movement_ids = []
            return
        self.__dict__.update(data)


@dataclass
class GetMovementSignalReq:
    """Request for getting movement signal."""
    simulation_id: str = ""  # 仿真ID
    movement_id: str = ""  # movementId


@dataclass
class GetMovementSignalRes:
    """Response for getting movement signal."""
    current_signal: int = 0  # 当前灯色
    countdown: int = 0  # 倒计时(s)

    def __init__(self, data: dict = None):
        if data is None:
            return
        self.__dict__.update(data)


@dataclass
class GetSignalPlanRes_Stage:
    """Stage information in signal plan."""
    movement_ids: List[str] = field(default_factory=list)
    duration: int = 0  # 时长(s)

    def __init__(self, data: dict = None):
        if data is None:
            self.movement_ids = []
            return
        self.__dict__.update(data)


@dataclass
class GetSignalPlanReq:
    """Request for getting signal plan."""
    simulation_id: str = ""
    junction_id: str = ""


@dataclass
class GetSignalPlanRes:
    """Response for getting signal plan."""
    junction_id: str = ""
    cycle: int = 0
    offset: int = 0
    stages: List[GetSignalPlanRes_Stage] = field(default_factory=list)

    def __init__(self, data: dict = None):
        if data is None:
            self.stages = []
            return
        stages = data.pop("stages", [])
        self.__dict__.update(data)
        self.stages = [GetSignalPlanRes_Stage(stage) for stage in stages]


@dataclass
class GetMovementListReq:
    """Request for getting movement list."""
    simulation_id: str = ""
    junction_id: str = ""


@dataclass
class GetMovementListRes:
    """Response for getting movement list."""
    list: List[Movement] = field(default_factory=list)  # movement 列表

    def __init__(self, data: dict = None):
        if data is None:
            self.list = []
            return
        self.list = [Movement(**item) for item in data.get("list", [])]


@dataclass
class GetVehicleIdListReq:
    """Request for getting vehicle ID list."""
    simulation_id: str = ""  # 仿真ID


@dataclass
class GetVehicleIdListRes:
    """Response for getting vehicle ID list."""
    list: List[str] = field(default_factory=list)  # 车辆ID列表

    def __init__(self, data: dict = None):
        if data is None:
            self.list = []
            return
        self.__dict__.update(data)


@dataclass
class GetTestVehicleIdListReq:
    """Request for getting test vehicle ID list."""
    simulation_id: str = ""  # 仿真ID


@dataclass
class GetTestVehicleIdListRes:
    """Response for getting test vehicle ID list."""
    list: List[str] = field(default_factory=list)

    def __init__(self, data: dict = None):
        if data is None:
            self.list = []
            return
        self.__dict__.update(data)


@dataclass
class GetVehicleBaseInfoReq:
    """Request for getting vehicle base information."""
    simulation_id: str = ""
    id_list: List[str] = field(default_factory=list)


@dataclass
class GetVehicleBaseInfoRes_VehicleBaseInfo:
    """Vehicle base information."""
    base_info: Optional[ObjBaseInfo] = None
    dynamic_info: Optional[DynamicInfo] = None

    def __init__(self, data: dict = None):
        if data is None:
            return
        base_info = data.pop("base_info", None)
        dynamic_info = data.pop("dynamic_info", None)
        self.base_info = None if base_info is None else ObjBaseInfo(**base_info)
        self.dynamic_info = None if dynamic_info is None else DynamicInfo(**dynamic_info)


@dataclass
class GetVehicleBaseInfoRes:
    """Response for getting vehicle base information."""
    info_dict: Dict[str, GetVehicleBaseInfoRes_VehicleBaseInfo] = field(default_factory=dict)

    def __init__(self, data: dict = None):
        if data is None:
            self.info_dict = {}
            return
        self.info_dict = {k: GetVehicleBaseInfoRes_VehicleBaseInfo(v) for k, v in data.get("info_dict", {}).items()}


@dataclass
class GetVehiclePositionReq:
    """Request for getting vehicle position."""
    simulation_id: str = ""
    id_list: List[str] = field(default_factory=list)


@dataclass
class GetVehiclePositionRes:
    """Response for getting vehicle position."""
    position_dict: Dict[str, Position] = field(default_factory=dict)

    def __init__(self, data: dict = None):
        if data is None:
            self.position_dict = {}
            return
        self.position_dict = {k: Position(v) for k, v in data.get("position_dict", {}).items()}


@dataclass
class GetVehicleMovingInfoReq:
    """Request for getting vehicle moving information."""
    simulation_id: str = ""
    id_list: List[str] = field(default_factory=list)


@dataclass
class GetVehicleMovingInfoRes:
    """Response for getting vehicle moving information."""
    moving_info_dict: Dict[str, ObjMovingInfo] = field(default_factory=dict)

    def __init__(self, data: dict = None):
        if data is None:
            self.moving_info_dict = {}
            return
        self.moving_info_dict = {k: ObjMovingInfo(**v) for k, v in data.get("moving_info_dict", {}).items()}


@dataclass
class GetVehicleControlInfoReq:
    """Request for getting vehicle control information."""
    simulation_id: str = ""
    id_list: List[str] = field(default_factory=list)


@dataclass
class GetVehicleControlInfoRes:
    """Response for getting vehicle control information."""
    control_info_dict: Dict[str, ControlInfo] = field(default_factory=dict)

    def __init__(self, data: dict = None):
        if data is None:
            self.control_info_dict = {}
            return
        self.control_info_dict = {k: ControlInfo(**v) for k, v in data.get("control_info_dict", {}).items()}


@dataclass
class GetVehiclePerceptionInfoReq:
    """Request for getting vehicle perception information."""
    simulation_id: str = ""
    vehicle_id: str = ""


@dataclass
class GetVehiclePerceptionInfoRes_PerceptionObj:
    """Perception object information."""
    obj_id: str = ""
    base_info: Optional[ObjBaseInfo] = None
    moving_info: Optional[ObjMovingInfo] = None
    position: Optional[Position] = None

    def __init__(self, data: dict = None):
        if data is None:
            return
        base_info = data.pop("base_info", None)
        moving_info = data.pop("moving_info", None)
        position = data.pop("position", None)
        self.__dict__.update(data)
        self.base_info = None if base_info is None else ObjBaseInfo(**base_info)
        self.moving_info = None if moving_info is None else ObjMovingInfo(**moving_info)
        self.position = None if position is None else Position(position)


@dataclass
class GetVehiclePerceptionInfoRes:
    """Response for getting vehicle perception information."""
    list: List[GetVehiclePerceptionInfoRes_PerceptionObj] = field(default_factory=list)

    def __init__(self, data: dict = None):
        if data is None:
            self.list = []
            return
        self.list = [GetVehiclePerceptionInfoRes_PerceptionObj(item) for item in data.get("list", [])]


@dataclass
class GetVehicleReferenceLinesReq:
    """Request for getting vehicle reference lines."""
    simulation_id: str = ""
    vehicle_id: str = ""


@dataclass
class GetVehicleReferenceLinesRes:
    """Response for getting vehicle reference lines."""
    reference_lines: List[ReferenceLine] = field(default_factory=list)

    def __init__(self, data: dict = None):
        if data is None:
            self.reference_lines = []
            return
        self.reference_lines = [ReferenceLine(item) for item in data.get("reference_lines", [])]


@dataclass
class GetVehiclePlanningInfoReq:
    """Request for getting vehicle planning information."""
    simulation_id: str = ""
    vehicle_id: str = ""


@dataclass
class GetVehiclePlanningInfoRes:
    """Response for getting vehicle planning information."""
    planning_path: List[Point] = field(default_factory=list)

    def __init__(self, data: dict = None):
        if data is None:
            self.planning_path = []
            return
        self.planning_path = [Point(**point) for point in data.get("planning_path", [])]


@dataclass
class GetVehicleNavigationInfoReq:
    """Request for getting vehicle navigation information."""
    simulation_id: str = ""
    vehicle_id: str = ""


@dataclass
class GetVehicleNavigationInfoRes:
    """Response for getting vehicle navigation information."""
    navigation_info: Optional[NavigationInfo] = None

    def __init__(self, data: dict = None):
        if data is None:
            return
        navigation_info = data.pop("navigation_info", None)
        self.navigation_info = None if navigation_info is None else NavigationInfo(navigation_info)


@dataclass
class GetVehicleCollisionStatusReq:
    """Request for getting vehicle collision status."""
    simulation_id: str = ""
    vehicle_id: str = ""


@dataclass
class GetVehicleCollisionStatusRes:
    """Response for getting vehicle collision status."""
    collision_status: bool = False

    def __init__(self, data: dict = None):
        if data is None:
            return
        self.__dict__.update(data)


@dataclass
class GetVehicleTargetSpeedReq:
    """Request for getting vehicle target speed."""
    simulation_id: str = ""
    vehicle_id: str = ""


@dataclass
class GetVehicleTargetSpeedRes:
    """Response for getting vehicle target speed."""
    target_speed: float = 0.0

    def __init__(self, data: dict = None):
        if data is None:
            return
        self.__dict__.update(data)


@dataclass
class SetVehiclePlanningInfoReq:
    """Request for setting vehicle planning information."""
    simulation_id: str = ""
    vehicle_id: str = ""
    planning_path: List[Point] = field(default_factory=list)


@dataclass
class SetVehiclePlanningInfoRes:
    """Response for setting vehicle planning information."""
    pass


@dataclass
class SetVehicleControlInfoReq:
    """Request for setting vehicle control information."""
    simulation_id: str = ""
    vehicle_id: str = ""
    ste_wheel: Optional[float] = None
    lon_acc: Optional[float] = None


@dataclass
class SetVehicleControlInfoRes:
    """Response for setting vehicle control information."""
    pass


@dataclass
class SetVehiclePositionReq:
    """Request for setting vehicle position."""
    simulation_id: str = ""
    vehicle_id: str = ""
    point: Optional[Point] = None
    phi: Optional[float] = None


@dataclass
class SetVehiclePositionRes:
    """Response for setting vehicle position."""
    pass


@dataclass
class SetVehicleMovingInfoReq:
    """Request for setting vehicle moving information."""
    simulation_id: str = ""
    vehicle_id: str = ""
    u: Optional[float] = None
    v: Optional[float] = None
    w: Optional[float] = None
    u_acc: Optional[float] = None
    v_acc: Optional[float] = None
    w_acc: Optional[float] = None


@dataclass
class SetVehicleMovingInfoRes:
    """Response for setting vehicle moving information."""
    pass


@dataclass
class SetVehicleBaseInfoReq:
    """Request for setting vehicle base information."""
    simulation_id: str = ""
    vehicle_id: str = ""
    base_info: Optional[ObjBaseInfo] = None
    dynamic_info: Optional[DynamicInfo] = None


@dataclass
class SetVehicleBaseInfoRes:
    """Response for setting vehicle base information."""
    pass


@dataclass
class SetVehicleRouteNavReq:
    """Request for setting vehicle route navigation."""
    simulation_id: str = ""
    vehicle_id: str = ""
    route_nav: List[str] = field(default_factory=list)


@dataclass
class SetVehicleRouteNavRes:
    """Response for setting vehicle route navigation."""
    pass


@dataclass
class SetVehicleLinkNavReq:
    """Request for setting vehicle link navigation."""
    simulation_id: str = ""
    vehicle_id: str = ""
    link_nav: List[str] = field(default_factory=list)


@dataclass
class SetVehicleLinkNavRes:
    """Response for setting vehicle link navigation."""
    pass


@dataclass
class SetVehicleLaneNavReq:
    """Request for setting vehicle lane navigation."""
    simulation_id: str = ""
    vehicle_id: str = ""
    lane_nav: List[LaneNav] = field(default_factory=list)


@dataclass
class SetVehicleLaneNavRes:
    """Response for setting vehicle lane navigation."""
    pass


@dataclass
class SetVehicleDestinationReq:
    """Request for setting vehicle destination."""
    simulation_id: str = ""
    vehicle_id: str = ""
    destination: Optional[Point] = None


@dataclass
class SetVehicleDestinationRes:
    """Response for setting vehicle destination."""
    pass


@dataclass
class GetPedIdListReq:
    """Request for getting pedestrian ID list."""
    simulation_id: str = ""


@dataclass
class GetPedIdListRes:
    """Response for getting pedestrian ID list."""
    list: List[str] = field(default_factory=list)

    def __init__(self, data: dict = None):
        if data is None:
            self.list = []
            return
        self.__dict__.update(data)


@dataclass
class GetPedBaseInfoReq:
    """Request for getting pedestrian base information."""
    simulation_id: str = ""
    ped_id_list: List[str] = field(default_factory=list)


@dataclass
class GetPedBaseInfoRes:
    """Response for getting pedestrian base information."""
    base_info_dict: Dict[str, ObjBaseInfo] = field(default_factory=dict)

    def __init__(self, data: dict = None):
        if data is None:
            self.base_info_dict = {}
            return
        self.base_info_dict = {k: ObjBaseInfo(**v) for k, v in data.get("base_info_dict", {}).items()}


@dataclass
class SetPedPositionReq:
    """Request for setting pedestrian position."""
    simulation_id: str = ""
    ped_id: str = ""
    point: Optional[Point] = None
    phi: Optional[float] = None


@dataclass
class SetPedPositionRes:
    """Response for setting pedestrian position."""
    pass


@dataclass
class GetNMVIdListReq:
    """Request for getting non-motor vehicle ID list."""
    simulation_id: str = ""


@dataclass
class GetNMVIdListRes:
    """Response for getting non-motor vehicle ID list."""
    list: List[str] = field(default_factory=list)

    def __init__(self, data: dict = None):
        if data is None:
            self.list = []
            return
        self.__dict__.update(data)


@dataclass
class GetNMVBaseInfoReq:
    """Request for getting non-motor vehicle base information."""
    simulation_id: str = ""
    nmv_id_list: List[str] = field(default_factory=list)


@dataclass
class GetNMVBaseInfoRes:
    """Response for getting non-motor vehicle base information."""
    base_info_dict: Dict[str, ObjBaseInfo] = field(default_factory=dict)

    def __init__(self, data: dict = None):
        if data is None:
            self.base_info_dict = {}
            return
        self.base_info_dict = {k: ObjBaseInfo(**v) for k, v in data.get("base_info_dict", {}).items()}


@dataclass
class SetNMVPositionReq:
    """Request for setting non-motor vehicle position."""
    simulation_id: str = ""
    nmv_id: str = ""
    point: Optional[Point] = None
    phi: Optional[float] = None


@dataclass
class SetNMVPositionRes:
    """Response for setting non-motor vehicle position."""
    pass
