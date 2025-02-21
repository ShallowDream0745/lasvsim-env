import torch
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Generic, Optional, Tuple, Union, ClassVar
from gops.env.env_gen_ocp.pyth_base import (Context, ContextState, Env, State, stateType)
from operator import attrgetter
from shapely.geometry import Point, LineString, Polygon

@dataclass(eq=False)
class EgoVehicle():
    x: float = 0.0
    y: float = 0.0
    phi: float = 0.0
    u: float = 0.0
    v: float = 0.0
    w: float = 0.0
    length: float = 0.0
    width: float = 0.0
    action: np.ndarray = np.array([0.0]*2)  # (real) acc, steer
    state: np.ndarray = np.array([0.0]*6)  # x, y, vx, vy, phi, w
    last_action: np.ndarray = np.array([0.0]*2)
    left_boundary_distance: float = 0.0
    right_boundary_distance: float = 0.0
    segment_id: str = 'default'
    junction_id: str = 'default'
    lane_id: str = 'default'
    link_id: str = 'default'
    in_junction: bool = False
    polygon: Polygon = None

    @property
    def ground_position(self) -> Tuple[float, float]:
        return self.state[0].item(), self.state[1].item()


@dataclass(eq=False)
class SurroundingVehicle():
    x: float = 0.0
    y: float = 0.0
    phi: float = 0.0
    rel_x: float = 0.0
    rel_y: float = 0.0
    rel_phi: float = 0.0
    u: float = 0.0
    distance: float = 0.0
    distance_key: ClassVar[attrgetter] = attrgetter("distance")
    length: float = 0.0
    width: float = 0.0
    veh_id: str = 'default'
    lane_id: str = 'default'
    mask: int = 0  # fake vehicle


@dataclass(eq=False)
class LasVSimContext():
    ego: EgoVehicle
    ref_list: List[LineString]
    sur_list: List[SurroundingVehicle]