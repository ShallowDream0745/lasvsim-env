import numpy as np
from functools import partial
from typing import Tuple, Dict, List, Optional, Any
class TrajectoryProcessor:
    def __init__(self, dense_ref_mode: str, dense_ref_param: Optional[Any] = None):
        dense_func_name = f"dense_ref_{dense_ref_mode}"
        dense_ref = getattr(self, dense_func_name)
        self.dense_ref = partial(dense_ref, dense_ref_param=dense_ref_param)

        self.traj_vocabulary = None
        if dense_ref_mode == "vocabulary":
            self.traj_vocabulary = self.get_vocabulary(traj_path = dense_ref_param)
        
    def get_vocabulary(self, traj_path: str) -> np.ndarray:
        """
        get vocabulary from trajectory file.

        Parameters:
        traj_path (str): path of *.npy trajectory file in ego vehicle frame
                                

        Returns:
        np.ndarray: traj_vocabulary in ego vehicle frame
        """
        dt = 0.1 # FIXME: hard code
        ref = np.load(traj_path) # ref：[R, N, 3]
        ref_v = np.sum(np.diff(ref[:, :, :2], axis=1) ** 2, axis=-1) ** 0.5 / dt # [R, N-1]
        last_v = ref_v[:, -1]
        ref_v = np.concatenate([ref_v, last_v[:, None]], axis=1) # [R, N]
        return np.concatenate([ref, ref_v[:, :, None]], axis=-1) # [R, N, 4]
    
    def generate_bezier_curve_with_phi(self, origin_point:np.array, dest_point:np.array, n_points=100) -> np.array:
        x0, y0, phi0, v_o = origin_point
        x3, y3, phi3, v_d = dest_point
        delta_v = v_d - v_o
        p1_x = x0 + np.cos(phi0) * 0.4 * np.linalg.norm([x3 - x0, y3 - y0])
        p1_y = y0 + np.sin(phi0) * 0.4 * np.linalg.norm([x3 - x0, y3 - y0])
        
        p2_x = x3 - np.cos(phi3) * 0.4 * np.linalg.norm([x3 - x0, y3 - y0])
        p2_y = y3 - np.sin(phi3) * 0.4 * np.linalg.norm([x3 - x0, y3 - y0])
        
        P0 = np.array([x0, y0])
        P1 = np.array([p1_x, p1_y])
        P2 = np.array([p2_x, p2_y])
        P3 = np.array([x3, y3])

        t_values = np.linspace(0, 1, n_points)

        bezier_points = []
        for t in t_values:
            x = (1 - t)**3 * P0[0] + 3 * (1 - t)**2 * t * P1[0] + 3 * (1 - t) * t**2 * P2[0] + t**3 * P3[0]
            y = (1 - t)**3 * P0[1] + 3 * (1 - t)**2 * t * P1[1] + 3 * (1 - t) * t**2 * P2[1] + t**3 * P3[1]

            dx = 3 * (1 - t)**2 * (P1[0] - P0[0]) + 6 * (1 - t) * t * (P2[0] - P1[0]) + 3 * t**2 * (P3[0] - P2[0])
            dy = 3 * (1 - t)**2 * (P1[1] - P0[1]) + 6 * (1 - t) * t * (P2[1] - P1[1]) + 3 * t**2 * (P3[1] - P2[1])

            phi = np.arctan2(dy, dx)
            bezier_points.append(np.array([x, y, phi, v_o + delta_v * t]))

        bezier_points = np.array(bezier_points)
        return bezier_points

    def dense_ref_bezier(self, ref_param: np.ndarray, robot_state: np.ndarray = None, dense_ref_param: Optional[list] = None) -> np.ndarray:
        """
        Densify reference parameters by add Bezier curves.

        Parameters:
        ref_param (np.ndarray): Input reference parameters with shape [R, 2N+1, 4].
                                Each element represents [ref_x, ref_y, ref_phi, ref_v].

        Returns:
        np.ndarray: Densified reference parameters with shape [R+(R-1)*2*len(ratio_list), 2N+1, 4].
                    Each element represents [ref_x, ref_y, ref_phi, ref_v].
        """
        ratio_list = [1] if dense_ref_param is None else dense_ref_param
        bezier_list=[]
        num_point = ref_param.shape[-2]
        for sample_ratio in ratio_list:
            target_index = int(sample_ratio * num_point)
            for i in range(ref_param.shape[0]):
                if i == 0:
                    ref_bezier = self.generate_bezier_curve_with_phi(ref_param[i][0], ref_param[i+1][target_index-1],target_index)
                    if int(num_point-target_index)!=0:
                        bezier_list.append(np.concatenate((ref_bezier,ref_param[i+1][-int(num_point-target_index):])))
                    else:
                        bezier_list.append(ref_bezier)
                elif i==ref_param.shape[0]-1:
                    ref_bezier = self.generate_bezier_curve_with_phi(ref_param[i][0], ref_param[i-1][target_index-1],target_index)
                    if int(num_point-target_index)!=0:
                        bezier_list.append(np.concatenate((ref_bezier,ref_param[i-1][-int(num_point-target_index):])))
                    else:
                        bezier_list.append(ref_bezier)
                else:
                    ref_bezier = self.generate_bezier_curve_with_phi(ref_param[i][0], ref_param[i-1][target_index-1],target_index)
                    if int(num_point-target_index)!=0:
                        bezier_list.append(np.concatenate((ref_bezier,ref_param[i-1][-int(num_point-target_index):])))
                    else:
                        bezier_list.append(ref_bezier)
                    
                    ref_bezier = self.generate_bezier_curve_with_phi(ref_param[i][0], ref_param[i+1][target_index-1],target_index)
                    if int(num_point-target_index)!=0:
                        bezier_list.append(np.concatenate((ref_bezier,ref_param[i+1][-int(num_point-target_index):])))
                    else:
                        bezier_list.append(ref_bezier)
                        
        return np.concatenate([np.array(bezier_list),ref_param])

    def dense_ref_boundary(self, ref_param: np.ndarray, robot_state: np.ndarray = None, dense_ref_param = None):
        """
        Densify reference parameters by add boundaries.

        Parameters:
        ref_param (np.ndarray): Input reference parameters with shape [R, 2N+1, 4].
                                Each element represents [ref_x, ref_y, ref_phi, ref_v].

        Returns:
        np.ndarray: Densified reference parameters with shape [2R-1, 2N+1, 4].
                    Each element represents [ref_x, ref_y, ref_phi, ref_v].
        """

        A, B, C = ref_param.shape
        ret = np.zeros((2*A - 1, B, C))

        for j in range(A):
            ret[2*j, :, :] = ref_param[j, :, :]
        for j in range(A - 1):
            ret[2*j + 1, :, :] = (ref_param[j, :, :] + ref_param[j + 1, :, :]) / 2

        return ret

    def dense_ref_no_dense(self, ref_param: np.ndarray, robot_state: np.ndarray = None, dense_ref_param = None):
        return ref_param

    def dense_ref_vocabulary(self, ref_param: np.ndarray, robot_state: np.ndarray = None, dense_ref_param = None):
        """
        transform self.traj_vocabulary from ego vehicle frame to world frame.
        xR_world = xE_world + xR_ego * cos(phiE) - yR_ego * sin(phiE)
        yR_world = yE_world + xR_ego * sin(phiE) + yR_ego * cos(phiE)
        phiR_world = phiE_world + phiR_ego
        shape of traj_vocabulary: [R, N, 4]

        Parameters:
        robot_state (np.ndarray): [x, y, vx, vy, phi, omega, last_last_action, last_action]
                                

        Returns:
        np.ndarray: traj_vocabulary in world frame
        """
        vocabulary_in_ego = self.traj_vocabulary
        ego_in_world = np.concatenate([robot_state[..., :2], robot_state[..., 4:5]], axis=-1)[None, None, :] # [1, 1, 3] x, y, phi
        vocabulary_in_world = np.zeros_like(vocabulary_in_ego)
        vocabulary_in_world[..., 0] = vocabulary_in_ego[..., 0] * np.cos(ego_in_world[..., 2]) - vocabulary_in_ego[..., 1] * np.sin(ego_in_world[..., 2]) + ego_in_world[..., 0]
        vocabulary_in_world[..., 1] = vocabulary_in_ego[..., 0] * np.sin(ego_in_world[..., 2]) + vocabulary_in_ego[..., 1] * np.cos(ego_in_world[..., 2]) + ego_in_world[..., 1]
        vocabulary_in_world[..., 2] = vocabulary_in_ego[..., 2] + ego_in_world[..., 2]
        vocabulary_in_world[..., 3] = vocabulary_in_ego[..., 3]
        return vocabulary_in_world


def compute_intervals_in_junction(total_num: int,
                      ref_v_junction: float,
                      dt: float,) -> Tuple[np.ndarray, np.ndarray]:
    # velocity planning for green mode
    intervals = np.zeros(total_num)
    ref_v = np.zeros(total_num)
    # ref_v from ref_v_junction to ref_v_lane
    # ahead_length = current_part['length'] - position_on_ref
    # n1 = int(ahead_length / ref_v_junction)
    # intervals[:n1] = ref_v_junction * dt
    # intervals[n1:] = ref_v_lane * dt
    # ref_v[:n1] = ref_v_junction
    # ref_v[n1:] = ref_v_lane

    # ref_v keeps ref_v_junction for all points if ego is in the junction
    intervals[:] = ref_v_junction * dt
    ref_v[:] = ref_v_junction
    return intervals, ref_v


def compute_intervals_initsegment_green(position_on_ref: float,
                      current_part: Dict[str, float],
                      total_num: int,
                      ref_v_lane: float,
                      ref_v_junction: float,
                      dt: float,
                      am: float) -> Tuple[np.ndarray, np.ndarray]:
    # velocity planning for green mode
    intervals = np.zeros(total_num)
    ref_v = np.zeros(total_num)
    position_on_ref = np.clip(position_on_ref, a_min=0,
                              a_max=current_part['length'])
    # ref_v from ref_v_lane to ref_v_junction
    ahead_length = current_part['length'] - position_on_ref
    v0 = np.sqrt(2 * am * ahead_length + ref_v_junction ** 2)
    ref_v = [v0 - am * dt * i for i in range(total_num)]
    # when v < v_junction and v > v_lane, a = am
    a_list = [am if v > ref_v_junction and v <
                    ref_v_lane else 0 for v in ref_v]
    ref_v = np.clip(ref_v, a_min=ref_v_junction, a_max=ref_v_lane)
    intervals = [v * dt - 0.5 * a * dt * dt for v, a in zip(ref_v, a_list)]
    intervals = np.clip(
        intervals, a_min=ref_v_junction * dt, a_max=ref_v_lane * dt)
    return intervals, ref_v


def compute_intervals_initsegment_red(position_on_ref: float,
                      current_part: Dict[str, float],
                      total_num: int,
                      ref_v_lane: float,
                      dt: float,
                      am: float,
                      min_ahead_lane_length: float) -> Tuple[np.ndarray, np.ndarray]:
    # velocity planning for red mode
    intervals = np.zeros(total_num)
    ref_v = np.zeros(total_num)
    position_on_ref = np.clip(position_on_ref, a_min=0,
                              a_max=current_part['length'])
    # ref_v from ref_v_lane to 0
    ahead_length = current_part['length'] - position_on_ref - (min_ahead_lane_length - 4) # FIXME: 4 is 0.8*ego_length, which has been considered in current_part['length'] for red light
    if ahead_length < 0.01:
        intervals = np.zeros((total_num, ))
        ref_v = np.zeros((total_num, ))
    else:
        v0 = np.sqrt(2 * am * ahead_length)
        ref_v = [v0 - am * dt * i for i in range(total_num)]
        # when v < v_junction and v > v_lane, a = am
        a_list = [am if v > 0 and v <
                        ref_v_lane else 0 for v in ref_v]
        ref_v = np.clip(ref_v, a_min=0, a_max=ref_v_lane)
        intervals = [v * dt for v, a in zip(ref_v, a_list)]
        intervals = np.clip(
            intervals, a_min=0 * dt, a_max=ref_v_lane * dt)
    return intervals, ref_v


def compute_intervals(ref_info: List[Dict[str, float]],
                      total_num: int,
                      cur_v: float,
                      ref_v_lane: float,
                      dt: float,
                      acc: float,
                      ) -> Tuple[np.ndarray, np.ndarray]:
    # velocity planning for green mode
    # total_num = total_num - 1
    ref_v = np.ones(total_num) * ref_v_lane
    current_part = ref_info[0]

    assert current_part['destination'] == True, "Error ref_line"
    if acc is not None: # calculate ref_v using acc
        ref_v = [cur_v + acc * dt * i for i in range(total_num)]
        ref_v = np.clip(ref_v, a_min=0, a_max=ref_v_lane+10)

    intervals = ref_v * dt

    return intervals, ref_v