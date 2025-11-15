from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import mujoco as mj
import numpy as np


@dataclass
class BodyPose:
    pos: np.ndarray  # shape (3, )
    rot: np.ndarray  # quaternion wxyz, shape (4, )


class RobotKinematics:
    """Lightweight MuJoCo-based kinematics helper.

    This utility mirrors the role of :class:`KinematicsModel` used in the forward
    retargeting pipeline, but is tailored for converting robot joint states back
    to human motion.  It exposes helpers for composing full ``qpos`` vectors and
    retrieving world-frame poses for all bodies in the model.
    """

    def __init__(self, xml_path: Path | str) -> None:
        self.xml_path = Path(xml_path) #notice：xml_path is the path to the robot model (xml file)
        if not self.xml_path.exists():
            raise FileNotFoundError(f"Robot MJCF xml not found: {self.xml_path}")

        self.model = mj.MjModel.from_xml_path(str(self.xml_path)) 
        self.data = mj.MjData(self.model) 

        self._body_names: List[str] = []
        for body_id in range(self.model.nbody):
            name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_BODY, body_id)
            # print(f"In init, RobotKinematics: body_id: {body_id}, body_name: {name}")
            # In init, RobotKinematics: body_id: 0, body_name: world
            # In init, RobotKinematics: body_id: 1, body_name: pelvis
            # In init, RobotKinematics: body_id: 2, body_name: left_hip_pitch_link
            # In init, RobotKinematics: body_id: 3, body_name: left_hip_roll_link
            # In init, RobotKinematics: body_id: 4, body_name: left_hip_yaw_link
            # In init, RobotKinematics: body_id: 5, body_name: left_knee_link
            # In init, RobotKinematics: body_id: 6, body_name: left_ankle_pitch_link
            # In init, RobotKinematics: body_id: 7, body_name: left_ankle_roll_link
            # In init, RobotKinematics: body_id: 8, body_name: left_toe_link
            # In init, RobotKinematics: body_id: 9, body_name: pelvis_contour_link
            # In init, RobotKinematics: body_id: 10, body_name: right_hip_pitch_link
            # In init, RobotKinematics: body_id: 11, body_name: right_hip_roll_link
            # In init, RobotKinematics: body_id: 12, body_name: right_hip_yaw_link
            # In init, RobotKinematics: body_id: 13, body_name: right_knee_link
            # In init, RobotKinematics: body_id: 14, body_name: right_ankle_pitch_link
            # In init, RobotKinematics: body_id: 15, body_name: right_ankle_roll_link
            # In init, RobotKinematics: body_id: 16, body_name: right_toe_link
            # In init, RobotKinematics: body_id: 17, body_name: waist_yaw_link
            # In init, RobotKinematics: body_id: 18, body_name: waist_roll_link
            # In init, RobotKinematics: body_id: 19, body_name: torso_link
            # In init, RobotKinematics: body_id: 20, body_name: head_link
            # In init, RobotKinematics: body_id: 21, body_name: head_mocap
            # In init, RobotKinematics: body_id: 22, body_name: imu_in_torso
            # In init, RobotKinematics: body_id: 23, body_name: left_shoulder_pitch_link
            # In init, RobotKinematics: body_id: 24, body_name: left_shoulder_roll_link
            # In init, RobotKinematics: body_id: 25, body_name: left_shoulder_yaw_link
            # In init, RobotKinematics: body_id: 26, body_name: left_elbow_link
            # In init, RobotKinematics: body_id: 27, body_name: left_wrist_roll_link
            # In init, RobotKinematics: body_id: 28, body_name: left_wrist_pitch_link
            # In init, RobotKinematics: body_id: 29, body_name: left_wrist_yaw_link
            # In init, RobotKinematics: body_id: 30, body_name: left_rubber_hand
            # In init, RobotKinematics: body_id: 31, body_name: right_shoulder_pitch_link
            # In init, RobotKinematics: body_id: 32, body_name: right_shoulder_roll_link
            # In init, RobotKinematics: body_id: 33, body_name: right_shoulder_yaw_link
            # In init, RobotKinematics: body_id: 34, body_name: right_elbow_link
            # In init, RobotKinematics: body_id: 35, body_name: right_wrist_roll_link
            # In init, RobotKinematics: body_id: 36, body_name: right_wrist_pitch_link
            # In init, RobotKinematics: body_id: 37, body_name: right_wrist_yaw_link
            # In init, RobotKinematics: body_id: 38, body_name: right_rubber_hand
            self._body_names.append(name if name is not None else "")

    @property
    def nq(self) -> int:
        return self.model.nq

    def compose_qpos(
        self,
        root_pos: Sequence[float],
        root_rot_wxyz: Sequence[float],
        dof_pos: Sequence[float],
    ) -> np.ndarray:
        """Compose a full MuJoCo ``qpos`` vector from root pose + actuated DoFs."""

        qpos = np.zeros(self.nq, dtype=np.float64)
        qpos[:3] = np.asarray(root_pos, dtype=np.float64)
        qpos[3:7] = np.asarray(root_rot_wxyz, dtype=np.float64)
        remaining = self.nq - 7
        dof_array = np.asarray(dof_pos, dtype=np.float64)
        if dof_array.size != remaining:
            raise ValueError(
                f"Unexpected DoF dimension: expected {remaining}, got {dof_array.size}"
            )
        qpos[7:] = dof_array
        return qpos

    def compose_qpos_sequence(
        self,
        root_pos: np.ndarray,
        root_rot_wxyz: np.ndarray,
        dof_pos: np.ndarray,
    ) -> np.ndarray:
        """Stack ``qpos`` vectors for a motion sequence."""

        num_frames = root_pos.shape[0]
        qpos_seq = np.zeros((num_frames, self.nq), dtype=np.float64)
        for idx in range(num_frames):
            qpos_seq[idx] = self.compose_qpos(root_pos[idx], root_rot_wxyz[idx], dof_pos[idx])
        return qpos_seq

    def forward_kinematics(self, qpos: np.ndarray) -> Dict[str, BodyPose]:
        """Compute world-frame pose for every body in the model."""

        if qpos.shape[-1] != self.nq:
            raise ValueError(f"qpos dimension mismatch: expected {self.nq}, got {qpos.shape[-1]}")

        #mark：如果将qpos直接赋值给self.data.qpos，则self.data.qpos会断开与 MuJoCo C 结构体的连接
        self.data.qpos[:] = qpos #notice：self.data是MjData 对象，存储模型的状态信息
        mj.mj_forward(self.model, self.data) #warning：源码复杂，不好直接检查。
        #less /home/retarget/workbench/mujoco_source/src/engine/engine_forward.c

        poses: Dict[str, BodyPose] = {} #define "poses", key is the str, value is the BodyPose which contains pos and quat
        for body_id, body_name in enumerate(self._body_names): #enumerate() ，可以接受一个可迭代对象（如list、tuple、string等），并返回一个枚举对象包含索引和对应的值
            # print(f"body_id: {body_id}, body_name: {body_name}")
            if not body_name:
                continue
            pos = self.data.xpos[body_id].copy()
            quat = self.data.xquat[body_id].copy()  #wxyz顺序 (从源码engine_util_spatial.c中可以证明)
            poses[body_name] = BodyPose(pos=pos, rot=quat)
        return poses

    # def forward_kinematics_sequence(self, qpos_sequence: np.ndarray) -> List[Dict[str, BodyPose]]:
    #     """Run forward kinematics for a batch of ``qpos`` vectors."""

    #     poses: List[Dict[str, BodyPose]] = []
    #     for frame_qpos in qpos_sequence:
    #         poses.append(self.forward_kinematics(frame_qpos))
    #     return poses 


__all__ = ["BodyPose", "RobotKinematics"]

