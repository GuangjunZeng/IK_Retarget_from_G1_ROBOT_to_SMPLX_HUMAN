from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import mink
import mujoco as mj
import numpy as np
import smplx
from scipy.spatial.transform import Rotation as R
from smplx.joint_names import JOINT_NAMES

from .params import REVERSE_IK_CONFIG_DICT, ROBOT_XML_DICT, SMPLX_HUMANOID_XML
from .robot import BodyPose, RobotKinematics


class RobotToSMPLXRetargeting:
    """Mirror of :class:`GeneralMotionRetargeting` that lifts robot motion to SMPL-X via IK."""

    def __init__(
        self,
        robot_type: str,
        smplx_model_path: Path | str,
        gender: str = "neutral",
        ik_config_path: Optional[Path | str] = None,
        solver: str = "daqp",
        damping: float = 5e-1,
        use_velocity_limit: bool = False,
        verbose: bool = False,
    ) -> None:
        self.robot_type = robot_type
        self.verbose = verbose

        # Robot kinematics (used to compute target poses)
        self.robot_xml_path = ROBOT_XML_DICT[robot_type]
        self.robot_kinematics = RobotKinematics(self.robot_xml_path)

        # Solver configuration
        self.solver = solver
        self.damping = damping
        self.use_velocity_limit = use_velocity_limit

        # Reverse IK configuration
        if ik_config_path is None:
            ik_config_path = REVERSE_IK_CONFIG_DICT.get(robot_type)
        if ik_config_path is None:
            raise ValueError(f"No reverse IK config registered for robot type: {robot_type}")
        self.ik_config_path = Path(ik_config_path)
        if not self.ik_config_path.exists():
            raise FileNotFoundError(f"IK config not found: {self.ik_config_path}")
        self.ik_config = self._load_ik_config(self.ik_config_path)

        self.robot_root_name = self.ik_config["robot_root_name"]
        self.smplx_root_name = self.ik_config["human_root_name"]
        self.use_ik_match_table1 = self.ik_config.get("use_ik_match_table1", True)
        self.use_ik_match_table2 = self.ik_config.get("use_ik_match_table2", True)
        self.robot_scale_table = self.ik_config.get("robot_scale_table", {})
        self.ground_height = float(self.ik_config.get("ground_height", 0.0))

        self.ground_offset = 0.0
        self._ground_offset_initialized = False
        self.scaled_robot_data: Dict[str, BodyPose] = {}

        # SMPL-X parametric model (for extracting pose parameters)
        self.smplx_model_path = Path(smplx_model_path)
        if not self.smplx_model_path.exists():
            raise FileNotFoundError(f"SMPL-X model folder not found: {self.smplx_model_path}")

        self.gender = gender
        self.smplx_model = smplx.create(
            model_path=str(self.smplx_model_path),
            model_type="smplx",
            gender=gender,
            use_pca=False,
        )

        self.smplx_joint_names = JOINT_NAMES[: len(self.smplx_model.parents)]
        self.smplx_name_to_idx = {name: i for i, name in enumerate(self.smplx_joint_names)}
        self.smplx_parents = self.smplx_model.parents.detach().cpu().numpy().astype(int)

        self.num_betas = getattr(self.smplx_model, "num_betas", None)
        if self.num_betas is None:
            betas_attr = getattr(self.smplx_model, "betas", None)
            if betas_attr is not None and hasattr(betas_attr, "shape") and betas_attr.shape[-1] > 0:
                self.num_betas = int(betas_attr.shape[-1])
        if self.num_betas is None:
            self.num_betas = 10

        # SMPL-X humanoid MJCF for IK
        self.smplx_xml_path = Path(SMPLX_HUMANOID_XML)
        if not self.smplx_xml_path.exists():
            raise FileNotFoundError(f"SMPL-X humanoid MJCF not found: {self.smplx_xml_path}")

        self.model = mj.MjModel.from_xml_path(str(self.smplx_xml_path))
        self.configuration = mink.Configuration(self.model)

        self.tasks1: List[mink.FrameTask] = []
        self.tasks2: List[mink.FrameTask] = []
        self.smplx_body_to_task1: Dict[str, mink.FrameTask] = {}
        self.smplx_body_to_task2: Dict[str, mink.FrameTask] = {}
        self.robot_body_for_smplx: Dict[str, str] = {}
        self.pos_offsets: Dict[str, np.ndarray] = {}
        self.rot_offsets: Dict[str, np.ndarray] = {}

        self.setup_retarget_configuration()

        self.ik_limits = [mink.ConfigurationLimit(self.model)]
        if self.use_velocity_limit:
            self.ik_limits.append(mink.VelocityLimit(self.model))

        self.body_name_to_id: Dict[str, int] = {}
        for body_id in range(self.model.nbody):
            name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_BODY, body_id)
            if name:
                self.body_name_to_id[name] = body_id

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------
    def _load_ik_config(self, config_path: Path) -> Dict:
        with config_path.open("r") as f:
            return json.load(f)

    def setup_retarget_configuration(self) -> None:
        for table_name in ("ik_match_table1", "ik_match_table2"):
            table = self.ik_config.get(table_name, {})
            for smplx_body, entry in table.items():
                if not entry or smplx_body not in JOINT_NAMES:
                    continue

                robot_body, pos_weight, rot_weight, pos_offset, rot_offset = entry
                self.robot_body_for_smplx[smplx_body] = robot_body
                self.pos_offsets[smplx_body] = np.asarray(pos_offset, dtype=np.float64)
                self.rot_offsets[smplx_body] = np.asarray(rot_offset, dtype=np.float64)

                task = mink.FrameTask(
                    frame_name=smplx_body,
                    frame_type="body",
                    position_cost=pos_weight,
                    orientation_cost=rot_weight,
                    lm_damping=1,
                )

                if table_name == "ik_match_table1" and (pos_weight != 0 or rot_weight != 0):
                    self.tasks1.append(task)
                    self.smplx_body_to_task1[smplx_body] = task
                elif table_name == "ik_match_table2" and (pos_weight != 0 or rot_weight != 0):
                    self.tasks2.append(task)
                    self.smplx_body_to_task2[smplx_body] = task

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def motion_to_smplx_params(
        self,
        root_pos: np.ndarray,
        root_rot_wxyz: np.ndarray,
        dof_pos: np.ndarray,
        betas: Optional[np.ndarray] = None,
        show_progress: bool = False,
    ) -> Dict[str, np.ndarray]:
        qpos_sequence = self.compose_robot_motion_sequence(root_pos, root_rot_wxyz, dof_pos)
        frames = self.map_robot_motion(qpos_sequence, show_progress=show_progress)
        return self.frames_to_smplx_parameters(frames, betas)

    # ------------------------------------------------------------------
    # Core pipeline steps
    # ------------------------------------------------------------------
    def compose_robot_motion_sequence(
        self,
        root_pos: np.ndarray,
        root_rot_wxyz: np.ndarray,
        dof_pos: np.ndarray,
    ) -> np.ndarray:
        return self.robot_kinematics.compose_qpos_sequence(root_pos, root_rot_wxyz, dof_pos)

    def map_robot_motion(
        self,
        qpos_sequence: np.ndarray,
        show_progress: bool = False,
    ) -> List[Dict[str, Dict[str, np.ndarray]]]:
        iterator = range(len(qpos_sequence))
        if show_progress:
            from tqdm import tqdm

            iterator = tqdm(iterator, desc="Processing frames", leave=False)

        mapped_frames: List[Dict[str, Dict[str, np.ndarray]]] = []
        for idx in iterator:
            robot_frame = self.robot_kinematics.forward_kinematics(qpos_sequence[idx])
            self.retarget(robot_frame)
            mapped_frames.append(self.extract_smplx_frame())
        return mapped_frames

    def frames_to_smplx_parameters(
        self,
        smplx_frames: List[Dict[str, Dict[str, np.ndarray]]],
        betas: Optional[np.ndarray],
    ) -> Dict[str, np.ndarray]:
        betas_array = self.prepare_betas(betas)
        root_orient_list: List[np.ndarray] = []
        trans_list: List[np.ndarray] = []
        body_pose_list: List[np.ndarray] = []

        for frame_joints in smplx_frames:
            if self.smplx_root_name in frame_joints:
                root_pos = frame_joints[self.smplx_root_name]["pos"]
                root_rot = frame_joints[self.smplx_root_name]["rot"]
                trans_list.append(root_pos)
                root_orient_list.append(R.from_quat(root_rot[[1, 2, 3, 0]]).as_rotvec())
            else:
                trans_list.append(np.zeros(3, dtype=np.float64))
                root_orient_list.append(np.zeros(3, dtype=np.float64))

            body_pose_list.append(self.compute_local_rotations(frame_joints))

        return {
            "betas": betas_array,
            "root_orient": np.asarray(root_orient_list, dtype=np.float64),
            "trans": np.asarray(trans_list, dtype=np.float64),
            "pose_body": np.asarray(body_pose_list, dtype=np.float64),
        }

    # ------------------------------------------------------------------
    # IK update / solve (mirrors GeneralMotionRetargeting)
    # ------------------------------------------------------------------
    def update_targets(self, robot_data: Dict[str, BodyPose], offset_to_ground: bool = True) -> None:
        robot_data = self.to_numpy(robot_data)
        robot_data = self.scale_robot_data(robot_data)

        if not self._ground_offset_initialized:
            try:
                lowest_z = min(pose.pos[2] for pose in robot_data.values())
                desired = self.ground_height
                self.set_ground_offset(max(0.0, desired - lowest_z))
            except ValueError:
                self.set_ground_offset(0.0)
            self._ground_offset_initialized = True

        robot_data = self.apply_ground_offset(robot_data)

        if offset_to_ground:
            robot_data = self.offset_robot_data_to_ground(robot_data)

        self.scaled_robot_data = robot_data

        if self.use_ik_match_table1:
            for smplx_body, task in self.smplx_body_to_task1.items():
                robot_body = self.robot_body_for_smplx.get(smplx_body)
                pose = robot_data.get(robot_body)
                if pose is None:
                    continue
                target_pos, target_rot = self.compute_target_pose(smplx_body, pose)
                task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3(target_rot), target_pos))

        if self.use_ik_match_table2:
            for smplx_body, task in self.smplx_body_to_task2.items():
                robot_body = self.robot_body_for_smplx.get(smplx_body)
                pose = robot_data.get(robot_body)
                if pose is None:
                    continue
                target_pos, target_rot = self.compute_target_pose(smplx_body, pose)
                task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3(target_rot), target_pos))

    def retarget(self, robot_data: Dict[str, BodyPose], offset_to_ground: bool = True) -> np.ndarray:
        self.update_targets(robot_data, offset_to_ground=offset_to_ground)

        if self.use_ik_match_table1:
            curr_error = self.error1()
            dt = self.configuration.model.opt.timestep
            vel1 = mink.solve_ik(self.configuration, self.tasks1, dt, self.solver, self.damping, self.ik_limits)
            self.configuration.integrate_inplace(vel1, dt)
            next_error = self.error1()
            num_iter = 0
            while curr_error - next_error > 0.001 and num_iter < 10:
                curr_error = next_error
                dt = self.configuration.model.opt.timestep
                vel1 = mink.solve_ik(self.configuration, self.tasks1, dt, self.solver, self.damping, self.ik_limits)
                self.configuration.integrate_inplace(vel1, dt)
                next_error = self.error1()
                num_iter += 1

        if self.use_ik_match_table2:
            curr_error = self.error2()
            dt = self.configuration.model.opt.timestep
            vel2 = mink.solve_ik(self.configuration, self.tasks2, dt, self.solver, self.damping, self.ik_limits)
            self.configuration.integrate_inplace(vel2, dt)
            next_error = self.error2()
            num_iter = 0
            while curr_error - next_error > 0.001 and num_iter < 10:
                curr_error = next_error
                dt = self.configuration.model.opt.timestep
                vel2 = mink.solve_ik(self.configuration, self.tasks2, dt, self.solver, self.damping, self.ik_limits)
                self.configuration.integrate_inplace(vel2, dt)
                next_error = self.error2()
                num_iter += 1

        return self.configuration.data.qpos.copy()

    # ------------------------------------------------------------------
    # Pre-/post-processing utilities
    # ------------------------------------------------------------------
    def to_numpy(self, body_poses: Dict[str, BodyPose]) -> Dict[str, BodyPose]:
        return body_poses

    def scale_robot_data(self, robot_data: Dict[str, BodyPose]) -> Dict[str, BodyPose]:
        if not self.robot_scale_table:
            return robot_data

        root_pose = robot_data.get(self.robot_root_name)
        if root_pose is None:
            return robot_data

        scaled: Dict[str, BodyPose] = {}
        root_pos = root_pose.pos
        for body_name, pose in robot_data.items():
            scale = self.robot_scale_table.get(body_name)
            if scale is None or body_name == self.robot_root_name:
                scaled[body_name] = pose
                continue
            local = pose.pos - root_pos
            scaled_pos = local * scale + root_pos
            scaled[body_name] = BodyPose(pos=scaled_pos, rot=pose.rot)

        return scaled

    def apply_ground_offset(self, body_poses: Dict[str, BodyPose]) -> Dict[str, BodyPose]:
        if self.ground_offset == 0.0:
            return body_poses
        shifted: Dict[str, BodyPose] = {}
        for body_name, pose in body_poses.items():
            shifted_pos = pose.pos + np.array([0.0, 0.0, self.ground_offset])
            shifted[body_name] = BodyPose(pos=shifted_pos, rot=pose.rot)
        return shifted

    def offset_robot_data_to_ground(self, robot_data: Dict[str, BodyPose]) -> Dict[str, BodyPose]:
        return robot_data

    def set_ground_offset(self, ground_offset: float) -> None:
        self.ground_offset = float(ground_offset)

    def compute_target_pose(self, smplx_joint: str, robot_pose: BodyPose) -> tuple[np.ndarray, np.ndarray]:
        pos_offset = self.pos_offsets.get(smplx_joint, np.zeros(3))
        rot_offset = self.rot_offsets.get(smplx_joint)

        robot_R = R.from_quat(robot_pose.rot[[1, 2, 3, 0]])
        if rot_offset is not None:
            offset_R = R.from_quat(rot_offset[[1, 2, 3, 0]])
            smplx_R = robot_R * offset_R.inv()
        else:
            smplx_R = robot_R

        smplx_pos = robot_pose.pos - robot_R.apply(pos_offset)
        smplx_rot = smplx_R.as_quat(scalar_first=True)
        return smplx_pos, smplx_rot

    def extract_smplx_frame(self) -> Dict[str, Dict[str, np.ndarray]]:
        frame: Dict[str, Dict[str, np.ndarray]] = {}
        for joint_name in self.smplx_joint_names:
            body_id = self.body_name_to_id.get(joint_name)
            if body_id is None:
                continue
            pos = self.configuration.data.xpos[body_id].copy()
            quat = self.configuration.data.xquat[body_id].copy()
            frame[joint_name] = {"pos": pos, "rot": quat}
        return frame

    def error1(self) -> float:
        if not self.tasks1:
            return 0.0
        return np.linalg.norm(
            np.concatenate([task.compute_error(self.configuration) for task in self.tasks1])
        )

    def error2(self) -> float:
        if not self.tasks2:
            return 0.0
        return np.linalg.norm(
            np.concatenate([task.compute_error(self.configuration) for task in self.tasks2])
        )

    # ------------------------------------------------------------------
    # Parameter utilities
    # ------------------------------------------------------------------
    def prepare_betas(self, betas: Optional[np.ndarray]) -> np.ndarray:
        if betas is None:
            return np.zeros(self.num_betas, dtype=np.float64)
        betas = np.asarray(betas, dtype=np.float64)
        if betas.shape[0] != self.num_betas:
            if betas.shape[0] < self.num_betas:
                betas = np.pad(betas, (0, self.num_betas - betas.shape[0]), mode="constant")
            else:
                betas = betas[: self.num_betas]
        return betas

    def compute_local_rotations(
        self,
        frame_joints: Dict[str, Dict[str, np.ndarray]],
    ) -> np.ndarray:
        body_pose = np.zeros(63, dtype=np.float64)
        global_rots: Dict[str, R] = {}

        for joint_name, joint_data in frame_joints.items():
            if joint_name not in self.smplx_name_to_idx:
                continue
            rot_quat = joint_data["rot"]
            global_rots[joint_name] = R.from_quat(rot_quat[[1, 2, 3, 0]])

        for i in range(1, min(22, len(self.smplx_joint_names))):
            joint_name = self.smplx_joint_names[i]
            parent_idx = self.smplx_parents[i]
            parent_name = self.smplx_joint_names[parent_idx]
            if joint_name not in global_rots or parent_name not in global_rots:
                continue
            parent_R = global_rots[parent_name]
            joint_R = global_rots[joint_name]
            local_R = parent_R.inv() * joint_R
            body_pose[(i - 1) * 3 : (i - 1) * 3 + 3] = local_R.as_rotvec()

        return body_pose


__all__ = ["RobotToSMPLXRetargeting"]
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import smplx
from scipy.spatial.transform import Rotation as R
from smplx.joint_names import JOINT_NAMES

from .params import REVERSE_IK_CONFIG_DICT, ROBOT_XML_DICT
from .robot import BodyPose, RobotKinematics


class RobotToSMPLXRetargeting:
    """Framework-style wrapper mirroring :class:`GeneralMotionRetargeting` for reverse conversion."""

    def __init__(
        self,
        robot_type: str,
        smplx_model_path: Path | str,
        gender: str = "neutral",
        ik_config_path: Optional[Path | str] = None,
    ) -> None:
        # ===== Robot kinematics assets =====
        self.robot_type = robot_type
        self.robot_xml_path = ROBOT_XML_DICT[robot_type]
        self.robot_kinematics = RobotKinematics(self.robot_xml_path)

        # ===== IK configuration =====
        if ik_config_path is None:
            ik_config_path = REVERSE_IK_CONFIG_DICT.get(robot_type)
        if ik_config_path is None:
            raise ValueError(f"No reverse IK config registered for robot type: {robot_type}")
        self.ik_config_path = Path(ik_config_path)
        if not self.ik_config_path.exists():
            raise FileNotFoundError(f"IK config not found: {self.ik_config_path}")
        self.ik_config = self._load_ik_config(self.ik_config_path)

        self.robot_root_name = self.ik_config["robot_root_name"]
        self.smplx_root_name = self.ik_config["human_root_name"]
        self.use_ik_match_table1 = self.ik_config.get("use_ik_match_table1", True)
        self.use_ik_match_table2 = self.ik_config.get("use_ik_match_table2", True)
        self.robot_scale_table = self.ik_config.get("robot_scale_table", {})
        self.ground_height = float(self.ik_config.get("ground_height", 0.0))

        self.ground_offset = 0.0
        self._ground_offset_initialized = False

        # ===== SMPL-X body model =====
        self.smplx_model_path = Path(smplx_model_path)
        if not self.smplx_model_path.exists():
            raise FileNotFoundError(f"SMPL-X model folder not found: {self.smplx_model_path}")

        self.gender = gender
        self.smplx_model = smplx.create(
            model_path=str(self.smplx_model_path),
            model_type="smplx",
            gender=gender,
            use_pca=False,
        )

        self.smplx_joint_names = JOINT_NAMES[: len(self.smplx_model.parents)]
        self.smplx_name_to_idx = {name: i for i, name in enumerate(self.smplx_joint_names)}
        self.smplx_parents = self.smplx_model.parents.detach().cpu().numpy().astype(int)

        self.num_betas = getattr(self.smplx_model, "num_betas", None)
        if self.num_betas is None:
            betas_attr = getattr(self.smplx_model, "betas", None)
            if betas_attr is not None and hasattr(betas_attr, "shape") and betas_attr.shape[-1] > 0:
                self.num_betas = int(betas_attr.shape[-1])
        if self.num_betas is None:
            self.num_betas = 10

        self.setup_retarget_configuration()

    # ---------------------------------------------------------------------
    # Configuration helpers
    # ---------------------------------------------------------------------
    def _load_ik_config(self, config_path: Path) -> Dict:
        with config_path.open("r") as f:
            return json.load(f)

    def setup_retarget_configuration(self) -> None:
        self.smplx_joint_to_robot: Dict[str, str] = {}
        self.pos_offsets: Dict[str, np.ndarray] = {}
        self.rot_offsets: Dict[str, np.ndarray] = {}
        self.pos_weights: Dict[str, float] = {}
        self.rot_weights: Dict[str, float] = {}

        def register_entry(entry_key: str, entry_value: List, table_name: str) -> None:
            if not entry_value:
                return

            if entry_key in JOINT_NAMES:
                smplx_joint = entry_key
                robot_body = entry_value[0]
            else:
                robot_body = entry_key
                smplx_joint = entry_value[0]

            if smplx_joint not in JOINT_NAMES:
                return

            pos_weight = entry_value[1]
            rot_weight = entry_value[2]
            pos_offset = np.asarray(entry_value[3], dtype=np.float64)
            rot_offset = np.asarray(entry_value[4], dtype=np.float64)

            self.smplx_joint_to_robot[smplx_joint] = robot_body
            self.pos_offsets[smplx_joint] = pos_offset
            self.rot_offsets[smplx_joint] = rot_offset
            self.pos_weights[smplx_joint] = pos_weight
            self.rot_weights[smplx_joint] = rot_weight

        for table_name in ("ik_match_table1", "ik_match_table2"):
            table = self.ik_config.get(table_name, {})
            for key, value in table.items():
                register_entry(key, value, table_name)

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def motion_to_smplx_params(
        self,
        root_pos: np.ndarray,
        root_rot_wxyz: np.ndarray,
        dof_pos: np.ndarray,
        betas: Optional[np.ndarray] = None,
        show_progress: bool = False,
    ) -> Dict[str, np.ndarray]:
        qpos_sequence = self.compose_robot_motion_sequence(root_pos, root_rot_wxyz, dof_pos)
        frames = self.map_robot_motion(qpos_sequence, show_progress=show_progress)
        return self.frames_to_smplx_parameters(frames, betas)

    # ---------------------------------------------------------------------
    # Core pipeline steps
    # ---------------------------------------------------------------------
    def compose_robot_motion_sequence(
        self,
        root_pos: np.ndarray,
        root_rot_wxyz: np.ndarray,
        dof_pos: np.ndarray,
    ) -> np.ndarray:
        return self.robot_kinematics.compose_qpos_sequence(root_pos, root_rot_wxyz, dof_pos)

    def map_robot_motion(
        self,
        qpos_sequence: np.ndarray,
        show_progress: bool = False,
    ) -> List[Dict[str, Dict[str, np.ndarray]]]:
        iterator = range(len(qpos_sequence))
        if show_progress:
            from tqdm import tqdm

            iterator = tqdm(iterator, desc="Processing frames", leave=False)

        mapped_frames: List[Dict[str, Dict[str, np.ndarray]]] = []
        for idx in iterator:
            robot_bodies = self.robot_kinematics.forward_kinematics(qpos_sequence[idx])
            smplx_frame = self.retarget(robot_bodies)
            mapped_frames.append(smplx_frame)
        return mapped_frames

    def frames_to_smplx_parameters(
        self,
        smplx_frames: List[Dict[str, Dict[str, np.ndarray]]],
        betas: Optional[np.ndarray],
    ) -> Dict[str, np.ndarray]:
        betas_array = self.prepare_betas(betas)
        root_orient_list: List[np.ndarray] = []
        trans_list: List[np.ndarray] = []
        body_pose_list: List[np.ndarray] = []

        for frame_joints in smplx_frames:
            if self.smplx_root_name in frame_joints:
                root_pos = frame_joints[self.smplx_root_name]["pos"]
                root_rot = frame_joints[self.smplx_root_name]["rot"]
                trans_list.append(root_pos)
                root_orient_list.append(R.from_quat(root_rot[[1, 2, 3, 0]]).as_rotvec())
            else:
                trans_list.append(np.zeros(3, dtype=np.float64))
                root_orient_list.append(np.zeros(3, dtype=np.float64))

            body_pose_list.append(self.compute_local_rotations(frame_joints))

        return {
            "betas": betas_array,
            "root_orient": np.asarray(root_orient_list, dtype=np.float64),
            "trans": np.asarray(trans_list, dtype=np.float64),
            "pose_body": np.asarray(body_pose_list, dtype=np.float64),
        }



    def update_targets(self, robot_data: Dict[str, BodyPose], offset_to_ground: bool = True) -> Dict[str, BodyPose]:
        robot_data = self.to_numpy(robot_data)
        robot_data = self.scale_robot_data(robot_data)
        robot_data = self.offset_robot_data(robot_data)

        if not self._ground_offset_initialized:
            try:
                lowest_z = min(pose.pos[2] for pose in robot_data.values())
                desired = self.ground_height
                self.set_ground_offset(max(0.0, desired - lowest_z))
            except ValueError:
                self.set_ground_offset(0.0)
            self._ground_offset_initialized = True

        robot_data = self.apply_ground_offset(robot_data)

        if offset_to_ground:
            robot_data = self.offset_robot_data_to_ground(robot_data)

        self.scaled_robot_data = robot_data
        return robot_data

    def retarget(self, robot_data: Dict[str, BodyPose], offset_to_ground: bool = True) -> Dict[str, Dict[str, np.ndarray]]:
        targets = self.update_targets(robot_data, offset_to_ground=offset_to_ground)

        smplx_targets: Dict[str, Dict[str, np.ndarray]] = {}
        for smplx_joint, robot_body in self.smplx_joint_to_robot.items():
            pose = targets.get(robot_body)
            if pose is None:
                continue
            smplx_pos, smplx_rot = self.apply_reverse_offsets(smplx_joint, pose)
            smplx_targets[smplx_joint] = {"pos": smplx_pos, "rot": smplx_rot}
        return smplx_targets

    def to_numpy(self, body_poses: Dict[str, BodyPose]) -> Dict[str, BodyPose]:
        return body_poses

    def scale_robot_data(self, robot_data: Dict[str, BodyPose]) -> Dict[str, BodyPose]:
        if not self.robot_scale_table:
            return robot_data

        root_pose = robot_data.get(self.robot_root_name)
        if root_pose is None:
            return robot_data

        scaled: Dict[str, BodyPose] = {}
        root_pos = root_pose.pos
        for body_name, pose in robot_data.items():
            scale = self.robot_scale_table.get(body_name)
            if scale is None or body_name == self.robot_root_name:
                scaled[body_name] = pose
                continue
            local = pose.pos - root_pos
            scaled_pos = local * scale + root_pos
            scaled[body_name] = BodyPose(pos=scaled_pos, rot=pose.rot)

        return scaled

    def offset_robot_data(self, robot_data: Dict[str, BodyPose]) -> Dict[str, BodyPose]:
        offset_data: Dict[str, BodyPose] = {}
        for body_name, pose in robot_data.items():
            pos_offset = self.pos_offsets.get(body_name, np.zeros(3))
            rot_offset = self.rot_offsets.get(body_name)

            offset_data[body_name] = pose
            if rot_offset is not None:
                updated_rot = (R.from_quat(pose.rot[[1, 2, 3, 0]]) * R.from_quat(rot_offset[[1, 2, 3, 0]])).as_quat(
                    scalar_first=True
                )
                offset_data[body_name] = BodyPose(pos=pose.pos, rot=updated_rot)

            offset_data[body_name] = BodyPose(
                pos=offset_data[body_name].pos - R.from_quat(offset_data[body_name].rot[[1, 2, 3, 0]]).apply(pos_offset),
                rot=offset_data[body_name].rot,
            )

        return offset_data

    def offset_robot_data_to_ground(self, robot_data: Dict[str, BodyPose]) -> Dict[str, BodyPose]:
        return robot_data

    def apply_ground_offset(self, body_poses: Dict[str, BodyPose]) -> Dict[str, BodyPose]:
        if self.ground_offset == 0.0:
            return body_poses
        shifted: Dict[str, BodyPose] = {}
        for body_name, pose in body_poses.items():
            shifted_pos = pose.pos + np.array([0.0, 0.0, self.ground_offset])
            shifted[body_name] = BodyPose(pos=shifted_pos, rot=pose.rot)
        return shifted

    def set_ground_offset(self, ground_offset: float) -> None:
        self.ground_offset = float(ground_offset)

    def apply_reverse_offsets(self, smplx_joint: str, pose: BodyPose) -> tuple[np.ndarray, np.ndarray]:
        return pose.pos, pose.rot

    # ---------------------------------------------------------------------
    # Parameter utilities
    # ---------------------------------------------------------------------
    def prepare_betas(self, betas: Optional[np.ndarray]) -> np.ndarray:
        if betas is None:
            return np.zeros(self.num_betas, dtype=np.float64)
        betas = np.asarray(betas, dtype=np.float64)
        if betas.shape[0] != self.num_betas:
            if betas.shape[0] < self.num_betas:
                betas = np.pad(betas, (0, self.num_betas - betas.shape[0]), mode="constant")
            else:
                betas = betas[: self.num_betas]
        return betas

    def compute_local_rotations(
        self,
        frame_joints: Dict[str, Dict[str, np.ndarray]],
    ) -> np.ndarray:
        body_pose = np.zeros(63, dtype=np.float64)
        global_rots: Dict[str, R] = {}

        for joint_name, joint_data in frame_joints.items():
            if joint_name not in self.smplx_name_to_idx:
                continue
            rot_quat = joint_data["rot"]
            global_rots[joint_name] = R.from_quat(rot_quat[[1, 2, 3, 0]])

        for i in range(1, min(22, len(self.smplx_joint_names))):
            joint_name = self.smplx_joint_names[i]
            parent_idx = self.smplx_parents[i]
            parent_name = self.smplx_joint_names[parent_idx]
            if joint_name not in global_rots or parent_name not in global_rots:
                continue
            parent_R = global_rots[parent_name]
            joint_R = global_rots[joint_name]
            local_R = parent_R.inv() * joint_R
            body_pose[(i - 1) * 3 : (i - 1) * 3 + 3] = local_R.as_rotvec()

        return body_pose


__all__ = ["RobotToSMPLXRetargeting"]

