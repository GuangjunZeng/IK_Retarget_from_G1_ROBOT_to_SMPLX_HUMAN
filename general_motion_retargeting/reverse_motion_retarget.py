from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import mink
import mujoco as mj
import numpy as np
import smplx
import torch
from scipy.spatial.transform import Rotation as R
from smplx.joint_names import JOINT_NAMES
import viser
import viser.transforms as tf
from viser.extras import ViserUrdf

from .params import REVERSE_IK_CONFIG_DICT, ROBOT_XML_DICT, SMPLX_HUMANOID_XML
from .robot import BodyPose, RobotKinematics


class RobotToSMPLXRetargeting:
    """Mirror of :class:`GeneralMotionRetargeting` that lifts robot motion to SMPL-X via IK."""

    _ROOT_TO_PELVIS_ROT = np.array(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    _PELVIS_TO_ROOT_ROT = _ROOT_TO_PELVIS_ROT.T

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
        actual_human_height: float = None,
    ) -> None:
        self.robot_type = robot_type
        self.verbose = verbose

        # Robot kinematics (used to compute target poses)
        self.robot_xml_path = ROBOT_XML_DICT[robot_type]
        #notice：将g1 robot model （xml file）传入到RobotKinematics类中，所以RobotKinematics.xml_path = self.robot_xml_path
        self.robot_kinematics = RobotKinematics(self.robot_xml_path)

        # Solver configuration
        self.solver = solver
        self.damping = damping
        self.use_velocity_limit = use_velocity_limit
        self.max_iter = 10

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
        self.use_ik_match_table1 = self.ik_config.get("use_ik_match_table1", True) #查找key "use_ik_match_table1" 的value，如果没有则默认返回True
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
        ) # smplx_model is from SMPLX_FEMALE/SMPLX_MALE/SMPLX_NEUTRAL.npz file

        self.smplx_joint_names = JOINT_NAMES[: len(self.smplx_model.parents)] #len(self.smplx_model.parents) = 关节数（由 parents 长度决定）
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

        #notice： self.model is the smplx model.
        self.model = mj.MjModel.from_xml_path(str(self.smplx_xml_path))
        self.configuration = mink.Configuration(self.model)

        self.body_name_to_id: Dict[str, int] = {}
        self.body_alias_map: Dict[str, str] = {}
        for body_id in range(self.model.nbody):
            name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_BODY, body_id)
            if name:
                self.body_name_to_id[name] = body_id
                lower = name.lower() #有的name是大写开头，如：L_Hip
                self.body_alias_map[lower] = name # l_hip -- L_Hip (大小写)
                if lower.startswith("l_"):
                    self.body_alias_map["left_" + lower[2:]] = name # left_hip -- L_hip (l/r缩写)
                if lower.startswith("r_"):
                    self.body_alias_map["right_" + lower[2:]] = name

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
        
        if actual_human_height is not None:
            ratio = self.ik_config["human_height_assumption"] / actual_human_height
            print(f"in reverse retargeting, ratio: {ratio}") #000005.npz: ratio= 1.052949396067974
            for key in self.robot_scale_table.keys():
                self.robot_scale_table[key] = self.robot_scale_table[key] * ratio
            
            # 打印完整的 robot_scale_table 以查看精度
            # print("\n=== Updated robot_scale_table ===")
            # for key, value in self.robot_scale_table.items():
            #     print(f"  {key}: {value:.15f}")  # 显示15位小数
            # print("==================================\n")
        else:
            ratio = 1

        #000005.npz:
        # pelvis: 1.169943656414483
        # torso_link: 1.169943656414483
        # left_hip_roll_link: 1.169943656414483
        # right_hip_roll_link: 1.169943656414483
        # left_knee_link: 1.169943656414483
        # right_knee_link: 1.169943656414483
        # left_toe_link: 1.169943656414483
        # right_toe_link: 1.169943656414483
        # left_shoulder_yaw_link: 1.316186745084968
        # right_shoulder_yaw_link: 1.316186745084968
        # left_elbow_link: 1.316186745084968
        # right_elbow_link: 1.316186745084968
        # left_wrist_yaw_link: 1.316186745084968
        # right_wrist_yaw_link: 1.316186745084968

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------
    def _load_ik_config(self, config_path: Path) -> Dict:
        with config_path.open("r") as f:
            return json.load(f)

    def resolve_smplx_body_name(self, candidate: str) -> str:
        custom_aliases = {
            "spine1": "torso",
            "spine2": "spine",
            "spine3": "chest",     #spine3 exist in g1_to_smplx.json, but not in smplx MJCF model
            #notice: spine3 is chest in smplx MJCF model. The torse_link (in g1_to_smplx.json) is not the same meaning as the torse in smplx MJCF model.
            "left_foot": "l_toe",  #left_foot exist in g1_to_smplx.json, but not in smplx MJCF model
            "right_foot": "r_toe", #right_foot exist in g1_to_smplx.json, but not in smplx MJCF model

            "left_collar": "l_thorax",
            "right_collar": "r_thorax",
        }

        key = candidate.lower() 
        if key == "jaw" or key == "left_eye_smplhf" or key == "right_eye_smplhf":
            return key

        alias_key = custom_aliases.get(key, key) #eg: 第一级, spine3->chest
        if alias_key in self.body_alias_map:     #eg: 第二级, chest->Chest
            return self.body_alias_map[alias_key]  #notice: body_alias_map includes all body names in smplx MJCF model from "smplx_humanoid.xml" file
        # fallback: try original candidate (in case config already uses MJCF name)
        if candidate in self.body_name_to_id: #查找self.body_name_to_id的key，不是value
            return candidate
        raise KeyError(
            f"SMPL-X body '{candidate}' not found in humanoid MJCF. Available bodies: {list(self.body_name_to_id.keys())}"
        )

    # high priority: set the configuration parameters for motion retargeting.
    def setup_retarget_configuration(self) -> None:
        for table_name in ("ik_match_table1", "ik_match_table2"):
            table = self.ik_config.get(table_name, {})
            for smplx_body, entry in table.items():
                if not entry or smplx_body not in JOINT_NAMES:
                    print(f"smplx_body: {smplx_body} not found in JOINT_NAMES or the entry is empty")
                    continue

                robot_body, pos_weight, rot_weight, pos_offset, rot_offset = entry
                self.robot_body_for_smplx[smplx_body] = robot_body     #key is the smplx body name in ik table, value is the robot body name in ik table 
                frame_name = self.resolve_smplx_body_name(smplx_body)  #notice: frame_name is body name in smplx MJCF format
                self.pos_offsets[smplx_body] = np.asarray(pos_offset, dtype=np.float64)
                self.rot_offsets[smplx_body] = R.from_quat(
                    np.asarray(rot_offset, dtype=np.float64), scalar_first=True
                )
                # print(f"smplx_body: {smplx_body}, frame_name in the ik task: {frame_name}") 
                

                task = mink.FrameTask(
                    frame_name=frame_name,  #notice: frame_name is body name in smplx MJCF format
                    frame_type="body",
                    position_cost=pos_weight,
                    orientation_cost=rot_weight,
                    lm_damping=1,
                )
                #warning: smplx_body 是 ik table中的key
                if table_name == "ik_match_table1" and (pos_weight != 0 or rot_weight != 0):
                    self.tasks1.append(task)
                    self.smplx_body_to_task1[smplx_body] = task #
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
        smplx_frames: List[Dict[str, Dict[str, np.ndarray]]],  #notice: smplx_frames includes dicts {body_name: {"pos": pos, "rot": quat}}
        betas: Optional[np.ndarray],
        qpos_list: List[np.ndarray],
    ) -> Dict[str, np.ndarray]:
        betas_array = self.prepare_betas(betas)
        pelvis_trans_list: List[np.ndarray] = []
        pelvis_quat_xyzw_list: List[np.ndarray] = []
        smpl_trans_list: List[np.ndarray] = []
        smpl_quat_xyzw_list: List[np.ndarray] = []
        body_pose_list: List[np.ndarray] = []

        pelvis_joint_offset = self.compute_pelvis_joint_offset(betas_array)

        # for frame_joints in smplx_frames: #iterate through the joints data for each frame
        for frame_joints, qpos in zip(smplx_frames, qpos_list):
            if self.smplx_root_name in frame_joints:
                pelvis_pos = frame_joints[self.smplx_root_name]["pos"]
                pelvis_rot = frame_joints[self.smplx_root_name]["rot"]  #wxyz format: 关节的绝对旋转
                pelvis_quat_xyzw = pelvis_rot[[1, 2, 3, 0]]
                pelvis_trans_list.append(pelvis_pos)
                pelvis_quat_xyzw_list.append(pelvis_quat_xyzw)


                # R_adjust_mat =  self._PELVIS_TO_ROOT_ROT
                # R_adjust_mat_inv = R_adjust_mat.T
                # root_rotations_mat = pelvis_rot_mat @ R_adjust_mat_inv
                # root_rotations = tf.SO3.from_matrix(root_rotations_mat)
                # root_quat_xyzw = root_rotations.as_quaternion_xyzw()
                # root_quat_wxyz = np.array([root_quat_xyzw[3], root_quat_xyzw[0], root_quat_xyzw[1], root_quat_xyzw[2]])


                # pelvis_rot_mat = R.from_quat(pelvis_quat_xyzw).as_matrix()
                # smpl_rot_mat = pelvis_rot_mat @ self._PELVIS_TO_ROOT_ROT
                # smpl_quat_xyzw = R.from_matrix(smpl_rot_mat).as_quat()
                # smpl_trans = pelvis_pos - smpl_rot_mat @ pelvis_joint_offset
                smpl_quat_wxyz = np.asarray(qpos[3:7], dtype=np.float64) 
                smpl_trans = np.asarray(qpos[:3], dtype=np.float64)
                smpl_trans[1] += 0.34 #?
                smpl_quat_xyzw = np.array(
                    [smpl_quat_wxyz[1], smpl_quat_wxyz[2], smpl_quat_wxyz[3], smpl_quat_wxyz[0]],
                    dtype=np.float64,
                )
                smpl_trans_list.append(smpl_trans.astype(np.float64))
                smpl_quat_xyzw_list.append(smpl_quat_xyzw.astype(np.float64))

                
            else:
                pass
                # trans_list.append(np.zeros(3, dtype=np.float64))
                # root_orient_list.append(np.zeros(3, dtype=np.float64))

            body_pose_list.append(self.compute_local_rotations(frame_joints)) 
        
        #warning:尚未验证smpl_trans, smpl_quat_xyzw计算的正确性。(计算逻辑来源: sRetarget/scripts/preparation/reference/reference_data.py)

        return {
            "betas": betas_array,
            "pelvis_trans": np.asarray(pelvis_trans_list, dtype=np.float64),
            "pelvis_quat_xyzw": np.asarray(pelvis_quat_xyzw_list, dtype=np.float64),
            "smpl_trans": np.asarray(smpl_trans_list, dtype=np.float64),
            "smpl_quat_xyzw": np.asarray(smpl_quat_xyzw_list, dtype=np.float64),
            "pose_body": np.asarray(body_pose_list, dtype=np.float64),
            "pose_hand": np.zeros((len(body_pose_list), 90), dtype=np.float64),
        }

    def frames_to_smplx_parameters_smplxmodel(
        self,
        smplx_frames: List[Dict[str, Dict[str, np.ndarray]]],
        qpos_list: List[np.ndarray],
        betas: Optional[np.ndarray],
    ) -> Dict[str, np.ndarray]:
        if len(smplx_frames) != len(qpos_list):
            raise ValueError("smplx_frames and qpos_list must have the same length.")

        num_frames = len(smplx_frames)
        betas_array = self.prepare_betas(betas)

        if num_frames == 0:
            empty = np.zeros((0,), dtype=np.float64)
            return {
                "betas": betas_array,
                "pose_body": empty.reshape(0, 63),
                "pose_hand": empty.reshape(0, 90),
                "smpl_trans": empty.reshape(0, 3),
                "smpl_quat_xyzw": empty.reshape(0, 4),
                "pelvis_trans": empty.reshape(0, 3),
                "pelvis_quat_xyzw": empty.reshape(0, 4),
                "joints_local": empty.reshape(0, 55, 3),
            }

        body_pose_list: List[np.ndarray] = []
        root_quat_xyzw_list: List[np.ndarray] = []
        transl_list: List[np.ndarray] = []
        pelvis_trans_list: List[np.ndarray] = []
        pelvis_quat_xyzw_list: List[np.ndarray] = []
        pelvis_joint_offset = self.compute_pelvis_joint_offset(betas_array).astype(np.float64)  # warning:
        print(f"pelvis_joint_offset: {pelvis_joint_offset}") #[ 0.00084991 -0.37823778  0.00606466]  (000005.npz)

        for frame_joints, qpos in zip(smplx_frames, qpos_list):
            body_pose_list.append(self.compute_local_rotations(frame_joints))

            pelvis_trans = np.asarray(qpos[:3], dtype=np.float64)
            pelvis_quat_wxyz = np.asarray(qpos[3:7], dtype=np.float64)
            pelvis_quat_xyzw = np.array(
                [pelvis_quat_wxyz[1], pelvis_quat_wxyz[2], pelvis_quat_wxyz[3], pelvis_quat_wxyz[0]],
                dtype=np.float64,
            )
            pelvis_trans_list.append(pelvis_trans)
            pelvis_quat_xyzw_list.append(pelvis_quat_xyzw)

            pelvis_rot_mat = R.from_quat(pelvis_quat_xyzw).as_matrix()
            # root_rot_mat = pelvis_rot_mat @ self._PELVIS_TO_ROOT_ROT # 这一步是多余的，因为这一步在从amass raw data生成reference data才需要。smplx to robot 或者反过来时已经对齐了。
            root_rot_mat = pelvis_rot_mat.copy()
            root_rot = tf.SO3.from_matrix(root_rot_mat)
            root_quat_xyzw = root_rot.as_quaternion_xyzw()
            # root_trans = pelvis_trans - root_rot_mat @ pelvis_joint_offset  
            root_trans = pelvis_trans - pelvis_joint_offset  
            print(f"root_trans: {root_trans}")
            print(f"pelvis_trans: {pelvis_trans}")
            print(f"root_quat_xyzw: {root_quat_xyzw}")
            print(f"pelvis_quat_xyzw: {pelvis_quat_xyzw}")

            root_quat_xyzw_list.append(root_quat_xyzw)  #warning: 这个计算怎么验证就是xyzw顺序?
            transl_list.append(root_trans)

        pose_body = np.asarray(body_pose_list, dtype=np.float64)
        smpl_trans = np.asarray(transl_list, dtype=np.float64)
        root_quat_xyzw = np.asarray(root_quat_xyzw_list, dtype=np.float64)
        pose_hand = np.zeros((num_frames, 90), dtype=np.float64)

        # root_rotations = R.from_quat(root_quat_xyzw)
        # global_orient = root_rotations.as_rotvec()

        # device = torch.device("cpu")
        # try:
        #     device = next(self.smplx_model.parameters()).device
        # except StopIteration:
        #     pass

        # betas_tensor = torch.from_numpy(betas_array.astype(np.float32)).unsqueeze(0).to(device)
        # betas_batched = betas_tensor.repeat(num_frames, 1)
        # global_orient_tensor = torch.from_numpy(global_orient.astype(np.float32)).to(device)
        # body_pose_tensor = torch.from_numpy(pose_body.astype(np.float32)).to(device)
        # transl_tensor = torch.from_numpy(smpl_trans.astype(np.float32)).to(device)
        # left_hand_tensor = torch.zeros((num_frames, 45), dtype=torch.float32, device=device)
        # right_hand_tensor = torch.zeros((num_frames, 45), dtype=torch.float32, device=device)
        # jaw_tensor = torch.zeros((num_frames, 3), dtype=torch.float32, device=device)
        # eye_tensor = torch.zeros((num_frames, 3), dtype=torch.float32, device=device)
        # num_expression = getattr(self.smplx_model, "num_expression_coeffs", 10)
        # expression_tensor = torch.zeros((num_frames, num_expression), dtype=torch.float32, device=device)

        # with torch.no_grad():
        #     smplx_output = self.smplx_model(
        #         betas=betas_batched,
        #         global_orient=global_orient_tensor,
        #         body_pose=body_pose_tensor,
        #         transl=transl_tensor,
        #         expression=expression_tensor,
        #         left_hand_pose=left_hand_tensor,
        #         right_hand_pose=right_hand_tensor,
        #         jaw_pose=jaw_tensor,
        #         leye_pose=eye_tensor,
        #         reye_pose=eye_tensor,
        #         return_verts=False,
        #         return_full_pose=True,
        #     )

        # joints_all = smplx_output.joints.detach().cpu().numpy().astype(np.float64)
        # global_orient_all = smplx_output.global_orient.detach().cpu().numpy().astype(np.float64)
        # pelvis_trans = joints_all[:, 0].copy()

        # root_rotations_out = R.from_rotvec(global_orient_all)
        # smpl_quat_xyzw = root_rotations_out.as_quat()
        # root_rotations_mat = root_rotations_out.as_matrix()

        # pelvis_rot_mat = np.einsum("nij,jk->nik", root_rotations_mat, self._ROOT_TO_PELVIS_ROT)
        # pelvis_quat_xyzw = R.from_matrix(pelvis_rot_mat).as_quat()

        # joints_body = joints_all[:, :55, :].copy()
        # joints_centered = joints_body - pelvis_trans[:, np.newaxis, :]
        # pelvis_rot_mat_T = pelvis_rot_mat.transpose(0, 2, 1)
        # joints_local = np.einsum("nij,nkj->nki", pelvis_rot_mat_T, joints_centered)

        return {
            "betas": betas_array,
            "pose_body": pose_body,
            "pose_hand": pose_hand,
            "smpl_trans": smpl_trans,
            "smpl_quat_xyzw": root_quat_xyzw.astype(np.float64),
            "pelvis_trans": np.asarray(pelvis_trans_list, dtype=np.float64),
            "pelvis_quat_xyzw": np.asarray(pelvis_quat_xyzw_list, dtype=np.float64),
        }

    # ------------------------------------------------------------------
    # IK update / solve (mirrors GeneralMotionRetargeting)
    # ------------------------------------------------------------------
     # high priority: using scale and offset to compute target pose of every body of smplx human, and update the (IK)task targets.
    def update_targets(self, robot_data: Dict[str, BodyPose], offset_to_ground: bool = True) -> None:
        robot_data = self.to_numpy(robot_data) #ensure that all data is in NumPy array format

        robot_data = self.offset_robot_data(robot_data) 
        robot_data = self.scale_robot_data(robot_data) #notice: robot_data is a dictionary, key is the robot body name, value is the BodyPose which contains pos and quat(wxyz)
        #可能原因一，函数本身的计算过程有问题
        #可能原因二，g1_to_smplx.json的数值有问题
        #warning: 可能原因三，没有做归一化
        #g1_to_smplx.json 一部分offset精度不够

        #warning: 这个输入文件的pelvis是怎么来的？ 怎么和retarget的输出文件的数值对得上？


        check2_pelvis = robot_data['pelvis']
        # print(f"check2_pelvis: {check2_pelvis}")


        check2_left_shoulder_yaw_link = robot_data['left_shoulder_yaw_link']
        left_shoulder_pos = np.asarray(check2_left_shoulder_yaw_link.pos, dtype=np.float64).reshape(-1)
        left_shoulder_quat = np.asarray(check2_left_shoulder_yaw_link.rot, dtype=np.float64).reshape(-1)
        # print("check2_left_shoulder_yaw_link: " + ", ".join(f"{value:.15f}" for value in np.concatenate([left_shoulder_pos, left_shoulder_quat])))
        # 000005.npz last frame:  BodyPose(pos=array([-0.47435894,  0.26292525,  1.42337835]), rot=array([ 0.52543509, -0.26132338,  0.66067818,  0.4681158 ]))
        # [ -0.482392021088843, 0.259669719500133, 1.395207188492965], [0.525435089028519, -0.261323381639568, 0.660678180602081, 0.468115796681096]



        if not self._ground_offset_initialized: #self._ground_offset_initialized is false -> not self._ground_offset_initialized is true
            try:
                lowest_z = min(pose.pos[2] for pose in robot_data.values())
                desired = self.ground_height #ground_height is zero.
                self.set_ground_offset(max(0.0, desired - lowest_z)) #notice: lowest_z is usually negative(since smplx human is usually higher than g1 robot).
            except ValueError:
                print("Warning: Failed to compute constant ground offset.")
                self.set_ground_offset(0.0)
            self._ground_offset_initialized = True

        robot_data = self.apply_ground_offset(robot_data)

        # if offset_to_ground:
        #     robot_data = self.offset_robot_data_to_ground(robot_data)

        self.scaled_robot_data = robot_data

        if self.use_ik_match_table1:
            for smplx_body, task in self.smplx_body_to_task1.items():

                #smplx_body is smplx body name only exist in ik table1 or table2
                robot_body = self.robot_body_for_smplx.get(smplx_body) 
                pose = robot_data.get(robot_body)
                # pose = robot_data.get(smplx_body) #notice: self.offset_robot_data将robot_data中的机器人body name转换为smplx body name
                if pose is None:
                    print(f"During the update_targets() function, no pose found for the robot body: {robot_body}")
                    continue
                target_pos, target_rot = pose.pos, pose.rot
                task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3(target_rot), target_pos))

        if self.use_ik_match_table2:
            for smplx_body, task in self.smplx_body_to_task2.items():
                robot_body = self.robot_body_for_smplx.get(smplx_body)
                pose = robot_data.get(robot_body) 
                # pose = robot_data.get(smplx_body) 
                if pose is None:
                    print(f"During the update_targets() function, no pose found for the robot body: {robot_body}")
                    continue
                target_pos, target_rot = pose.pos, pose.rot
                task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3(target_rot), target_pos))

    def retarget(self, robot_data: Dict[str, BodyPose], offset_to_ground: bool = False) -> np.ndarray:
        self.update_targets(robot_data, offset_to_ground=offset_to_ground)

        if self.use_ik_match_table1:
            curr_error = self.error1()
            dt = self.configuration.model.opt.timestep
            vel1 = mink.solve_ik(self.configuration, self.tasks1, dt, self.solver, self.damping, self.ik_limits)
            self.configuration.integrate_inplace(vel1, dt)
            next_error = self.error1()
            num_iter = 0
            while curr_error - next_error > 0.001 and num_iter < self.max_iter:
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
            while curr_error - next_error > 0.001 and num_iter < self.max_iter:
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
        # Debug-only check: verify that BodyPose fields are numpy arrays; do not mutate/convert.
        # This mirrors motion_retarget.to_numpy semantics but keeps this as a no-op converter.
        try:
            for body_name, pose in body_poses.items():
                pos_is_np = isinstance(pose.pos, np.ndarray)
                rot_is_np = isinstance(pose.rot, np.ndarray)
                # print(
                #     f"[to_numpy] {body_name}: pos is np.ndarray={pos_is_np}, rot is np.ndarray={rot_is_np}"
                # )
        except Exception as e:
            print(f"[to_numpy] Warning: failed to inspect body poses: {e}")
        return body_poses
    
    # high priority: scale the robot data in the Global Coordinate System
    def scale_robot_data(self, robot_data: Dict[str, BodyPose]) -> Dict[str, BodyPose]:
        if not self.robot_scale_table:
            print("No robot scale table found. This may cause incorrect motion retargeting.")
            return robot_data

        root_pose = robot_data.get(self.robot_root_name)  #notice: robot_data is a dictionary, key is the robot body name, value is the BodyPose which contains pos and quat(wxyz)
        if root_pose is None:
            print("No robot root pose found. This may cause incorrect motion retargeting.")
            return robot_data

        scaled: Dict[str, BodyPose] = {} #mark: 这种定义dict的好处是value "BodyPose" 可以方便地对子键值对进行操作。eg: 就像下一行的root_pose.pos(root_pose is the BodyPose in robot_data)
        root_pos = root_pose.pos

        # print(f"In scale_robot_data(), root_pose: {root_pose}")


        for body_name, pose in robot_data.items():
            scale = self.robot_scale_table.get(body_name)
            # if body_name == "left_shoulder_yaw_link":
            #     # print(f"In scale_robot_data(), scale: {scale}")
            #     pass
            if scale is None or body_name == self.robot_root_name:
                scaled[body_name] = pose
                continue
            local = pose.pos - root_pos
            scaled_pos = local * scale + root_pos
            scaled[body_name] = BodyPose(pos=scaled_pos, rot=pose.rot)
            # if body_name == "left_shoulder_yaw_link":
            #     # print(f"In scale_robot_data(), scaled[left_shoulder_yaw_link]: {scaled[body_name]}")
            #     pass

        return scaled
 
    # high priority: offset the robot data according to the IK config (eg: g1_to_smplx.json)
    def offset_robot_data(self, robot_data: Dict[str, BodyPose]) -> Dict[str, BodyPose]:
        offset_data: Dict[str, BodyPose] = {}
        CHECK = True
        for smplx_body, robot_body in self.robot_body_for_smplx.items(): #notice： robot_body_for_smplx is dict, key is the smplx body name in ik table, value is the robot body name in ik table 
            pose = robot_data.get(robot_body)
            if pose is None:
                print(f"During offset_robot_data(), no pose found for robot body: {robot_body}")
                continue

            # if CHECK == True:
            #     # if robot_body == "left_hip_roll_link":
            #     #     pose.pos = np.array([-0.3933704 ,  0.27901529,  0.88634385])
            #     #     # pose.pos = np.array([-0.39319336,  0.27932851,  0.88629498])
            #     #     pose.rot = np.array([ 0.72764389,  0.04312718, -0.07358992,  0.68063128])
            #     #     # pose.rot = np.array([ 0.72755596,  0.04284811, -0.07307196,  0.68079868])
            #     #     before_check2_left_hip_roll_link = robot_data['left_hip_roll_link']
            #     #     print(f"before_check2_left_hip_roll_link: {before_check2_left_hip_roll_link}")
            #     if robot_body == "pelvis":
            #         pose.pos = np.array([-0.326287060976028, 0.291343450546265, 0.970250546932220])
            #         pose.rot = np.array([ 0.718291127715090, 0.024670563882039, -0.034901141980084, 0.694428635218921])
                    
            #     if robot_body == "left_shoulder_yaw_link":
            #         pose.pos = np.array([-0.444891021011150, 0.267278680737261, 1.293120111716746])
            #         pose.rot = np.array([0.838708736211145, 0.146224318973807, 0.095631307160299, 0.515791389453685])
                    
        
            
            pos, quat = pose.pos, pose.rot  # pos: position of body; quat: quaternion of body (wxyz format)
            offset_data[robot_body] = BodyPose(pos=pos, rot=quat)  # initialize with original values
            if robot_body == "pelvis":
                before_check2_pelvis = offset_data['pelvis']
                # print(f"before_check2_pelvis: {before_check2_pelvis}")
                pass
            if robot_body == "left_shoulder_yaw_link":
                before_check2_left_shoulder_yaw_link = offset_data['left_shoulder_yaw_link']
                # print(f"before_check2_left_shoulder_yaw_link: {before_check2_left_shoulder_yaw_link}")
                pass
            
            # apply rotation offset first
            # notice: quat is from robot_data, rot_offsets is from ik config (eg: g1_to_smplx.json)
            rot_offset = self.rot_offsets.get(smplx_body)
            if rot_offset is not None:
                updated_quat = (R.from_quat(quat, scalar_first=True) * rot_offset).as_quat(scalar_first=True)
            else:
                print(f"rot_offset is None for smplx body: {smplx_body}")
                continue
            offset_data[robot_body].rot = updated_quat
            
            # apply position offset
            local_offset = self.pos_offsets.get(smplx_body)  # notice: pos_offsets is from ik config (eg: g1_to_smplx.json)
            if local_offset is None:
                print(f"pos_offset is None for smplx body: {smplx_body}")
                continue
            # compute the global position offset using the updated rotation
            global_pos_offset = R.from_quat(updated_quat, scalar_first=True).apply(local_offset)  # R is a matrix, local_offset is a vector, global_pos_offset is a vector
            # 将全局位置偏移加到原始位置上
            offset_data[robot_body].pos = pos + global_pos_offset

        return offset_data

    # high priority: offset the robot data to the ground (before ik retargeting)
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

    # def compute_target_pose(self, smplx_joint: str, robot_pose: BodyPose) -> tuple[np.ndarray, np.ndarray]:
    #     pos_offset = self.pos_offsets.get(smplx_joint, np.zeros(3))
    #     rot_offset = self.rot_offsets.get(smplx_joint)

    #     robot_R = R.from_quat(robot_pose.rot, scalar_first=True)
    #     if rot_offset is not None:
    #         smplx_R = robot_R * rot_offset
    #     else:
    #         smplx_R = robot_R

    #     smplx_pos = smplx_R.apply(pos_offset) + robot_pose.pos
    #     smplx_rot = smplx_R.as_quat(scalar_first=True)
    #     return smplx_pos, smplx_rot

    def extract_smplx_frame(self) -> Dict[str, Dict[str, np.ndarray]]:
        frame: Dict[str, Dict[str, np.ndarray]] = {}
        for joint_name in self.smplx_joint_names: #notice: joint_name is from python lib "smplx.joint_names"
            mjcf_name = self.resolve_smplx_body_name(joint_name)
            #notice: body_name_to_id is constructed from smplx MJCF model of "smplx_humanoid.xml" file
            body_id = self.body_name_to_id.get(mjcf_name) #body_id is the id of the body in smplx MJCF model
            # print(f"joint_name: {joint_name}, mjcf_name: {mjcf_name}, body_id: {body_id}")
            if body_id is None:
                continue
            pos = self.configuration.data.xpos[body_id].copy()
            quat = self.configuration.data.xquat[body_id].copy() #warning: 无法验证 quat is wxyz format
            #xquat是绝对旋转（源码engine_core_smooth.c可以证明), 源码应该大概率已经证明是wxyz格式
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
            print(f"betas is None, return np.zeros(self.num_betas, dtype=np.float64)")
            return np.zeros(self.num_betas, dtype=np.float64)
        betas = np.asarray(betas, dtype=np.float64)
        if betas.shape[0] != self.num_betas:
            print(f"betas.shape[0] != self.num_betas, self.num_betas: {self.num_betas}")
            if betas.shape[0] < self.num_betas:
                betas = np.pad(betas, (0, self.num_betas - betas.shape[0]), mode="constant")
            else:
                betas = betas[: self.num_betas]
        return betas

    def compute_pelvis_joint_offset(self, betas_array: np.ndarray) -> np.ndarray:
        device = torch.device("cpu")
        try:
            device = next(self.smplx_model.parameters()).device
        except StopIteration:
            pass

        betas_tensor = torch.from_numpy(betas_array.astype(np.float32)).unsqueeze(0).to(device)
        zero_pose = torch.zeros((1, 63), dtype=torch.float32, device=device)
        zero_trans = torch.zeros((1, 3), dtype=torch.float32, device=device)
        zero_hand = torch.zeros((1, 45), dtype=torch.float32, device=device)
        num_expression = getattr(self.smplx_model, "num_expression_coeffs", 10)
        expression = torch.zeros((1, num_expression), dtype=torch.float32, device=device)
        zero_three = torch.zeros((1, 3), dtype=torch.float32, device=device)

        with torch.no_grad():
            output = self.smplx_model(
                betas=betas_tensor,
                body_pose=zero_pose,
                global_orient=zero_three,
                transl=zero_trans,
                left_hand_pose=zero_hand,
                right_hand_pose=zero_hand,
                expression=expression,
                jaw_pose=zero_three,
                leye_pose=zero_three,
                reye_pose=zero_three,
                return_verts=False,
                return_full_pose=False,
            )

        pelvis_offset = output.joints[0, 0].detach().cpu().numpy().astype(np.float64)
        #warning: 我还可以从这里面去计算pelvis_quat_offset去进行一些验证吗？
        return pelvis_offset

    # high priority: compute the local rotation of each body joint
    def compute_local_rotations(
        self,
        frame_joints: Dict[str, Dict[str, np.ndarray]],
    ) -> np.ndarray:
        body_pose = np.zeros(63, dtype=np.float64)
        global_rots: Dict[str, R] = {}
        #notice: frame_joints: {body_names: {"pos": pos, "rot": quat}} for one frame
        for joint_name, joint_data in frame_joints.items():
            if joint_name not in self.smplx_name_to_idx:
                print(f"joint_name: {joint_name} not in self.smplx_name_to_idx")
                continue
            rot_quat = joint_data["rot"] #wxyz  
            global_rots[joint_name] = R.from_quat(rot_quat[[1, 2, 3, 0]]) #xyzw format

        for i in range(1, min(22, len(self.smplx_joint_names))): #notice: self.smplx_joint_names is from python lib "smplx.joint_names"
            joint_name = self.smplx_joint_names[i]
            parent_idx = self.smplx_parents[i] 
            parent_name = self.smplx_joint_names[parent_idx]
            if joint_name not in global_rots or parent_name not in global_rots:
                print(f"joint_name: {joint_name} or parent_name: {parent_name} not in global_rots")
                continue
            parent_R = global_rots[parent_name] 
            joint_R = global_rots[joint_name]
            local_R = parent_R.inv() * joint_R 
            body_pose[(i - 1) * 3 : (i - 1) * 3 + 3] = local_R.as_rotvec()
            #notice: body_pose doesn't include pelvis!!!

        return body_pose


__all__ = ["RobotToSMPLXRetargeting"]

