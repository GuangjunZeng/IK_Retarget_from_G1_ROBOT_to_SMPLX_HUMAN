import copy
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R

from general_motion_retargeting.motion_retarget import GeneralMotionRetargeting
from general_motion_retargeting.reverse_motion_retarget import RobotToSMPLXRetargeting
from general_motion_retargeting.robot import BodyPose

# python scripts/validate_g1_to_smplx_roundtrip.py

def _random_unit_quat(rng: np.random.Generator) -> np.ndarray:
    vec = rng.normal(size=4)
    vec /= np.linalg.norm(vec)
    return vec


def _collect_human_names(retargeter: GeneralMotionRetargeting) -> Iterable[str]:
    names = {retargeter.human_root_name}
    names.update(retargeter.pos_offsets1.keys())
    names.update(retargeter.pos_offsets2.keys())
    return names


def _generate_dummy_human_data(retargeter: GeneralMotionRetargeting, seed: int = 0) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    data: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for name in _collect_human_names(retargeter):
        if name == retargeter.human_root_name:
            pos = rng.uniform(-0.1, 0.1, size=3)
            quat = np.array([1.0, 0.0, 0.0, 0.0])
        else:
            pos = rng.uniform(-0.5, 0.5, size=3)
            quat = _random_unit_quat(rng)
        data[name] = (pos.astype(np.float64), quat.astype(np.float64))
    return data


def _human_offset_to_robot_bodyposes(
    human_offset: Dict[str, Tuple[np.ndarray, np.ndarray]],
    gmr: GeneralMotionRetargeting,
) -> Dict[str, BodyPose]:
    robot_data: Dict[str, BodyPose] = {}
    root_pos, root_quat = human_offset[gmr.human_root_name]
    robot_data[gmr.robot_root_name] = BodyPose(pos=root_pos.copy(), rot=root_quat.copy())

    for mapping in (gmr.human_body_to_task1, gmr.human_body_to_task2):
        for human_name, task in mapping.items():
            if human_name not in human_offset:
                continue
            pos, quat = human_offset[human_name]
            robot_data[task.frame_name] = BodyPose(pos=pos.copy(), rot=quat.copy())
    return robot_data


def _invert_offset_robot_data(
    smplx_targets: Dict[str, BodyPose],
    reverse: RobotToSMPLXRetargeting,
) -> Dict[str, BodyPose]:
    robot_scaled: Dict[str, BodyPose] = {}
    for smplx_body, pose in smplx_targets.items():
        robot_body = reverse.robot_body_for_smplx.get(smplx_body)
        if smplx_body == reverse.smplx_root_name:
            robot_body = reverse.robot_root_name
        if robot_body is None:
            continue

        rot_offset = reverse.rot_offsets.get(smplx_body)
        updated_rot = R.from_quat(pose.rot, scalar_first=True)
        if rot_offset is not None:
            original_rot = (updated_rot * rot_offset.inv()).as_quat(scalar_first=True)
        else:
            original_rot = pose.rot.copy()

        pos_offset = reverse.pos_offsets.get(smplx_body, np.zeros(3))
        global_offset = updated_rot.apply(pos_offset)
        original_pos = pose.pos - global_offset
        robot_scaled[robot_body] = BodyPose(pos=original_pos, rot=original_rot)
    return robot_scaled


def _invert_scale_robot_data(
    scaled_robot: Dict[str, BodyPose],
    reverse: RobotToSMPLXRetargeting,
) -> Dict[str, BodyPose]:
    original: Dict[str, BodyPose] = {}
    root_pose = scaled_robot[reverse.robot_root_name]
    original[reverse.robot_root_name] = BodyPose(pos=root_pose.pos.copy(), rot=root_pose.rot.copy())
    root_pos = root_pose.pos

    for body_name, pose in scaled_robot.items():
        if body_name == reverse.robot_root_name:
            continue
        scale = reverse.robot_scale_table.get(body_name)
        if scale is None:
            original[body_name] = BodyPose(pos=pose.pos.copy(), rot=pose.rot.copy())
            continue
        local_scaled = pose.pos - root_pos
        local_original = local_scaled / scale
        original_pos = local_original + root_pos
        original[body_name] = BodyPose(pos=original_pos, rot=pose.rot.copy())
    return original


def _robot_bodyposes_to_human(
    robot_data: Dict[str, BodyPose],
    gmr: GeneralMotionRetargeting,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    human_data: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    root_pose = robot_data[gmr.robot_root_name]
    human_data[gmr.human_root_name] = (root_pose.pos.copy(), root_pose.rot.copy())

    robot_to_human: Dict[str, str] = {}
    for human_name, task in gmr.human_body_to_task1.items():
        robot_to_human[task.frame_name] = human_name
    for human_name, task in gmr.human_body_to_task2.items():
        robot_to_human.setdefault(task.frame_name, human_name)

    for robot_name, pose in robot_data.items():
        if robot_name == gmr.robot_root_name:
            continue
        human_name = robot_to_human.get(robot_name)
        if human_name is None:
            continue
        human_data[human_name] = (pose.pos.copy(), pose.rot.copy())
    return human_data


def _undo_offset_human_data(
    offset_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    pos_offsets: Dict[str, np.ndarray],
    rot_offsets: Dict[str, R],
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    recovered: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for body_name, (pos, quat) in offset_data.items():
        rot_offset = rot_offsets.get(body_name)
        if rot_offset is None:
            recovered[body_name] = (pos.copy(), quat.copy())
            continue
        updated_rot = R.from_quat(quat, scalar_first=True)
        original_rot = (updated_rot * rot_offset.inv()).as_quat(scalar_first=True)
        local_offset = pos_offsets.get(body_name, np.zeros(3))
        global_offset = updated_rot.apply(local_offset)
        original_pos = pos - global_offset
        recovered[body_name] = (original_pos, original_rot)
    return recovered


def _undo_scale_human_data(
    scaled_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    root_name: str,
    scale_table: Dict[str, float],
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    recovered: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    root_pos, root_quat = scaled_data[root_name]
    recovered[root_name] = (root_pos.copy(), root_quat.copy())
    for body_name, (pos, quat) in scaled_data.items():
        if body_name == root_name:
            continue
        scale = scale_table.get(body_name)
        if scale is None:
            recovered[body_name] = (pos.copy(), quat.copy())
            continue
        local_scaled = pos - root_pos
        local_original = local_scaled / scale
        original_pos = local_original + root_pos
        recovered[body_name] = (original_pos, quat.copy())
    return recovered


def _quat_max_error(q1: np.ndarray, q2: np.ndarray) -> float:
    direct = np.linalg.norm(q1 - q2)
    flipped = np.linalg.norm(q1 + q2)
    return float(min(direct, flipped))


def validate_roundtrip(seed: int = 0) -> None:
    base_dir = Path(__file__).resolve().parents[1]
    smplx_models = base_dir / "assets" / "body_models"

    gmr = GeneralMotionRetargeting(src_human="smplx", 
                        tgt_robot="unitree_g1", 
                        verbose=False)
    reverse = RobotToSMPLXRetargeting(
        robot_type="unitree_g1",
        smplx_model_path=smplx_models,
        gender="neutral",
        verbose=False,)

    original_human = _generate_dummy_human_data(gmr, seed)
    original_human_copy = copy.deepcopy(original_human)

    scaled_human = gmr.scale_human_data(copy.deepcopy(original_human_copy), gmr.human_root_name, gmr.human_scale_table)
    offset_human = gmr.offset_human_data(copy.deepcopy(scaled_human), gmr.pos_offsets1, gmr.rot_offsets1)

    offset_bodyposes = {
        body_name: BodyPose(pos=entry[0].copy(), rot=entry[1].copy()) for body_name, entry in offset_human.items()
    }

    robot_targets = _human_offset_to_robot_bodyposes(offset_human, gmr)

    robot_after_offset = reverse.offset_robot_data(reverse.scale_robot_data(robot_targets))

    max_pos_err_after_forward = 0.0
    max_rot_err_after_forward = 0.0
    for body_name, pose in offset_bodyposes.items():
        candidate = robot_after_offset.get(body_name)
        if candidate is None:
            continue
        max_pos_err_after_forward = max(max_pos_err_after_forward, float(np.linalg.norm(pose.pos - candidate.pos)))
        max_rot_err_after_forward = max(max_rot_err_after_forward, _quat_max_error(pose.rot, candidate.rot))

    robot_scaled = _invert_offset_robot_data(offset_bodyposes, reverse)
    robot_original = _invert_scale_robot_data(robot_scaled, reverse)

    recovered_human_offset = _robot_bodyposes_to_human(robot_original, gmr)
    recovered_human_scaled = _undo_offset_human_data(recovered_human_offset, gmr.pos_offsets1, gmr.rot_offsets1)
    recovered_human = _undo_scale_human_data(recovered_human_scaled, gmr.human_root_name, gmr.human_scale_table)

    max_pos_error = 0.0
    max_rot_error = 0.0
    for body_name, (orig_pos, orig_quat) in original_human.items():
        rec_pos, rec_quat = recovered_human.get(body_name, (None, None))
        if rec_pos is None:
            continue
        max_pos_error = max(max_pos_error, float(np.linalg.norm(orig_pos - rec_pos)))
        max_rot_error = max(max_rot_error, _quat_max_error(orig_quat, rec_quat))

    print("Forward check (scale+offset -> reverse scale+offset):")
    print(f"  max position error: {max_pos_err_after_forward:.6e}")
    print(f"  max quaternion error: {max_rot_err_after_forward:.6e}")
    print("Roundtrip (human -> robot -> human):")
    print(f"  max position error: {max_pos_error:.6e}")
    print(f"  max quaternion error: {max_rot_error:.6e}")


if __name__ == "__main__":
    validate_roundtrip()
