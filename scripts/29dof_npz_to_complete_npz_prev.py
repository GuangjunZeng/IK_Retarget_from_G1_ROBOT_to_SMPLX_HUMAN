#warning ： 代码暂时有问题

from __future__ import annotations

import argparse
import pathlib
import sys
import xml.etree.ElementTree as ET
from typing import Dict, Iterable, List, Tuple

import numpy as np

# Example usage:
# python scripts/29dof_npz_to_complete_npz.py \
#   --input /home/retarget/workbench/server3_data/locomotion/robot/ik_based/npz/000001.npz \
#   --output-dir /home/retarget/workbench/server3_data/locomotion/human/ik_based/npz

DEFAULT_GENDER = "female"
DEFAULT_FPS = 30.0
DEFAULT_BETAS = np.array(
    [
        0.63490343,
        0.22382046,
        -1.02493083,
        0.44071582,
        -0.99539453,
        -2.14731956,
        1.5268985,
        -0.18637267,
        2.42483139,
        1.88858294,
    ],
    dtype=np.float64,
)

HERE = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.append(str(HERE))

G1_MJCF_PATH = (HERE / ".." / "assets" / "unitree_g1" / "g1_mocap_29dof.xml").resolve()


def _as_numpy(
    array: np.ndarray,
    dtype: np.dtype | type | None = np.float64,
    target_shape: Tuple[int, ...] | None = None,
) -> np.ndarray:
    if array is None:
        raise ValueError("Encountered None while attempting to read NPZ data.")
    result = np.asarray(array)
    if dtype is not None:
        result = result.astype(dtype, copy=False)
    if target_shape is not None and result.shape != target_shape:
        raise ValueError(f"Expected shape {target_shape}, got {result.shape}")
    return result


def parse_betas(arg_betas):
    if arg_betas is None:
        return DEFAULT_BETAS.astype(np.float64, copy=True)
    betas_array = np.asarray(arg_betas, dtype=np.float64)
    if betas_array.size == 0:
        return DEFAULT_BETAS.astype(np.float64, copy=True)
    return betas_array


def load_joint_axes(mjcf_path: pathlib.Path) -> Dict[str, np.ndarray]:
    if not mjcf_path.exists():
        raise FileNotFoundError(f"g1 MJCF model not found: {mjcf_path}")

    tree = ET.parse(str(mjcf_path))
    root = tree.getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("Could not find <worldbody> in MJCF file.")

    joint_axes: Dict[str, np.ndarray] = {}

    def recurse(body_elem: ET.Element, parent_name: str = "") -> None:
        for joint_elem in body_elem.findall("joint"):
            name = joint_elem.get("name")
            if not name:
                continue
            axis_str = joint_elem.get("axis", "0 0 1")
            axis = np.fromstring(axis_str, sep=" ", dtype=np.float64)
            if axis.shape != (3,):
                axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            norm = np.linalg.norm(axis)
            if norm == 0:
                axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            else:
                axis = axis / norm
            joint_axes[name] = axis
        for child_body in body_elem.findall("body"):
            recurse(child_body, child_body.get("name", parent_name))

    recurse(worldbody)
    return joint_axes


def load_29dof_npz(npz_path: pathlib.Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
    if not npz_path.exists():
        raise FileNotFoundError(f"Input NPZ file not found: {npz_path}")

    data = np.load(npz_path, allow_pickle=True)

    joint_names = data.get("joint_names")
    if joint_names is None:
        raise ValueError("Input NPZ missing 'joint_names'.")
    joint_names = np.asarray(list(joint_names))

    if "joints" in data.files:
        joints = _as_numpy(data["joints"], dtype=np.float64)
    elif "full_data" in data.files:
        joints = _as_numpy(data["full_data"][:, -len(joint_names) :], dtype=np.float64)
    else:
        raise ValueError("Input NPZ must provide 'joints' or 'full_data'.")

    if "root_pos" in data.files:
        root_pos = _as_numpy(data["root_pos"], dtype=np.float64)
    elif "full_data" in data.files:
        root_pos = _as_numpy(data["full_data"][:, :3], dtype=np.float64)
    else:
        raise ValueError("Input NPZ missing root position information.")

    if "root_quat" in data.files:
        root_quat = _as_numpy(data["root_quat"], dtype=np.float64)
    elif "full_data" in data.files:
        root_quat = _as_numpy(data["full_data"][:, 3:7], dtype=np.float64)
    else:
        raise ValueError("Input NPZ missing root quaternion information.")

    include_base = bool(np.array(data.get("include_base", True)).item())

    return root_pos, root_quat, joints, joint_names.astype("U"), include_base


def convert_single_npz(
    npz_path: pathlib.Path,
    output_path: pathlib.Path,
    gender: str,
    betas: np.ndarray,
    fps: float,
) -> Tuple[int, float]:
    joint_axis_map = load_joint_axes(G1_MJCF_PATH)
    root_pos, root_quat_xyzw, joints, joint_names, include_base = load_29dof_npz(npz_path)

    num_frames = root_pos.shape[0]
    motion_fps = fps if not np.isnan(fps) else DEFAULT_FPS

    # Normalize quaternions to avoid numerical drift
    quat_norm = np.linalg.norm(root_quat_xyzw, axis=1, keepdims=True)
    quat_norm[quat_norm == 0] = 1.0
    root_quat_xyzw = root_quat_xyzw / quat_norm

    # Convert joint angles into axis-angle vectors using joint axes
    joint_axis_angles = np.zeros((num_frames, joints.shape[1], 3), dtype=np.float64)
    for joint_idx, joint_name in enumerate(joint_names):
        axis = joint_axis_map.get(joint_name)
        if axis is None:
            axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        joint_axis_angles[:, joint_idx, :] = joints[:, joint_idx][:, None] * axis[None, :]

    pose_body = np.zeros((num_frames, 63), dtype=np.float32)
    pose_hand = np.zeros((num_frames, 90), dtype=np.float32)
    joints_local = np.zeros((num_frames, 55, 3), dtype=np.float64)

    num_pose_joints = min(joint_axis_angles.shape[1], 21)
    pose_body[:, : num_pose_joints * 3] = joint_axis_angles[:, :num_pose_joints, :].reshape(num_frames, -1)

    num_joint_local = min(joint_axis_angles.shape[1], 55)
    joints_local[:, :num_joint_local, :] = joint_axis_angles[:, :num_joint_local, :]

    out = {
        "gender": np.array(gender),
        "betas": betas.astype(np.float32),
        "pose_body": pose_body,
        "pose_hand": pose_hand,
        "smpl_trans": root_pos.astype(np.float32),
        "smpl_quat_xyzw": root_quat_xyzw.astype(np.float32),
        "pelvis_trans": root_pos.astype(np.float32),
        "pelvis_quat_xyzw": root_quat_xyzw.astype(np.float64),
        "joints_local": joints_local,
        "fps": np.array(int(round(motion_fps)), dtype=np.int64),
    }

    np.savez(output_path, **out)

    return num_frames, motion_fps


def default_output_path(input_path: pathlib.Path, output_dir: pathlib.Path) -> pathlib.Path:
    stem = input_path.stem
    return output_dir / f"{stem}.npz"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repackage a 29-DoF g1 robot NPZ into SMPL-X style NPZ."
    )

    parser.add_argument("--input", required=True, help="Path to the 29-DoF robot NPZ file")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional explicit output NPZ path. Overrides --output-dir if provided.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to place the converted NPZ (defaults to server3_data/locomotion/human/ik_based/npz)",
    )
    parser.add_argument(
        "--gender",
        choices=["neutral", "female", "male"],
        default=DEFAULT_GENDER,
        help="Gender string to store in the output NPZ",
    )
    parser.add_argument(
        "--betas",
        type=float,
        nargs="*",
        default=None,
        help="Optional SMPL-X betas; defaults to a template if omitted",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=DEFAULT_FPS,
        help="FPS to record in the output NPZ (default: 30)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=True,
        help="Overwrite the output file if it already exists (default: True)",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    input_path = pathlib.Path(args.input).expanduser().resolve()
    if args.output:
        output_path = pathlib.Path(args.output).expanduser().resolve()
        output_dir = output_path.parent
    else:
        if args.output_dir:
            output_dir = pathlib.Path(args.output_dir).expanduser().resolve()
        else:
            output_dir = (
                PROJECT_ROOT / "server3_data" / "locomotion" / "human" / "ik_based" / "npz"
            ).resolve()
        output_path = default_output_path(input_path, output_dir)

    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output file {output_path} already exists. Use --overwrite to replace it."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    betas = parse_betas(args.betas)

    frame_count, motion_fps = convert_single_npz(
        npz_path=input_path,
        output_path=output_path,
        gender=args.gender,
        betas=betas,
        fps=args.fps,
    )

    print(
        f"✅ Converted {input_path.name}: frames={frame_count}, fps={motion_fps:.2f} -> {output_path}"
    )


if __name__ == "__main__":
    main()


