
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pickle


# def _load_pickle_motion(path: Path) -> Tuple[Dict[str, Any], float, np.ndarray, np.ndarray, np.ndarray, Any, Any]:
#     with path.open("rb") as f:
#         motion_data = pickle.load(f)

#     motion_fps = float(motion_data["fps"])
#     motion_root_pos = np.asarray(motion_data["root_pos"], dtype=np.float64)
#     motion_root_rot_xyzw = np.asarray(motion_data["root_rot"], dtype=np.float64)
#     motion_root_rot = motion_root_rot_xyzw[:, [3, 0, 1, 2]]  # xyzw -> wxyz #?
#     motion_dof_pos = np.asarray(motion_data["dof_pos"], dtype=np.float64)
#     motion_local_body_pos = motion_data.get("local_body_pos")
#     motion_link_body_list = motion_data.get("link_body_list")

#     return (
#         motion_data,
#         motion_fps,
#         motion_root_pos,
#         motion_root_rot,
#         motion_dof_pos,
#         motion_local_body_pos,
#         motion_link_body_list,
#     )


def _load_npz_motion(path: Path) -> Tuple[Dict[str, Any], float, np.ndarray, np.ndarray, np.ndarray, Any, Any]:
    with np.load(path, allow_pickle=True) as data:
        qpos = None
        if "qpos" in data:
            pass
            # qpos = np.asarray(data["qpos"], dtype=np.float64)
        elif "motion" in data:
            pass
            # qpos = np.asarray(data["motion"], dtype=np.float64)
        elif "full_data" in data: #notice: fulldata: 3+4+29=34 columns
            full_data = np.asarray(data["full_data"], dtype=np.float64)
            if full_data.ndim != 2 or full_data.shape[1] < 7:
                raise ValueError(
                    f"Unexpected full_data shape {full_data.shape} in {path}; expected (frames, 7 + dofs)"
                )
            qpos = full_data  #qpos: fulldata: 3+4+29=34 columns
        else:
            root_pos = data.get("root_pos")
            root_quat = data.get("root_quat") # or data.get("root_rot")
            joints = data.get("joints") #or data.get("dof_pos") #notice: joints: 29 columns。 但是加上手部关节之后呢?
            if root_pos is None or root_quat is None or joints is None:
                raise KeyError(
                    f"NPZ motion file {path} must contain either 'qpos'/'motion', 'full_data', or the trio ('root_pos', 'root_quat', 'joints')"
                )
            root_pos = np.asarray(root_pos, dtype=np.float64)
            root_quat = np.asarray(root_quat, dtype=np.float64)
            joints = np.asarray(joints, dtype=np.float64)
            if root_pos.ndim != 2 or root_pos.shape[1] != 3:
                raise ValueError(f"root_pos shape {root_pos.shape} in {path} is invalid; expected (frames, 3)")
            if root_quat.ndim != 2 or root_quat.shape[1] != 4:
                raise ValueError(f"root_quat shape {root_quat.shape} in {path} is invalid; expected (frames, 4)")
            if joints.ndim != 2: 
                raise ValueError(f"joints shape {joints.shape} in {path} is invalid; expected 2-D array")
            if not (root_pos.shape[0] == root_quat.shape[0] == joints.shape[0]):
                raise ValueError(
                    f"Frame count mismatch in {path}: root_pos {root_pos.shape[0]}, root_quat {root_quat.shape[0]}, joints {joints.shape[0]}"
                )
            qpos = np.concatenate([root_pos, root_quat, joints], axis=1) #这个参量没问题

        if qpos.ndim != 2 or qpos.shape[1] < 7:
            raise ValueError(
                f"Unexpected qpos shape {qpos.shape} in {path}; expected (frames, 7 + dofs)"
            )

        root_pos = qpos[:, :3]
        root_rot_xyzw = qpos[:, 3:7] #notice: qpos is xyzw format （from input robot npz file）
        #notice：root_rot is wxyz format
        root_rot = np.concatenate([root_rot_xyzw[:, 3:4], root_rot_xyzw[:, :3]], axis=1) # wxyz format
        dof_pos = qpos[:, 7:]

        fps = None
        for key in ("fps", "frame_rate", "framerate"):
            if key in data:
                fps_array = np.asarray(data[key])
                fps = float(fps_array.item() if fps_array.shape == () else fps_array.flatten()[0])
                break
        if fps is None:
            print("There is no fps info in the input npz file, use default fps 30.0!")
            fps = 30.0

        motion_data: Dict[str, Any] = {
            "fps": fps,
            "root_pos": root_pos,
            "root_rot_xyzw": root_rot_xyzw,
            "root_rot": root_rot,
            "dof_pos": dof_pos,
            "qpos": qpos, #notice: root_pos, root_rot, and dof_pos
        }

        if "local_body_pos" in data:
            print("There is local_body_pos in the input robot motion npz file")
            motion_data["local_body_pos"] = data["local_body_pos"]
        if "link_body_list" in data:
            print("There is link_body_list in the input robot motion npz file")
            link_array = data["link_body_list"]
            motion_data["link_body_list"] = (
                link_array.tolist() if isinstance(link_array, np.ndarray) else link_array
            )

        motion_local_body_pos = motion_data.get("local_body_pos")
        motion_link_body_list = motion_data.get("link_body_list")

    return (
        motion_data,
        fps,
        root_pos,
        root_rot,
        dof_pos,
        motion_local_body_pos,
        motion_link_body_list,
    )


def load_robot_motion(motion_file: str):
    """Load robot motion data from either pickle or NPZ format."""

    path = Path(motion_file)
    if not path.exists():
        raise FileNotFoundError(f"Motion file not found: {motion_file}")

    suffix = path.suffix.lower()
    if suffix == ".pkl":
        print("input robot motion file is pkl file")
        return _load_pickle_motion(path)
    if suffix == ".npz":
        print("input robot motion file is npz file")
        return _load_npz_motion(path)

    raise ValueError(f"Unsupported motion file format '{suffix}' for {motion_file}")


