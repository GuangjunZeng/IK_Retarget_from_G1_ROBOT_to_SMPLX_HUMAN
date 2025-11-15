import argparse
import csv
import json
import math
import os
import pathlib
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from queue import Queue  # noqa: F401 (reserved for future extensions matching smplx_to_robot_batch style)

import numpy as np
import smplx
import torch  # noqa: F401 (kept for consistency with smplx_to_robot_batch imports)
from rich import print
from scipy.spatial.transform import Rotation as R
from smplx.joint_names import JOINT_NAMES

from general_motion_retargeting import (  # noqa: E402
    RobotMotionViewer,  # noqa: F401 (placeholder for future visualization parity)
    load_robot_motion,
)
from general_motion_retargeting.params import (  # noqa: E402
    IK_CONFIG_DICT,
    ROBOT_XML_DICT,
)


# 可选的 CSV 行范围（由命令行参数设置）
START_ROW = 0     # 0-based inclusive（默认第一行数据，已跳过header）
END_ROW = None    # 0-based exclusive（默认处理到末尾）


# ===== 默认 SMPL-X 体型参数（来自 reference/010220.npz） =====
DEFAULT_BETAS = np.array(
    [
        0.63490343,
        0.22382046,
        -1.0249308,
        0.44071582,
        -0.9953945,
        -2.1473196,
        1.5268985,
        -0.18637267,
        2.4248314,
        1.888583,
    ],
    dtype=np.float32,
)


# ===== CPU 限制（用于全局限制 CPU 占用比例） =====
def cap_cpu_affinity_by_percent(percent):
    try:
        total_visible = None
        if hasattr(os, "sched_getaffinity"):
            current_affinity = os.sched_getaffinity(0)
            total_visible = len(current_affinity)
            allowed = max(1, int(math.floor(total_visible * (percent / 100.0))))
            print("允许使用的 cpu 线程数: ", allowed)
            cpus_sorted = sorted(current_affinity)
            target_set = set(cpus_sorted[:allowed])
            os.sched_setaffinity(0, target_set)
            return allowed, total_visible
        total_cpus = os.cpu_count() or 1
        allowed = max(1, int(math.floor(total_cpus * (percent / 100.0))))
        return allowed, total_cpus
    except Exception:
        total_cpus = os.cpu_count() or 1
        allowed = max(1, int(math.floor(total_cpus * (percent / 100.0))))
        return allowed, total_cpus

#? do need the downsample since the fps of robot motion is suitable?

# ===== 进度跟踪 =====
class ProgressTracker:
    def __init__(self, total_files):
        self.total_files = total_files
        self.completed = 0
        self.successful = 0
        self.failed = 0
        self.lock = threading.Lock()
        self.start_time = time.time()

    def update(self, success=True):
        with self.lock:
            self.completed += 1
            if success:
                self.successful += 1
            else:
                self.failed += 1

    def get_summary(self):
        with self.lock:
            total_time = time.time() - self.start_time
            return {
                "total_files": self.total_files,
                "completed": self.completed,
                "successful": self.successful,
                "failed": self.failed,
                "total_time": total_time,
            }


# mark: motion_retarget中的类和函数
class RobotToSMPLXBatchConverter:
    """Batch-friendly robot motion -> SMPL-X converter."""

    def __init__(self, robot_type, smplx_model_path, gender="neutral"):
        self.robot_type = robot_type
        self.gender = gender
        self.smplx_model_path = pathlib.Path(smplx_model_path)

        robot_xml_path = ROBOT_XML_DICT[robot_type]
        config_root = pathlib.Path(__file__).parent.parent / "general_motion_retargeting" / "ik_configs"
        reverse_config_path = config_root / f"{robot_type}_to_smplx.json"
        if reverse_config_path.exists():
            ik_config_path = reverse_config_path
        else:
            ik_config_path = IK_CONFIG_DICT["smplx"][robot_type]

        print(f"📦 加载机器人模型: {robot_xml_path}")
        import mujoco as mj  # deferred import to keep top-level fast in forked workers

        self.robot_model = mj.MjModel.from_xml_path(str(robot_xml_path))
        self.robot_data = mj.MjData(self.robot_model)

        print(f"📦 加载 IK 配置: {ik_config_path}")
        with open(ik_config_path, "r") as f:
            self.ik_config = json.load(f)

        print(f"📦 加载 SMPL-X 模型: {self.smplx_model_path}")
        self.smplx_model = smplx.create(
            model_path=str(self.smplx_model_path),
            model_type="smplx",
            gender=gender,
            use_pca=False,
        )

        self.num_betas = getattr(self.smplx_model, "num_betas", None)
        if self.num_betas is None:
            betas_attr = getattr(self.smplx_model, "betas", None)
            if betas_attr is not None and hasattr(betas_attr, "shape") and betas_attr.shape[-1] > 0:
                self.num_betas = int(betas_attr.shape[-1])
        if self.num_betas is None:
            self.num_betas = 10

        self.smplx_joint_names = JOINT_NAMES[: len(self.smplx_model.parents)]
        self.smplx_name_to_idx = {name: i for i, name in enumerate(self.smplx_joint_names)}
        self.smplx_parents = self.smplx_model.parents.detach().cpu().numpy().astype(int)

        self.smplx_to_robot_map = self._build_reverse_mapping()

        print(
            f"✅ 反向转换器完成初始化 -> robot={robot_type}, joint_mappings={len(self.smplx_to_robot_map)}"
        )

    def _build_reverse_mapping(self):
        """构建 SMPL-X 关节到机器人 body 的映射表。"""

        def _collect(table):
            mapping = {}
            for key, entry in table.items():
                if not entry:
                    continue
                if key in JOINT_NAMES:
                    smplx_joint_name = key
                    robot_body_name = entry[0]
                    pos_offset = entry[3]
                    rot_offset = entry[4]
                else:
                    robot_body_name = key
                    smplx_joint_name = entry[0]
                    pos_offset = entry[3]
                    rot_offset = entry[4]

                if smplx_joint_name not in JOINT_NAMES:
                    continue

                pos_offset = np.array(pos_offset, dtype=np.float64)
                rot_offset = np.array(rot_offset, dtype=np.float64)

                mapping.setdefault(smplx_joint_name, {
                    "robot_body": robot_body_name,
                    "pos_offset": pos_offset,
                    "rot_offset": rot_offset,
                })
            return mapping

        mapping = {}
        table1 = self.ik_config.get("ik_match_table1", {})
        table2 = self.ik_config.get("ik_match_table2", {})
        mapping.update(_collect(table1))
        for joint_name, info in _collect(table2).items():
            mapping.setdefault(joint_name, info)
        return mapping

    def _get_robot_body_pose(self, qpos):
        import mujoco as mj

        self.robot_data.qpos[:] = qpos
        mj.mj_forward(self.robot_model, self.robot_data)

        body_poses = {}
        for i in range(self.robot_model.nbody):
            body_name = mj.mj_id2name(self.robot_model, mj.mjtObj.mjOBJ_BODY, i)
            if not body_name:
                continue
            pos = self.robot_data.xpos[i].copy()
            quat = self.robot_data.xquat[i].copy()  # wxyz
            body_poses[body_name] = {"pos": pos, "rot": quat}
        return body_poses

    def _apply_reverse_offset(self, robot_pos, robot_rot, pos_offset, rot_offset):
        robot_R = R.from_quat(robot_rot[[1, 2, 3, 0]])  # wxyz -> xyzw
        offset_R = R.from_quat(rot_offset[[1, 2, 3, 0]])
        smplx_R = robot_R * offset_R.inv()
        smplx_pos = robot_pos - robot_R.apply(pos_offset)
        smplx_rot = smplx_R.as_quat(scalar_first=True)
        return smplx_pos, smplx_rot

    def robot_frame_to_smplx_joints(self, qpos):
        body_poses = self._get_robot_body_pose(qpos)
        smplx_joints = {}
        for smplx_joint_name, mapping in self.smplx_to_robot_map.items():
            robot_body_name = mapping["robot_body"]
            if robot_body_name not in body_poses:
                continue
            pos_offset = mapping["pos_offset"]
            rot_offset = mapping["rot_offset"]
            robot_pose = body_poses[robot_body_name]
            smplx_pos, smplx_rot = self._apply_reverse_offset(
                robot_pose["pos"],
                robot_pose["rot"],
                pos_offset,
                rot_offset,
            )
            smplx_joints[smplx_joint_name] = {
                "pos": smplx_pos,
                "rot": smplx_rot,
            }
        return smplx_joints

    def _compute_local_rotations(self, frame_joints):
        body_pose = np.zeros(63, dtype=np.float64)
        global_rots = {}
        for joint_name, joint_data in frame_joints.items():
            if joint_name not in self.smplx_name_to_idx:
                continue
            rot_quat = np.asarray(joint_data["rot"], dtype=np.float64)
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
            body_pose[(i - 1) * 3: (i - 1) * 3 + 3] = local_R.as_rotvec()

        return body_pose

    def _joints_to_smplx_params(self, smplx_joints_list, betas=None):
        num_frames = len(smplx_joints_list)
        if betas is None:
            betas = DEFAULT_BETAS.astype(np.float64, copy=True)
        else:
            betas = np.asarray(betas, dtype=np.float64)
            if betas.shape[0] != self.num_betas:
                print(
                    f"⚠️ betas length {betas.shape[0]} != expected {self.num_betas}, "
                    "resizing with zero padding/truncation"
                )
                if betas.shape[0] < self.num_betas:
                    betas = np.pad(betas, (0, self.num_betas - betas.shape[0]), mode="constant")
                else:
                    betas = betas[: self.num_betas]

        root_orient_list = []
        trans_list = []
        body_pose_list = []

        for frame_joints in smplx_joints_list:
            if "pelvis" in frame_joints:
                pelvis_pos = frame_joints["pelvis"]["pos"]
                pelvis_rot = frame_joints["pelvis"]["rot"]
                trans_list.append(pelvis_pos)
                pelvis_R = R.from_quat(pelvis_rot[[1, 2, 3, 0]])
                root_orient_list.append(pelvis_R.as_rotvec())
            else:
                trans_list.append(np.zeros(3, dtype=np.float64))
                root_orient_list.append(np.zeros(3, dtype=np.float64))

            body_pose = self._compute_local_rotations(frame_joints)
            body_pose_list.append(body_pose)

        return {
            "betas": betas,
            "root_orient": np.asarray(root_orient_list, dtype=np.float64),
            "trans": np.asarray(trans_list, dtype=np.float64),
            "pose_body": np.asarray(body_pose_list, dtype=np.float64),
        }

    def convert_robot_motion(
        self,
        pkl_path,
        output_npz_path,
        betas=None,
        fps_override=None,
        show_progress=False,
    ):
        robot_data, motion_fps, root_pos, root_rot, dof_pos, _, _ = load_robot_motion(pkl_path)
        num_frames = len(root_pos)

        if fps_override is not None:
            motion_fps = fps_override

        iterator = range(num_frames)
        if show_progress:
            from tqdm import tqdm

            iterator = tqdm(iterator, desc="Processing frames", leave=False)

        smplx_joints_list = []
        for i in iterator:
            qpos = np.concatenate([root_pos[i], root_rot[i], dof_pos[i]])
            smplx_joints = self.robot_frame_to_smplx_joints(qpos)
            smplx_joints_list.append(smplx_joints)

        smplx_params = self._joints_to_smplx_params(smplx_joints_list, betas=betas)

        np.savez(
            output_npz_path,
            betas=smplx_params["betas"],
            pose_body=smplx_params["pose_body"],
            root_orient=smplx_params["root_orient"],
            trans=smplx_params["trans"],
            gender=np.array(self.gender),
            mocap_frame_rate=np.array(motion_fps),
        )

        return {
            "frames": num_frames,
            "fps": motion_fps,
            "output_path": output_npz_path,
        }


# ===== 调试信息打印（预留，与原脚本风格一致） =====
def print_smplx_info(smplx_params):
    pass


# ====== 全局转换器缓存（每个进程独立） ======
CONVERTER_CACHE = {}


def get_converter(robot, smplx_folder, gender):
    key = (robot, str(smplx_folder), gender)
    if key not in CONVERTER_CACHE:
        CONVERTER_CACHE[key] = RobotToSMPLXBatchConverter(
            robot_type=robot,
            smplx_model_path=smplx_folder,
            gender=gender,
        )
    return CONVERTER_CACHE[key]


def process_single_robot_file(
    pkl_path,
    output_path,
    robot,
    smplx_folder,
    gender,
    betas,
    fps_override,
    show_progress=False,
):
    converter = get_converter(robot, smplx_folder, gender)
    return converter.convert_robot_motion(
        pkl_path=pkl_path,
        output_npz_path=output_path,
        betas=betas,
        fps_override=fps_override,
        show_progress=show_progress,
    )


def process_single_file_worker(args):
    (
        index,
        pkl_path,
        output_path,
        robot,
        smplx_folder,
        gender,
        betas,
        fps_override,
        show_progress,
    ) = args

    try:
        if not os.path.exists(pkl_path):
            return False, index, "the input file does not exist", {}
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if os.path.exists(output_path):
            return True, index, "the output file already exists, skipping processing", {}

        betas_array = np.array(betas, dtype=np.float64) if betas is not None else None
        result = process_single_robot_file(
            pkl_path=pkl_path,
            output_path=output_path,
            robot=robot,
            smplx_folder=smplx_folder,
            gender=gender,
            betas=betas_array,
            fps_override=fps_override,
            show_progress=show_progress,
        )
        print_smplx_info(result)
        return True, index, "processing completed", result
    except Exception as e:
        return False, index, str(e), {}


def resolve_motion_path(path_str, base_paths):
    path_candidate = pathlib.Path(path_str)
    if path_candidate.is_absolute():
        return str(path_candidate)
    for base in base_paths:
        candidate = pathlib.Path(base) / path_str
        if candidate.exists():
            return str(candidate)
    return str(pathlib.Path(base_paths[0]) / path_str)


def parse_csv_for_robot_paths(csv_file, base_paths):
    global START_ROW, END_ROW
    file_entries = []
    with open(csv_file, "r") as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader, None)
        if header is None:
            return file_entries

        header_lower = [h.lower() for h in header]
        try:
            index_col = header_lower.index("index")
        except ValueError:
            index_col = 0

        source_col = None
        fps_col = None
        for idx, name in enumerate(header_lower):
            if source_col is None and any(key in name for key in ["source_path", "pkl", "robot_path", "motion_path"]):
                source_col = idx
            if fps_col is None and any(key in name for key in ["fps", "frame_rate", "frame rate"]):
                fps_col = idx

        if source_col is None:
            raise ValueError("CSV header must contain a source path column (e.g., source_path/pkl_path)")

        for row_idx, row in enumerate(reader):
            if START_ROW is not None and row_idx < START_ROW:
                continue
            if END_ROW is not None and row_idx >= END_ROW:
                break

            if len(row) <= source_col:
                continue

            index_value = row[index_col].strip() if len(row) > index_col else f"{row_idx:05d}"
            source_value = row[source_col].strip()
            if not source_value:
                continue
            fps_value = None
            if fps_col is not None and len(row) > fps_col:
                try:
                    fps_value = float(row[fps_col]) if row[fps_col].strip() else None
                except ValueError:
                    fps_value = None

            motion_path = resolve_motion_path(source_value, base_paths)
            file_entries.append(
                {
                    "index": index_value,
                    "pkl_path": motion_path,
                    "fps": fps_value,
                    "raw_source": source_value,
                }
            )
    return file_entries


def process_batch_from_csv(
    csv_file,
    batch_save_path,
    robot,
    smplx_folder,
    gender,
    betas,
    fps_override,
    use_multithreading=True,
    num_threads=None,
    base_data_path=None,
    show_progress=False,
):
    base_paths = []
    if base_data_path:
        base_paths.append(pathlib.Path(base_data_path))
    here = pathlib.Path(__file__).parent
    default_candidate = here.parent.parent / "server3_data"
    if default_candidate.exists():
        base_paths.append(default_candidate)
    base_paths.append(here.parent.parent / "data")
    base_paths.append(pathlib.Path("."))

    file_pairs = parse_csv_for_robot_paths(csv_file, base_paths)
    print(f"处理 {len(file_pairs)} 个文件")

    os.makedirs(batch_save_path, exist_ok=True)
    progress_tracker = ProgressTracker(len(file_pairs))

    betas_serializable = betas.tolist() if isinstance(betas, np.ndarray) else (
        list(betas) if betas is not None else None
    )

    if use_multithreading:
        cpu_limit_percent = 60
        allowed_cores, _ = cap_cpu_affinity_by_percent(cpu_limit_percent)
        user_workers = num_threads if num_threads is not None else allowed_cores
        max_threads = max(1, min(user_workers, allowed_cores))
        print("实际真正允许的最大进程数是: ", max_threads)

        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["NUMEXPR_NUM_THREADS"] = "1"

        tasks = []
        for entry in file_pairs:
            output_path = os.path.join(batch_save_path, f"{entry['index']}.npz")
            task_args = (
                entry["index"],
                entry["pkl_path"],
                output_path,
                robot,
                str(smplx_folder),
                gender,
                betas_serializable,
                fps_override if fps_override is not None else entry["fps"],
                show_progress,
            )
            tasks.append(task_args)

        index_to_path = {entry["index"]: entry["pkl_path"] for entry in file_pairs}
        failure_counts = {}
        sample_failures = []

        with ProcessPoolExecutor(max_workers=max_threads) as executor:
            futures = [executor.submit(process_single_file_worker, task) for task in tasks]
            for future in as_completed(futures):
                try:
                    success, index, msg, details = future.result()
                    progress_tracker.update(success=success)
                    frames_info = f", frames={details.get('frames', 'n/a')}" if details else ""
                    print(
                        f"[{'ok' if success else 'failed'}] index={index} path={index_to_path.get(index, '')} msg={msg}{frames_info}"
                    )
                    if not success:
                        failure_counts[msg] = failure_counts.get(msg, 0) + 1
                        if len(sample_failures) < 20:
                            sample_failures.append((index, index_to_path.get(index, ""), msg))
                except Exception as e:
                    progress_tracker.update(success=False)
                    print(f"[failed] index=<unknown> msg={str(e)}")
                    failure_counts["worker_exception"] = failure_counts.get("worker_exception", 0) + 1
                    if len(sample_failures) < 20:
                        sample_failures.append(("<unknown>", "", str(e)))

        total_failed = sum(failure_counts.values())
        if total_failed > 0:
            print(f"[failure] total failed in workers: {total_failed}")
            print("failure reasons (count):")
            for msg, cnt in sorted(failure_counts.items(), key=lambda kv: -kv[1])[:10]:
                print(f"  {cnt} x {msg}")
            print("sample failed items:")
            for idx, path, msg in sample_failures[:10]:
                print(f"  index={idx} path={path} reason={msg}")
    else:
        for entry in file_pairs:
            output_path = os.path.join(batch_save_path, f"{entry['index']}.npz")
            try:
                details = process_single_robot_file(
                    pkl_path=entry["pkl_path"],
                    output_path=output_path,
                    robot=robot,
                    smplx_folder=str(smplx_folder),
                    gender=gender,
                    betas=np.asarray(betas_serializable) if betas_serializable is not None else None,
                    fps_override=fps_override if fps_override is not None else entry["fps"],
                    show_progress=show_progress,
                )
                print(
                    f"[ok] index={entry['index']} path={entry['pkl_path']} msg=processing completed, frames={details['frames']}"
                )
                progress_tracker.update(success=True)
            except Exception as e:
                print(f"[failed] index={entry['index']} path={entry['pkl_path']} msg={str(e)}")
                progress_tracker.update(success=False)

    summary = progress_tracker.get_summary()
    print(
        f"\nprocessing completed: successful {summary['successful']}/{summary['total_files']}, failed {summary['failed']}"
    )
    print(f"time: {summary['total_time']/60:.1f} minutes")


def parse_betas(arg_betas):
    if arg_betas is None:
        return DEFAULT_BETAS.astype(np.float64, copy=True)
    if isinstance(arg_betas, (list, tuple, np.ndarray)):
        betas = np.array(arg_betas, dtype=np.float64)
        if betas.size == 0:
            return DEFAULT_BETAS.astype(np.float64, copy=True)
        return betas
    raise ValueError("betas must be provided as a sequence of floats")


def main():
    parser = argparse.ArgumentParser(
        description="Reverse retargeting: Convert robot motion sequence back to SMPL-X human motion sequence in batch mode"
    )

    parser.add_argument(
        "--robot",
        choices=list(ROBOT_XML_DICT.keys()),
        default="unitree_g1",
        help="Robot type used to generate the PKL motions",
    )

    parser.add_argument(
        "--gender",
        choices=["male", "female", "neutral"],
        default="female",
        help="SMPL-X gender to instantiate",
    )

    parser.add_argument(
        "--betas",
        type=float,
        nargs="*",
        default=None,
        help="Optional SMPL-X body shape parameters (defaults to reference/010220.npz values; padded/truncated to model size)",
    )

    parser.add_argument(
        "--fps",
        type=float,
        default=30,
        help="Override FPS for all files (if not provided, uses PKL metadata or CSV column)",
    )

    parser.add_argument(
        "--csv_file",
        type=str,
        default=None,
        help="CSV file containing indices and PKL paths for batch processing",
    )

    parser.add_argument(
        "--batch_save_path",
        type=str,
        default=None,
        help="Directory path to save converted SMPL-X NPZ files",
    )

    parser.add_argument(
        "--start_row",
        type=int,
        default=0,
        help="Start row index (0-based, inclusive) for CSV processing",
    )

    parser.add_argument(
        "--end_row",
        type=int,
        default=None,
        help="End row index (0-based, exclusive) for CSV processing",
    )

    parser.add_argument(
        "--use_multithreading",
        action="store_true",
        default=True,
        help="Enable multiprocessing for batch processing",
    )

    parser.add_argument(
        "--num_threads",
        type=int,
        default=5,
        help="Number of processes to use (default: 10)",
    )

    parser.add_argument(
        "--base_data_path",
        type=str,
        default=None,
        help="Optional base directory to resolve relative PKL paths from CSV",
    )

    parser.add_argument(
        "--show_progress",
        action="store_true",
        help="Show per-file frame progress bars (slower when multiprocessing)",
    )

    parser.add_argument(
        "--robot_file",
        type=str,
        default=None,
        help="Single robot motion PKL file to convert (non-batch mode)",
    )

    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="Output path for single-file conversion (requires --robot_file)",
    )

    args = parser.parse_args()

    global START_ROW, END_ROW
    START_ROW = args.start_row
    END_ROW = args.end_row

    here = pathlib.Path(__file__).parent
    smplx_folder = here / "assets" / "body_models"

    betas = parse_betas(args.betas)
    print(f"Actual betas parameter setting: {betas.tolist()}")

    if args.csv_file and args.batch_save_path:
        process_batch_from_csv(
            csv_file=args.csv_file,
            batch_save_path=args.batch_save_path,
            robot=args.robot,
            smplx_folder=smplx_folder,
            gender=args.gender,
            betas=betas,
            fps_override=args.fps,
            use_multithreading=args.use_multithreading,
            num_threads=args.num_threads,
            base_data_path=args.base_data_path,
            show_progress=args.show_progress,
        )
    else:
        if args.robot_file and args.save_path:
            try:
                details = process_single_robot_file(
                    pkl_path=args.robot_file,
                    output_path=args.save_path,
                    robot=args.robot,
                    smplx_folder=str(smplx_folder),
                    gender=args.gender,
                    betas=betas,
                    fps_override=args.fps,
                    show_progress=args.show_progress,
                )
                print(
                    f"✅ Single file processed: frames={details['frames']} fps={details['fps']} -> {details['output_path']}"
                )
            except Exception as e:
                print(f"❌ Single file processing failed: {e}")
        else:
            print("⚠️  No batch or single-file processing requested. Nothing to do.")


if __name__ == "__main__":
    main()


