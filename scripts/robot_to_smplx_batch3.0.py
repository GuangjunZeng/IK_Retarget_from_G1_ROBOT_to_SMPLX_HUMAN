import argparse
import csv
import math
import os
import pathlib
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from rich import print

from general_motion_retargeting import RobotToSMPLXRetargeting, load_robot_motion
from general_motion_retargeting.params import REVERSE_IK_CONFIG_DICT


# 可选的 CSV 行范围（由命令行参数设置）
START_ROW = 0     # 0-based inclusive（默认第一行数据，已跳过 header）
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


def convert_robot_motion(
    pkl_path,
    output_path,
    robot,
    smplx_folder,
    gender,
    betas,
    fps_override,
    show_progress=False,
):
    _, motion_fps, root_pos, root_rot_wxyz, dof_pos, _, _ = load_robot_motion(pkl_path)
    if fps_override is not None:
        motion_fps = fps_override

    retarget = RobotToSMPLXRetargeting(
        robot_type=robot,
        smplx_model_path=smplx_folder,
        gender=gender,
    )

    num_frames = root_pos.shape[0]
    iterator = range(num_frames)
    if show_progress:
        from tqdm import tqdm

        iterator = tqdm(iterator, desc=f"retarget {pathlib.Path(pkl_path).name}", leave=False)

    frames = []
    for idx in iterator:
        qpos = retarget.robot_kinematics.compose_qpos(root_pos[idx], root_rot_wxyz[idx], dof_pos[idx])
        robot_frame = retarget.robot_kinematics.forward_kinematics(qpos)
        retarget.retarget(robot_frame)
        frames.append(retarget.extract_smplx_frame())

    smplx_params = retarget.frames_to_smplx_parameters(frames, betas=betas)

    np.savez(
        output_path,
        betas=smplx_params["betas"],
        pose_body=smplx_params["pose_body"],
        root_orient=smplx_params["root_orient"],
        trans=smplx_params["trans"],
        gender=np.array(gender),
        mocap_frame_rate=np.array(motion_fps),
    )

    return {
        "frames": num_frames,
        "fps": motion_fps,
        "output_path": output_path,
    }


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
        result = convert_robot_motion(
            pkl_path=pkl_path,
            output_path=output_path,
            robot=robot,
            smplx_folder=smplx_folder,
            gender=gender,
            betas=betas_array,
            fps_override=fps_override,
            show_progress=show_progress,
        )
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
                details = convert_robot_motion(
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
        description="Reverse retargeting: Convert robot motion (PKL) back to SMPL-X (NPZ) in batch mode"
    )

    parser.add_argument(
        "--robot",
        choices=list(REVERSE_IK_CONFIG_DICT.keys()),
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
                details = convert_robot_motion(
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
 