import argparse
import pathlib
import os
import time
import csv
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed
import math
import numpy as np
from general_motion_retargeting import RobotToSMPLXRetargeting
from general_motion_retargeting.robot import load_robot_motion
from rich import print

from general_motion_retargeting import RobotToSMPLXRetargeting, load_robot_motion
from general_motion_retargeting.params import REVERSE_IK_CONFIG_DICT


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
            current_affinity = os.sched_getaffinity(0) #获取cpu可用的所有线程数
            total_visible = len(current_affinity) #计算cpu预期使用的线程数，根据percent参数计算
            allowed = max(1, int(math.floor(total_visible * (percent / 100.0))))
            print("允许使用的 cpu 线程数: ", allowed)
            # 选择前 allowed 个 CPU
            cpus_sorted = sorted(current_affinity)  #将CPU线程编号集合排序
            target_set = set(cpus_sorted[:allowed]) #取出allowed个的cpu线程
            os.sched_setaffinity(0, target_set) #参数0表示调用这个函数的进程本身
            return allowed, total_visible
        # 回退：无法设置亲和性，仅返回估算
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
        """更新进度"""
        with self.lock:
            self.completed += 1
            if success:
                self.successful += 1
            else:
                self.failed += 1
                
    def get_summary(self):
        """获取统计信息"""
        with self.lock:
            total_time = time.time() - self.start_time
            return {
                'total_files': self.total_files,
                'completed': self.completed,
                'successful': self.successful,
                'failed': self.failed,
                'total_time': total_time
            }


# ===== 子任务处理函数（用于并行进程） =====
def process_single_file_worker(args):
    """处理单个文件（供进程池调用）"""
    index, pkl_path, output_path, robot, SMPLX_FOLDER, gender, betas, no_visualize, rate_limit = args
    try:
        # 检查输入文件是否存在
        if not os.path.exists(pkl_path):
            print(f"the input file {pkl_path} does not exist")
            return False, index, "the input file does not exist"
        # 检查输出文件是否已存在，如果存在则跳过处理
        if os.path.exists(output_path):
            print(f"the output file {output_path} already exists, skipping processing")
            return True, index, "the output file already exists, skipping processing"
        # 处理文件
        success = process_single_pkl_file(pkl_path, output_path, robot, SMPLX_FOLDER, gender, betas, no_visualize, rate_limit)
        return success, index, "processing completed" if success else "processing failed"
    except Exception as e:
        return False, index, str(e)


# ===== 批量处理模块化函数 =====
def process_batch_from_csv(csv_file, batch_save_path, robot, SMPLX_FOLDER, gender, betas, no_visualize=False, rate_limit=False, use_multithreading=True, num_threads=None):
    global START_ROW, END_ROW
    # 设置数据路径
    BASE_DATA_PATH = pathlib.Path(__file__).parent.parent.parent / "server3_data" / "locomotion" / "robot" / "ik_based" / "pkl"
    
    # 读取CSV文件
    file_pairs = []
    try:
        with open(csv_file, 'r') as csvfile:
            reader = csv.reader(csvfile)
            # 跳过标题行
            header = next(reader, None)
            # 找到source_path列
            source_path_column_index = None
            for i, column_name in enumerate(header):
                if 'source_path' in column_name.lower() or 'pkl' in column_name.lower():
                    source_path_column_index = i
                    break
            if source_path_column_index is None:
                source_path_column_index = 1  # 默认第二列
            
            row_idx = 0
            for row in reader:
                # 行范围过滤
                if START_ROW is not None and row_idx < START_ROW:
                    row_idx += 1
                    continue
                if END_ROW is not None and row_idx >= END_ROW:
                    break
                index = row[0].strip()
                relative_path = row[source_path_column_index].strip()
                if relative_path:
                    pkl_path = BASE_DATA_PATH / relative_path #组成绝对路径
                    file_pairs.append((index, str(pkl_path)))
                row_idx += 1
    except Exception as e:
        print(f"Error reading CSV file {csv_file}: {e}")
        return
    
    print(f"处理 {len(file_pairs)} 个文件")
    
    # 检查输出目录
    if not os.path.exists(batch_save_path):
        os.makedirs(batch_save_path)
    
    # 初始化进度跟踪器
    progress_tracker = ProgressTracker(len(file_pairs))
    
    if use_multithreading:
        # 计算在 CPU 使用上限的可用的最大进程数
        cpu_limit_percent = 60
        allowed_cores, visible_cores = cap_cpu_affinity_by_percent(cpu_limit_percent)
        cap_workers = allowed_cores
        user_workers = num_threads if num_threads is not None else cap_workers
        max_threads = max(1, min(user_workers, cap_workers))
        print("实际真正允许的最大进程数是: ", max_threads) 

        # 限制数值库的线程数，避免过度并行
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["NUMEXPR_NUM_THREADS"] = "1"
        
        # 准备任务参数
        tasks = []
        for index, pkl_path in file_pairs:
            output_path = os.path.join(batch_save_path, f"{index}.npz")
            task_args = (index, pkl_path, output_path, robot, SMPLX_FOLDER, gender, betas, no_visualize, rate_limit)
            tasks.append(task_args)
        
        # 并行处理所有任务（进程池）
        index_to_path = {index: pkl_path for index, pkl_path in file_pairs}
        failure_counts = {}
        sample_failures = []
        with ProcessPoolExecutor(max_workers=max_threads) as executor:
            futures = [executor.submit(process_single_file_worker, task) for task in tasks]
            for future in as_completed(futures):
                try:
                    success, index, msg = future.result()
                    progress_tracker.update(success=success)
                    print(f"[{'ok' if success else 'failed'}] index={index} path={index_to_path.get(index, '')} msg={msg}")
                    if not success:
                        failure_counts[msg] = failure_counts.get(msg, 0) + 1
                        if len(sample_failures) < 20:
                            sample_failures.append((index, index_to_path.get(index, ''), msg))
                except Exception as e:
                    progress_tracker.update(success=False)
                    print(f"[failed] index=<unknown> msg={str(e)}")
                    failure_counts['worker_exception'] = failure_counts.get('worker_exception', 0) + 1
                    if len(sample_failures) < 20:
                        sample_failures.append(("<unknown>", "", str(e)))
        
        # 打印失败统计与样例
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
        for index, pkl_path in file_pairs:
            output_path = os.path.join(batch_save_path, f"{index}.npz")
            task_args = (index, pkl_path, output_path, robot, SMPLX_FOLDER, gender, betas, no_visualize, rate_limit)
            success, _, msg = process_single_file_worker(task_args)
            print(f"[{'ok' if success else 'failed'}] index={index} path={pkl_path} msg={msg}")
            progress_tracker.update(success=success)
    
    # 显示最终统计
    summary = progress_tracker.get_summary()
    print(f"\nprocessing completed: successful {summary['successful']}/{summary['total_files']}, failed {summary['failed']}")
    print(f"time: {summary['total_time']/60:.1f} minutes")


# note: whole process of retargeting single pkl file  
def process_single_pkl_file(pkl_file_path, output_path, robot, SMPLX_FOLDER, gender, betas, no_visualize=False, rate_limit=False):
    """
    process a single PKL file 
    """
    try:
        # high priority: 加载机器人运动数据
        robot_data, motion_fps, root_pos, root_rot_wxyz, dof_pos, _, _ = load_robot_motion(pkl_file_path)
        
        # high priority: initialize the retargeting system
        retarget = RobotToSMPLXRetargeting(
            robot_type=robot,
            smplx_model_path=SMPLX_FOLDER,
            gender=gender,
        )

        # medium priority
        save_dir = os.path.dirname(output_path)
        if save_dir:  # Only create directory if it's not empty
            os.makedirs(save_dir, exist_ok=True)

        # high priority: doing retargeting frame by frame
        smplx_data_frames = []
        i = 0
        num_frames = root_pos.shape[0]
        while True:
            if i >= num_frames: #完成所有帧的处理
                break
            
            # 组合当帧的 qpos
            qpos = retarget.robot_kinematics.compose_qpos(
                root_pos[i],
                root_rot_wxyz[i],
                dof_pos[i],
            )
            
            # 计算机器人各 body 的世界姿态 (FK)
            robot_frame_data = retarget.robot_kinematics.forward_kinematics(qpos)
            
            # retarget: 调用 IK，将机器人姿态映射到 SMPL-X
            retarget.retarget(robot_frame_data)
            
            # 保存当帧 SMPL-X 关节姿态
            smplx_data_frames.append(retarget.extract_smplx_frame())
            
            i += 1

        # high priority: 转换为 SMPL-X 参数格式
        smplx_params = retarget.frames_to_smplx_parameters(smplx_data_frames, betas=betas)
        
        # high priority: save the retargeted SMPL-X motion data to npz file
        np.savez(
            output_path,
            betas=smplx_params["betas"],
            pose_body=smplx_params["pose_body"],
            root_orient=smplx_params["root_orient"],
            trans=smplx_params["trans"],
            gender=np.array(gender),
            mocap_frame_rate=np.array(motion_fps),
        )
        
        return True
        
    except Exception as e:
        print(f"❌ Error processing {pkl_file_path}: {e}")
        return False


if __name__ == "__main__":
    
    HERE = pathlib.Path(__file__).parent

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--robot_file",
        help="Robot motion PKL file to load.",
        type=str,
    )
    
    parser.add_argument(
        "--robot",
        choices=["unitree_g1", "unitree_g1_with_hands", "unitree_h1", "unitree_h1_2",
                 "booster_t1", "booster_t1_29dof","stanford_toddy", "fourier_n1", 
                "engineai_pm01", "kuavo_s45", "hightorque_hi", "galaxea_r1pro", "berkeley_humanoid_lite", "booster_k1",
                "pnd_adam_lite", "openloong", "tienkung"],
        default="unitree_g1",
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
        help="Optional SMPL-X body shape parameters (defaults to reference/010220.npz values)",
    )
    
    parser.add_argument(
        "--save_path",
        default=None,
        help="Path to save the SMPL-X motion.",
    )
    
    parser.add_argument(
        "--loop",
        default=False,
        action="store_true",
        help="Loop the motion.",
    )

    parser.add_argument(
        "--record_video",
        default=False,
        action="store_true",
        help="Record the video.",
    )

    parser.add_argument(
        "--rate_limit",
        default=False,
        action="store_true",
        help="Limit the rate of the retargeted motion.",
    )
    
    parser.add_argument(
        "--no_visualize",
        action="store_true",
        help="Disable visualization.",
    )
    
    # 批量处理参数
    parser.add_argument(
        "--csv_file",
        help="CSV file containing index and PKL file paths for batch processing.",
        type=str,
        default=None,
    )
    
    parser.add_argument(
        "--batch_save_path",
        help="Directory path to save batch processed SMPL-X motion files.",
        type=str,
        default=None,
    )
    
    # CSV 行范围（0-based）
    parser.add_argument(
        "--start_row",
        type=int,
        default=0,
        help="Start row index (0-based, inclusive) for CSV processing.",
    )
    parser.add_argument(
        "--end_row",
        type=int,
        default=None,
        help="End row index (0-based, exclusive) for CSV processing.",
    )
    
    # 多线程处理参数
    parser.add_argument(
        "--use_multithreading",
        action="store_true",
        default=True,
        help="Enable multithreading for batch processing.",
    )
    
    parser.add_argument(
        "--num_threads",
        type=int,
        default=10,
        help="Number of threads to use (default: 10).",
    )

    args = parser.parse_args()
    
    # 处理 betas 参数
    if args.betas is None or len(args.betas) == 0:
        betas = DEFAULT_BETAS.astype(np.float64, copy=True)
    else:
        betas = np.array(args.betas, dtype=np.float64)
    
    print(f"Actual betas parameter setting: {betas.tolist()}")

    # 检查是否使用批量处理模式
    if args.csv_file and args.batch_save_path:
        START_ROW = args.start_row
        END_ROW = args.end_row
        process_batch_from_csv(
            args.csv_file, 
            args.batch_save_path, 
            args.robot, 
            HERE / "assets" / "body_models", 
            args.gender,
            betas,
            args.no_visualize, 
            args.rate_limit,
            args.use_multithreading,
            args.num_threads
        )
    else:
        # 单文件处理模式
        SMPLX_FOLDER = HERE / "assets" / "body_models"
        
        # 单文件处理 - 调用模块化函数
        if args.save_path is not None:
            success = process_single_pkl_file(args.robot_file, args.save_path, args.robot, SMPLX_FOLDER, args.gender, betas, args.no_visualize, args.rate_limit)
            if not success:
                print("❌ Single file processing failed!")
        else:
            pass
