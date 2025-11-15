import argparse
import pathlib
import os
import time
import csv
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed
from queue import Queue
import math
import numpy as np
from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
from general_motion_retargeting.utils.smpl import load_smplx_file, get_smplx_data_offline_fast
from scipy.spatial.transform import Rotation as R
from smplx.joint_names import JOINT_NAMES
from rich import print
from tqdm import tqdm

# 可选的 CSV 行范围（由命令行参数设置）
START_ROW = 0     # 0-based inclusive（默认第一行数据，已跳过header）
END_ROW = None    # 0-based exclusive（默认处理到末尾）

#运行指令:
# python scripts/smplx_to_robot_batch.py --csv_file ../data/locomotion/manifest_raw.csv --batch_save_path ../data/locomotion/robot/ik_based/pkl/   --robot unitree_g1 --no_visualize --num_threads 1 --start_row 13611

#isaac4090: 
# python scripts/smplx_to_robot_batch.py --csv_file ../server3_data/locomotion/manifest_raw.csv --batch_save_path ../data/locomotion/robot/ik_based/pkl/   --robot unitree_g1 --no_visualize --num_threads 1 --end_row 6

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

# ===== 降采样函数 =====
def manual_downsample_smplx_data(smplx_data, body_model, smplx_output, down_sample=4):
    
    # Get original data
    num_frames = smplx_data["pose_body"].shape[0]
    #notice: smplx_output.global_orient is the root_orient 
    global_orient = smplx_output.global_orient.squeeze() #root_orient
    #notice: smplx_output.full_pose is the rotation of all controllable body joints (55*3 dimensions)
    #notice: full_body_pose/smplx_output.full_pose中数据顺序: global_orient(3) + body_pose(63) + hand_pose(90) + jaw_pose(3) + leye_pose(3) + reye_pose(3) -> (N, 55, 3)
    full_body_pose = smplx_output.full_pose.reshape(num_frames, -1, 3) #full_pose是SMPLX模型自带的一个参量
    print("In manual_downsample_smplx_data: full_body_pose shape: ", full_body_pose.shape) #reshape后的shape: (N, 55, 3)
    #notice: smplx_output.joints is the 3D position 绝对位置 of all joints [N, 127, 3]
    joints = smplx_output.joints.detach().numpy().squeeze() #joints是SMPLX模型自带的一个参量: 3D位置
    joint_names = JOINT_NAMES[: len(body_model.parents)] #parents是SMPLX模型自带的一个参量
    parents = body_model.parents # body_model is from SMPLX_FEMALE/SMPLX_MALE/SMPLX_NEUTRAL.npz file
    
    # Downsample by taking every down_sample-th frame
    downsampled_global_orient = global_orient[::down_sample]
    downsampled_full_body_pose = full_body_pose[::down_sample]
    downsampled_joints = joints[::down_sample]
    
    # Create smplx_data_frames with same structure as original function: 这部分代码和原降采样函数中的一模一样，为了以相同结构组织降采样后的数据
    smplx_data_frames = []
    for curr_frame in range(len(downsampled_global_orient)):
        result = {}
        single_global_orient = downsampled_global_orient[curr_frame]
        single_full_body_pose = downsampled_full_body_pose[curr_frame] #notice: single_full_body_pose是当前帧所有关节的旋转参数
        single_joints = downsampled_joints[curr_frame]
        joint_orientations = []
        for i, joint_name in enumerate(joint_names): #notice: joint_name顺序是标准的SMPLX模型中的顺序(from smplx.joint_names import JOINT_NAMES)
            if i == 0: #pelvis的旋转对象是global_orient
                rot = R.from_rotvec(single_global_orient) #pelvis的旋转对象
            else:
                #该关节的世界旋转
                rot = joint_orientations[parents[i]] * R.from_rotvec(
                    single_full_body_pose[i].squeeze() #single_full_body_pose[i]: 当前关节相对于父关节的局部旋转
                )
            joint_orientations.append(rot)
            result[joint_name] = (single_joints[i], rot.as_quat(scalar_first=True)) #!wxyz
            #rot.as_quat(scalar_first=True)   # → [w, x, y, z] 
            #rot.as_quat(scalar_first=False)  # → [x, y, z, w]

        smplx_data_frames.append(result)
    
    # ===== Verification: Test compute_local_rotations =====
    # Validate that compute_local_rotations can correctly recover local rotations
    print("\n[Verification] Testing compute_local_rotations correctness:")
    for test_frame_idx in range(0, min(11, len(smplx_data_frames))):  # check frame 0-10
        frame_data = smplx_data_frames[test_frame_idx]
        
        # Compute local rotations using the function to test
        global_rots = {}
        for joint_name, (pos, quat_wxyz) in frame_data.items():
            global_rots[joint_name] = R.from_quat(quat_wxyz[[1, 2, 3, 0]])  # wxyz -> xyzw
        
        computed_body_pose = np.zeros(63, dtype=np.float64)
        for i in range(1, min(22, len(joint_names))):
            joint_name = joint_names[i]
            parent_idx = parents[i]
            parent_name = joint_names[parent_idx] # joint_names is from python lib "smplx.joint_names" JOINT_NAMES
            if joint_name in global_rots and parent_name in global_rots:
                parent_R = global_rots[parent_name]
                joint_R = global_rots[joint_name]
                local_R = parent_R.inv() * joint_R
                computed_body_pose[(i - 1) * 3 : (i - 1) * 3 + 3] = local_R.as_rotvec()
        
        # Compare with ground truth
        ground_truth_tensor = downsampled_full_body_pose[test_frame_idx][1:22]  # Skip pelvis (index 0)
        # notice： body-pose确实不包含pelvis
        if hasattr(ground_truth_tensor, "detach"):
            ground_truth = ground_truth_tensor.detach().cpu().numpy().reshape(-1)
        else:
            ground_truth = np.asarray(ground_truth_tensor).reshape(-1)
        diff = np.abs(computed_body_pose - ground_truth)
        # print(f"  Frame {test_frame_idx}: computed_body_pose={computed_body_pose}, ground_truth={ground_truth}")
        
        max_error = np.max(diff)
        mean_error = np.mean(diff)
        
        # print(f"  Frame {test_frame_idx}: max_error={max_error:.6f}, mean_error={mean_error:.6f}")
        if max_error > 0.01:
            print(f"  ⚠️[WARNING] Large error detected in compute_local_rotations!")
            print(f"  Computed (first 9): {computed_body_pose[:9]}")
            print(f"  GroundTruth (first 9): {ground_truth[:9]}")
    
    # Calculate aligned fps based on downsampling
    src_fps = smplx_data["mocap_frame_rate"].item()
    aligned_fps = src_fps / down_sample
    
    return smplx_data_frames, aligned_fps
    

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

# ===== 打印 DOF 信息（封装原有打印逻辑） =====
def print_dof_info(retarget, dof_pos, root_pos=None, root_rot=None, local_body_pos=None, body_names=None, qpos_list=None):
    pass
    # print(f"root_pos shape: {root_pos.shape}")
    # print(f"root_rot shape: {root_rot.shape}")
    # print(f"dof_pos shape: {dof_pos.shape}")
    # print(f"local_body_pos: {local_body_pos}")
    # print(f"body_names: {body_names}")
    # print(f"qpos_list length: {len(qpos_list)}")
    # print(f"qpos shape: {qpos_list[0].shape}")
    
    # print(f"\n=== DOF (Degrees of Freedom) Information ===")
    # if hasattr(retarget, 'robot_dof_names'):
    #     all_dof_names = list(retarget.robot_dof_names.keys())
    #     print(f"All DOF names (including pelvis): {all_dof_names}")
    #     print(f"All DOF names count: {len(all_dof_names)}")
    #     print(f"dof_pos shape[1]: {dof_pos.shape[1]}")
    #     # dof_pos starts from index 7 (after root_pos and root_rot)
    #     # So we need to skip the first DOF name (pelvis-related)
    #     dof_names = all_dof_names[1:]  # Skip the first one (pelvis)
    #     print(f"DOF names (excluding pelvis): {dof_names}")
    #     print(f"DOF names count (excluding pelvis): {len(dof_names)}")
    #     # Check if lengths match
    #     if len(dof_names) != dof_pos.shape[1]:
    #         print(f"⚠️  WARNING: DOF names count ({len(dof_names)}) != dof_pos count ({dof_pos.shape[1]})")
    #     # Print first frame DOF values with names
    #     if len(dof_pos) > 0:
    #         print(f"\nFirst frame DOF values:")
    #         # Use the minimum length to avoid index errors
    #         min_length = min(len(dof_names), len(dof_pos[0]))
    #         for i in range(min_length):
    #             name = dof_names[i] if i < len(dof_names) else f"unknown_dof_{i}"
    #             value = dof_pos[0][i]
    #             print(f"  {i:2d}: {name:25s} = {value:8.4f}")
    # else:
    #     print("DOF names not available")

# ===== 子任务处理函数（用于并行进程） =====
def process_single_file_worker(args):
    """处理单个文件（供进程池调用）"""
    index, npz_path, output_path, robot, SMPLX_FOLDER, no_visualize, rate_limit, downsample_factor = args
    # print(f"[check] index={index} input={os.path.basename(npz_path)}")
    try:
        # 检查输入文件是否存在
        if not os.path.exists(npz_path):
            print(f"the input file {npz_path} does not exist")
            return False, index, "the input file does not exist"
        # 检查输出文件是否已存在，如果存在则跳过处理
        if os.path.exists(output_path):
            print(f"the output file {output_path} already exists, skipping processing")
            return True, index, "the output file already exists, skipping processing"
        # 处理文件
        success = process_single_npz_file(npz_path, output_path, robot, SMPLX_FOLDER, no_visualize, rate_limit, downsample_factor)
        return success, index, "processing completed" if success else "processing failed"
    except Exception as e:
        return False, index, str(e)

# ===== 批量处理模块化函数 =====
def process_batch_from_csv(csv_file, batch_save_path, robot, SMPLX_FOLDER, no_visualize=False, rate_limit=False, use_multithreading=True, num_threads=None):
    global START_ROW, END_ROW
    # 设置数据路径
    # BASE_DATA_PATH = pathlib.Path(__file__).parent.parent.parent / "data" / "locomotion" / "raw"
    BASE_DATA_PATH = pathlib.Path(__file__).parent.parent.parent / "data" / "locomotion" 
    BASE_DATA_PATH = pathlib.Path(__file__).parent.parent.parent / "server3_data" / "locomotion" 
    
    # 读取CSV文件
    file_pairs = []
    try:
        with open(csv_file, 'r') as csvfile:
            reader = csv.reader(csvfile)
            # 跳过标题行
            header = next(reader, None)
            # 找到source_path和downsample_factor列
            source_path_column_index = None
            downsample_factor_column_index = None
            for i, column_name in enumerate(header):
                if 'source_path' in column_name.lower():
                    source_path_column_index = i
                elif 'downsample_factor' in column_name.lower():
                    downsample_factor_column_index = i
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
                # absolute_path = 
                downsample_factor = int(row[downsample_factor_column_index].strip())
                if relative_path:
                    npz_path = BASE_DATA_PATH / relative_path #组成绝对路径
                    file_pairs.append((index, str(npz_path), downsample_factor))
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
    
    #!实际运行中，运行的最大线程数可以比运行的最大进程数大
    #处理单个npz文件会放入单个进程，而该进程里调用的数值库可能会自己“开多线程”加速单个运算
    if use_multithreading:
        # 计算在 CPU 使用上限的可用的最大进程数
        cpu_limit_percent = 60
        allowed_cores, visible_cores = cap_cpu_affinity_by_percent(cpu_limit_percent)
        cap_workers = allowed_cores #cap_workers本质是线程数
        #但是user_workers本质是进程数
        user_workers = num_threads if num_threads is not None else cap_workers
        max_threads = max(1, min(user_workers, cap_workers))
        print("实际真正允许的最大进程数是: ", max_threads) 
        #这个本质数量计算本质没什么用，cap_cpu_affinity_by_percent(60)把父进程（及子进程）绑在约60%的CPU核心上

        # 限制数值库的线程数，避免过度并行
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["NUMEXPR_NUM_THREADS"] = "1"
        
        # 准备任务参数
        tasks = []
        for index, npz_path, downsample_factor in file_pairs:
            output_path = os.path.join(batch_save_path, f"{index}.pkl")
            task_args = (index, npz_path, output_path, robot, SMPLX_FOLDER, no_visualize, rate_limit, downsample_factor)
            tasks.append(task_args)
        
        # 并行处理所有任务（进程池）， 进程并行适用于cpu密集型任务（即本任务）
        index_to_path = {index: npz_path for index, npz_path, _ in file_pairs}
        failure_counts = {}
        sample_failures = []
        #用ProcessPoolExecutor线程池本身就不太合理
        with ProcessPoolExecutor(max_workers=max_threads) as executor: # 创建进程池
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
        # print("single-thread processing")
        for index, npz_path, downsample_factor in file_pairs:
            output_path = os.path.join(batch_save_path, f"{index}.pkl")
            task_args = (index, npz_path, output_path, robot, SMPLX_FOLDER, no_visualize, rate_limit, downsample_factor)
            success, _, msg = process_single_file_worker(task_args)
            print(f"[{'ok' if success else 'failed'}] index={index} path={npz_path} msg={msg}")
            progress_tracker.update(success=success)
    
    # 显示最终统计
    summary = progress_tracker.get_summary()
    print(f"\nprocessing completed: successful {summary['successful']}/{summary['total_files']}, failed {summary['failed']}")
    print(f"time: {summary['total_time']/60:.1f} minutes")


# note: whole process of retargeting signle npz file  
def process_single_npz_file(smplx_file_path, output_path, robot, SMPLX_FOLDER, no_visualize=False, rate_limit=False, downsample_factor=4):
    """
    process a single NPZ file 
    """
    try:
        # high priority:::  smplx_data: used in retargeting process
        # notice: smplx_data includes the complete data structure of reference human data (npz file)
        smplx_data, body_model, smplx_output, actual_human_height = load_smplx_file(
            smplx_file_path, SMPLX_FOLDER
        ) #load_smplx_file() is in /general_motion_retargeting/utils/smpl.py 
        #notice: smplx_output 
        
        
        #自己手写的降采样方式
        #? it seems that we need to downsampl in reverse of retargeting process
        smplx_data_frames, aligned_fps = manual_downsample_smplx_data(
            smplx_data, body_model, smplx_output, down_sample=downsample_factor
        )
   
        # high priority: initialize the retargeting system
        retarget = GMR(
            actual_human_height=actual_human_height,
            src_human="smplx",
            tgt_robot=robot,
        ) #GMR is the class "GeneralMotionRetargeting" from general_motion_retargeting  in /general_motion_retargeting/motion_retarget.py
        

        # low priority: visualize
        if not no_visualize:
            robot_motion_viewer = RobotMotionViewer(robot_type=robot,
                                                    motion_fps=aligned_fps,
                                                    transparent_robot=0,
                                                    record_video=False,
                                                    video_path=f"videos/{robot}_{os.path.basename(smplx_file_path).split('.')[0]}.mp4",)
        else:
            robot_motion_viewer = None
        curr_frame = 0
        fps_counter = 0
        fps_start_time = time.time()
        fps_display_interval = 2.0  # Display FPS every 2 seconds

        
        #medium priority
        save_dir = os.path.dirname(output_path)
        if save_dir:  # Only create directory if it's not empty
            os.makedirs(save_dir, exist_ok=True)


        # high priority: doing retargeting frame by frame
        qpos_list = []
        i = 0
        while True:
            if i >= len(smplx_data_frames): #完成所有帧的处理
                break
            # FPS measurement
            fps_counter += 1
            current_time = time.time()
            if current_time - fps_start_time >= fps_display_interval:
                actual_fps = fps_counter / (current_time - fps_start_time)
                # print(f"Actual rendering FPS: {actual_fps:.2f}")
                fps_counter = 0
                fps_start_time = current_time
            # Update task targets.
            #notice: 这里的smplx_data并不再有原始npz文件的完整数据结构，它只包含单帧的数据。而且单帧的数据结构也有变化(由manual_downsample_smplx_data()进行重构)
            smplx_data = smplx_data_frames[i]
            # retarget
            qpos = retarget.retarget(smplx_data)
            qpos_list.append(qpos) #add processed qpos (a frame) to qpos_list
            i += 1



            # low priority: visualize
            if robot_motion_viewer:
                robot_motion_viewer.step(
                    root_pos=qpos[:3],
                    root_rot=qpos[3:7],
                    dof_pos=qpos[7:],
                    human_motion_data=retarget.scaled_human_data,
                    # human_motion_data=smplx_data,
                    human_pos_offset=np.array([0.0, 0.0, 0.0]),
                    show_human_body_name=False,
                    rate_limit=rate_limit,
                )


        # high priority: save the retargeted robot motion data to pkl file
        import pickle
        root_pos = np.array([qpos[:3] for qpos in qpos_list])
        # save from wxyz to xyzw
        root_rot = np.array([qpos[3:7][[1,2,3,0]] for qpos in qpos_list])
        dof_pos = np.array([qpos[7:] for qpos in qpos_list]) #除了root关节之外的自由度
        local_body_pos = None
        body_names = None
        # Print DOF names and values
        print_dof_info(retarget, dof_pos, root_pos, root_rot, local_body_pos, body_names, qpos_list)
        # high priority: save the retargeted robot motion data to pkl file
        motion_data = {
            "fps": aligned_fps,
            "root_pos": root_pos, 
            "root_rot": root_rot, #notice：这里的root_rot是xyzw格式
            "dof_pos": dof_pos,  #notice: dof_pos includes angle(一维) for all joints (robot joint从物理硬件上只能在一个维度旋转)
            "local_body_pos": local_body_pos,  #is none? # npz文件呢？ #没有手部信息吗？ #那怎么可视化如果不用前向运动学？ 
            "link_body_list": body_names,
        }
        with open(output_path, "wb") as f:
            pickle.dump(motion_data, f)
        # print(f"Saved to {output_path}")
            

        #low priority: visualize        
        if robot_motion_viewer:
            robot_motion_viewer.close()
        
        return True
        
    except Exception as e:
        print(f"❌ Error processing {smplx_file_path}: {e}")
        return False



if __name__ == "__main__":
    
    HERE = pathlib.Path(__file__).parent

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smplx_file",
        help="SMPLX motion file to load.",
        type=str,
        # required=True,
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
        "--save_path",
        default=None,
        help="Path to save the robot motion.",
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

    #播放速度，当true时，机器人动作会按照原始人类动作的帧率播放。
    parser.add_argument(
        "--rate_limit",
        default=False,
        action="store_true",
        help="Limit the rate of the retargeted robot motion to keep the same as the human motion.",
    )
    
    parser.add_argument(
        "--no_visualize",
        action="store_true",
        help="Disable visualization.",
    )
    
    # 批量处理参数
    parser.add_argument(
        "--csv_file",
        help="CSV file containing index and NPZ file paths for batch processing.",
        type=str,
        default=None,
    )
    
    parser.add_argument(
        "--batch_save_path",
        help="Directory path to save batch processed robot motion files.",
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

    # 检查是否使用批量处理模式
    if args.csv_file and args.batch_save_path:
        START_ROW = args.start_row
        END_ROW = args.end_row
        # print("🔄 Starting batch processing mode...")
        process_batch_from_csv(
            args.csv_file, 
            args.batch_save_path, 
            args.robot, 
            HERE / "assets" / "body_models", 
            args.no_visualize, 
            args.rate_limit,
            args.use_multithreading,
            args.num_threads
        )
    else:
        # 单文件处理模式
        SMPLX_FOLDER = HERE / ".." / "assets" / "body_models"
        SMPLX_FOLDER = HERE / "assets" / "body_models"
        
        # 单文件处理 - 调用模块化函数
        if args.save_path is not None:
            success = process_single_npz_file(args.smplx_file, args.save_path, args.robot, SMPLX_FOLDER, args.no_visualize, args.rate_limit, downsample_factor=4)
            if not success:
                print("❌ Single file processing failed!")
        else:
            # print("⚠️  No save_path specified, skipping processing")
            pass
