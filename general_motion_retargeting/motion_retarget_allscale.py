
import mink
import mujoco as mj
import numpy as np
import json
import os
import copy
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from .params import ROBOT_XML_DICT, IK_CONFIG_DICT
from rich import print

class GeneralMotionRetargeting_allscale:
    """General Motion Retargeting (GMR) with all scale.
    """
    def __init__(
        self,
        src_human: str,  #Input human motion data format: "smplx", "bvh_lafan1", "bvh_nokov", "fbx", "fbx_offline"
        tgt_robot: str,  #robot option: g1, h1, ....
        actual_human_height: float = None,
        solver: str="daqp", #IK solver type: change from "quadprog" to "daqp". (影响性能)
        damping: float=5e-1, #IK damping coefficient: change from 1e-1 to 1e-2. (影响稳定性)
        verbose: bool=True,  #Whether to print debug information
        use_velocity_limit: bool=False, #Whether to limit joint speed
    ) -> None:


        # high priority: load the robot model 
        self._tgt_robot = tgt_robot
        self.xml_file = str(ROBOT_XML_DICT[tgt_robot])
        if verbose:
            # print("Use robot model: ", self.xml_file)
            pass
        self.model = mj.MjModel.from_xml_path(self.xml_file)
        

        # low priority: print DoF names in order
        # print("[GMR] Robot Degrees of Freedom (DoF) names and their order:")
        self.robot_dof_names = {}
        for i in range(self.model.nv):  # 'nv' is the number of DoFs
            dof_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_JOINT, self.model.dof_jntid[i])
            self.robot_dof_names[dof_name] = i
            if verbose:
                # print(f"DoF {i}: {dof_name}")
                pass
        # low priority: print("[GMR] Robot Body names and their IDs:")
        self.robot_body_names = {}
        for i in range(self.model.nbody):  # 'nbody' is the number of bodies
            body_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_BODY, i)
            self.robot_body_names[body_name] = i
            if verbose:
                # print(f"Body ID {i}: {body_name}")
                pass
        # low priority: print("[GMR] Robot Motor (Actuator) names and their IDs:")
        self.robot_motor_names = {}
        for i in range(self.model.nu):  # 'nu' is the number of actuators (motors)
            motor_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_ACTUATOR, i)
            self.robot_motor_names[motor_name] = i
            if verbose:
                # print(f"Motor ID {i}: {motor_name}")
                pass


        # high priority: Load the IK config
        with open(IK_CONFIG_DICT[src_human][tgt_robot]) as f:
            ik_config = json.load(f)
        if verbose:
            # print("Use IK config: ", IK_CONFIG_DICT[src_human][tgt_robot])
            pass
        
        # high priority: compute the scale ratio based on given human height and the assumption in the IK config
        # warning
        if actual_human_height is not None:
            #得到放缩比例
            ratio = actual_human_height / ik_config["human_height_assumption"] #what is the "human_height_assumption" in the IK config?
        else:
            ratio = 1.0
        # Apply the height ratio to the scaling factor of each body part
        # print(f"ratio: {ratio}") #000005.npz: ratio is 0.9497133053910921
        for key in ik_config["human_scale_table"].keys():
            ik_config["human_scale_table"][key] = ik_config["human_scale_table"][key] * ratio
    
        # print("\n=== Updated human_scale_table ===")
        # for key, value in ik_config["human_scale_table"].items():
        #         print(f"  {key}: {value:.15f}")  # 显示15位小数
        # print("==================================\n")

        #000005.npz:
        # pelvis: 0.854741974851983
        # spine3: 0.854741974851983
        # left_hip: 0.854741974851983
        # right_hip: 0.854741974851983
        # left_knee: 0.854741974851983
        # right_knee: 0.854741974851983
        # left_foot: 0.854741974851983
        # right_foot: 0.854741974851983
        # left_shoulder: 0.759770644312874
        # right_shoulder: 0.759770644312874
        # left_elbow: 0.759770644312874
        # right_elbow: 0.759770644312874
        # left_wrist: 0.759770644312874
        # right_wrist: 0.759770644312874

        # high priority: used for retargeting
        self.ik_match_table1 = ik_config["ik_match_table1"]
        #pelvis, left_hip_roll_link, left_knee_link, left_toe_link, right_hip_roll_link, right_knee_link, right_toe_link, torso_link, left_shoulder_yaw_link, left_elbow_link, left_wrist_yaw_link, right_shoulder_yaw_link, right_elbow_link, right_wrist_yaw_link,  
        self.ik_match_table2 = ik_config["ik_match_table2"]
        #pelvis, left_hip_roll_link, left_knee_link, left_toe_link, right_hip_roll_link, right_knee_link, right_toe_link, torso_link, left_shoulder_yaw_link, left_elbow_link, left_wrist_yaw_link, right_shoulder_yaw_link, right_elbow_link, right_wrist_yaw_link
        self.human_root_name = ik_config["human_root_name"]
        self.robot_root_name = ik_config["robot_root_name"]
        self.use_ik_match_table1 = ik_config["use_ik_match_table1"]
        self.use_ik_match_table2 = ik_config["use_ik_match_table2"]
        self.human_scale_table = ik_config["human_scale_table"]
        self.ground = ik_config["ground_height"] * np.array([0, 0, 1])
        self.max_iter = 10
        self.solver = solver
        self.damping = damping
        self.human_body_to_task1 = {}
        self.human_body_to_task2 = {}
        self.pos_offsets1 = {}
        self.rot_offsets1 = {}
        self.pos_offsets2 = {}
        self.rot_offsets2 = {}
        self.task_errors1 = {}
        self.task_errors2 = {}
        self.ik_limits = [mink.ConfigurationLimit(self.model)]
        if use_velocity_limit:
            VELOCITY_LIMITS = {k: 3*np.pi for k in self.robot_motor_names.keys()}
            self.ik_limits.append(mink.VelocityLimit(self.model, VELOCITY_LIMITS)) 
        self.setup_retarget_configuration()
        self.ground_offset = 0.0
        #！ one-time flag to mimic PHC-style constant ground alignment using the first frame
        self._ground_offset_initialized = False


    # high priority: set the configuration parameters for motion retargeting.
    def setup_retarget_configuration(self):
        # create a "robot state container", and the IK solver will continuously update the joint angles in this container
        self.configuration = mink.Configuration(self.model) 
        self.tasks1 = []  #Phase I IK Task List (Coarse Adjustment)
        self.tasks2 = []  #Phase II IK Task List (Fine-Tuning)
        #both ik_match_table1 and ik_match_table2 are derived from smplx_to_g1.json
        for frame_name, entry in self.ik_match_table1.items():
            body_name, pos_weight, rot_weight, pos_offset, rot_offset = entry
            if pos_weight != 0 or rot_weight != 0:
                # create a task for every joint of robot
                task = mink.FrameTask(
                    frame_name=frame_name,
                    frame_type="body",
                    position_cost=pos_weight,
                    orientation_cost=rot_weight,
                    lm_damping=1,
                )

                #warning: body_name 是 ik_match_table1中value的一个, it's the smplx human body name
                #establish the mapping relationship from "human joint names" to "robot IK tasks"
                self.human_body_to_task1[body_name] = task  #
                self.pos_offsets1[body_name] = np.array(pos_offset) - self.ground #将偏移量从"相对于地面"转换为"相对于世界坐标系原点"
                self.rot_offsets1[body_name] = R.from_quat(
                    rot_offset, scalar_first=True
                ) #将四元数转换为旋转对象并存储
                self.tasks1.append(task) #a task is essentially a constraint condition for IK solving
                self.task_errors1[task] = []
        
        for frame_name, entry in self.ik_match_table2.items():
            body_name, pos_weight, rot_weight, pos_offset, rot_offset = entry
            if pos_weight != 0 or rot_weight != 0:
                task = mink.FrameTask(
                    frame_name=frame_name, #frame_name is robot body name
                    frame_type="body",
                    position_cost=pos_weight,
                    orientation_cost=rot_weight,
                    lm_damping=1,
                )
                self.human_body_to_task2[body_name] = task
                self.pos_offsets2[body_name] = np.array(pos_offset) - self.ground
                self.rot_offsets2[body_name] = R.from_quat(
                    rot_offset, scalar_first=True
                )
                self.tasks2.append(task)
                self.task_errors2[task] = []

    # high priority: using scale and offset to compute target pose of every joint of robot, and update the (IK)task targets.
    def update_targets(self, human_data, offset_to_ground=False):
        human_data = self.to_numpy(human_data) #ensure that all data is in NumPy array format
        # scale the human data in the Global Coordinate System
        human_data = self.scale_human_data(human_data, self.human_root_name, self.human_scale_table)
        # print(f"after scale_human_data(), human_data_left_shoulder: {human_data['left_shoulder']}")
        # offset the human data according to the ik config (eg: smplx_to_g1.json)
        human_data = self.offset_human_data(human_data, self.pos_offsets1, self.rot_offsets1)
        # print(f"after offset_human_data(), human_data_left_shoulder: {human_data['left_shoulder']}")
        human_data = self.apply_ground_offset(human_data) #所有帧z轴都都下平移一个常量
        if offset_to_ground: #offset_to_ground默认是false
            human_data = self.offset_human_data_to_ground(human_data)
        # All scale and offset operations completed, saved to self.scaled_human_data
        self.scaled_human_data = human_data

        if self.use_ik_match_table1:
            for body_name in self.human_body_to_task1.keys():
                task = self.human_body_to_task1[body_name] #notice: body_name always is the smplx human body name
                pos, rot = human_data[body_name] #notice: currently, pos, rot is the scaled and offsetted data (targeted robot joint position and rotation)
                task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3(rot), pos))
                #mink.SO3(rot):将四元数 rot = [w, x, y, z] 转换为 SO(3) 旋转对象
                #mink.SE3.from_rotation_and_translation(...): 组合旋转和平移，创建完整的 6D 位姿（pose）
                #task.set_target(transform):将目标位姿存储在task中, IK 求解器会尝试调整机器人关节角度，使机器人的关节达到这个目标
        
        if self.use_ik_match_table2:
            for body_name in self.human_body_to_task2.keys():
                task = self.human_body_to_task2[body_name]
                pos, rot = human_data[body_name]
                task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3(rot), pos))
            

    # high priority: solve the IK problem using the task targets gotten in update_targets().
    # notice: "human_data" is the "smplx_data" in smplx_to_robot_batch.py, and smplx_data includes the complete data structure of reference human data (npz file)
    def retarget(self, human_data, offset_to_ground=False):
        # Update the task targets
        self.update_targets(human_data, offset_to_ground)

        if self.use_ik_match_table1:
            # Solve the IK problem
            curr_error = self.error1()
            dt = self.configuration.model.opt.timestep
            # notice: self.configuration includes info of "g1_mocap_29dof.xml", self.tasks1 includes info of all ik constraints of ik first phase
            vel1 = mink.solve_ik(
                self.configuration, self.tasks1, dt, self.solver, self.damping, self.ik_limits
            )
            self.configuration.integrate_inplace(vel1, dt)
            next_error = self.error1()
            num_iter = 0
            while curr_error - next_error > 0.001 and num_iter < self.max_iter:
                curr_error = next_error
                dt = self.configuration.model.opt.timestep
                vel1 = mink.solve_ik(
                    self.configuration, self.tasks1, dt, self.solver, self.damping, self.ik_limits
                )
                self.configuration.integrate_inplace(vel1, dt)
                next_error = self.error1()
                num_iter += 1

        if self.use_ik_match_table2:
            curr_error = self.error2()
            dt = self.configuration.model.opt.timestep
            vel2 = mink.solve_ik(
                self.configuration, self.tasks2, dt, self.solver, self.damping, self.ik_limits
            )
            self.configuration.integrate_inplace(vel2, dt)
            next_error = self.error2()
            num_iter = 0
            while curr_error - next_error > 0.001 and num_iter < self.max_iter:
                curr_error = next_error
                # Solve the IK problem with the second task
                dt = self.configuration.model.opt.timestep
                vel2 = mink.solve_ik(
                    self.configuration, self.tasks2, dt, self.solver, self.damping, self.ik_limits
                )
                self.configuration.integrate_inplace(vel2, dt)
                
                next_error = self.error2()
                num_iter += 1
                
        # qpos includes the root position and rotation, and the joint angles of all joints
        return self.configuration.data.qpos.copy()


    def error1(self):
        return np.linalg.norm(
            np.concatenate(
                [task.compute_error(self.configuration) for task in self.tasks1]
            )
        )
    
    def error2(self):
        return np.linalg.norm(
            np.concatenate(
                [task.compute_error(self.configuration) for task in self.tasks2]
            )
        )


    def to_numpy(self, human_data):
        for body_name in human_data.keys():
            human_data[body_name] = [np.asarray(human_data[body_name][0]), np.asarray(human_data[body_name][1])]
        return human_data


    # high priority: scale the human data in the Global Coordinate System
    def scale_human_data(self, human_data, human_root_name, human_scale_table):
        
        human_data_local = {}
        root_pos, root_quat = human_data[human_root_name]
        
        # scale root
        scaled_root_pos = human_scale_table[human_root_name] * root_pos
        
         # scale other body parts in local frame
        for body_name in human_data.keys():
            if body_name not in human_scale_table:
                continue
            if body_name == human_root_name:
                continue
            else:
                # transform to local frame (only position)
                human_data_local[body_name] = (human_data[body_name][0] - root_pos) * human_scale_table[body_name]
    
            
        # transform the human data back to the global frame
        human_data_global = {human_root_name: (scaled_root_pos, root_quat)}
        for body_name in human_data_local.keys():
            human_data_global[body_name] = (human_data_local[body_name] + scaled_root_pos, human_data[body_name][1])

        return human_data_global
    
    # high priority: offset the human data according to the ik config (eg: smplx_to_g1.json)
    def offset_human_data(self, human_data, pos_offsets, rot_offsets):
        """the pos offsets are applied in the local frame"""
        offset_human_data = {}
        for body_name in human_data.keys():   #notice: quat: 绝对旋转，不是相对旋转  
            pos, quat = human_data[body_name] #pos: position of joint; quat: quaternion of joint;
            offset_human_data[body_name] = [pos, quat]  
            # apply rotation offset first
            # notice: quat is from human_data, rot_offsets is from ik config (eg: smplx_to_g1.json)
            updated_quat = (R.from_quat(quat, scalar_first=True) * rot_offsets[body_name]).as_quat(scalar_first=True)
            offset_human_data[body_name][1] = updated_quat
            
            local_offset = pos_offsets[body_name] # notice:  pos_offsets is from ik config (eg: smplx_to_g1.json)
            # compute the global position offset using the updated rotation
            #? 这个参量的作用是?
            global_pos_offset = R.from_quat(updated_quat, scalar_first=True).apply(local_offset) #R is a matrix, local_offset is a vector, global_pos_offset is a vector
            # 将全局位置偏移加到原始位置上
            offset_human_data[body_name][0] = pos + global_pos_offset
           
        return offset_human_data
            
    # high priority: offset the human data to the ground
    def offset_human_data_to_ground(self, human_data):
        """find the lowest point of the human data and offset the human data to the ground"""
        offset_human_data = {}
        ground_offset = 0.1
        lowest_pos = np.inf

        for body_name in human_data.keys():
            # only consider the foot/Foot
            if "Foot" not in body_name and "foot" not in body_name:
                continue
            pos, quat = human_data[body_name]
            if pos[2] < lowest_pos:
                lowest_pos = pos[2]
                lowest_body_name = body_name
        for body_name in human_data.keys():
            pos, quat = human_data[body_name]
            offset_human_data[body_name] = [pos, quat]
            offset_human_data[body_name][0] = pos - np.array([0, 0, lowest_pos]) + np.array([0, 0, ground_offset])
        return offset_human_data

    def set_ground_offset(self, ground_offset):
        self.ground_offset = ground_offset
        # print(f"Ground offset set to: {self.ground_offset}")

    # high priority: offset the human data to the ground
    def apply_ground_offset(self, human_data): #human_data是一帧的数据
        for body_name in human_data.keys():
            pos, quat = human_data[body_name]
            #human_data[body_name][0]是把关节中的position这个三元数组提取出来
            human_data[body_name][0] = pos - np.array([0, 0, self.ground_offset]) 
        return human_data
    
    