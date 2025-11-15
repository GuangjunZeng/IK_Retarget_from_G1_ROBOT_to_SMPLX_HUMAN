
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

class GeneralMotionRetargeting:
    """General Motion Retargeting (GMR).
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
        print(f"ratio: {ratio}") #000005.npz: ratio is 0.9497133053910921
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

        bofore_pelvis = human_data['pelvis']
        # print(f"bofore_pelvis: {bofore_pelvis}")
        ##000005.npz last frame:  [array([-0.32628706,  0.29134345,  0.97025055], dtype=float32), array([0.01704654, 0.00681596, 0.67657403, 0.73614573])]
        #? 也可以自己认为设定human_data['left_shoulder']的值, ，这样数值差异的幅度更大更明显
        before_check1_left_shoulder = human_data['left_shoulder']
        print(f"before_check1_left_shoulder: {before_check1_left_shoulder}")
        #000005.npz last frame: [array([-0.482392  ,  0.25966972,  1.3952072 ], dtype=float32), array([ 0.52543509, -0.26132338,  0.66067818,  0.4681158 ])]




        # scale the human data in the Global Coordinate System
        human_data = self.scale_human_data(human_data, self.human_root_name, self.human_scale_table)
        # print(f"after scale_human_data(), human_data_left_shoulder: {human_data['left_shoulder']}")
        # offset the human data according to the ik config (eg: smplx_to_g1.json)
        human_data = self.offset_human_data(human_data, self.pos_offsets1, self.rot_offsets1)
        # print(f"after offset_human_data(), human_data_left_shoulder: {human_data['left_shoulder']}")


        pelvis = human_data['pelvis']
        pelvis_pos = np.asarray(pelvis[0], dtype=np.float64).reshape(-1)
        pelvis_quat = np.asarray(pelvis[1], dtype=np.float64).reshape(-1)
        # print("pelvis: " + ", ".join(f"{value:.15f}" for value in np.concatenate([pelvis_pos, pelvis_quat])))
        ##000005.npz last frame: [array([-0.32628706,  0.29134345,  0.97025055]), array([ 0.71829113,  0.02467056, -0.03490114,  0.69442864])]
        # [array([ -0.326287060976028, 0.291343450546265, 0.970250546932220]), array([ 0.017046535297107, 0.006815957199062, 0.676574028535944, 0.736145734398067])]
        #  -0.326287060976028, 0.291343450546265, 0.970250546932220,                 0.718291127715090, 0.024670563882039, -0.034901141980084, 0.694428635218921
        check1_left_shoulder = human_data['left_shoulder']
        left_shoulder_pos = np.asarray(check1_left_shoulder[0], dtype=np.float64).reshape(-1)
        left_shoulder_quat = np.asarray(check1_left_shoulder[1], dtype=np.float64).reshape(-1)
        print("check1_left_shoulder: " + ", ".join(f"{value:.15f}" for value in np.concatenate([left_shoulder_pos, left_shoulder_quat])))
        # 000005.npz last frame: [array([-0.444891021011150, 0.267278680737261, 1.293120111716746]), array([0.838708736211145, 0.146224318973807, 0.095631307160299, 0.515791389453685])]
        # [array([-0.444891021011150, 0.267278680737261, 1.293120111716746]), array([ 0.525435089028519, -0.261323381639568, 0.660678180602081, 0.468115796681096 ])]
        #  -0.444891021011150, 0.267278680737261, 1.293120111716746,            0.838708736211145, 0.146224318973807, 0.095631307160299, 0.515791389453685 



       
        #human_data: dict{body_name: [pos, quat]}
        #robot_data: dict{robot_body_name: BodyPose(pos, quat)}

        #! PHC-style: compute a constant ground offset from the first frame (once)
        if not self._ground_offset_initialized: #self._ground_offset_initialized is false -> not self._ground_offset_initialized is true
            try:
                min_z = np.inf
                for body_name in human_data.keys(): #human_data是当前帧的人体数据，但是但由于 _ground_offset_initialized 标志的作用，地面偏移的计算只会在第一次调用 update_targets 时触发，所以实际上用的是"第一次传入的那一帧"（通常就是第一帧）。
                    pos, _ = human_data[body_name]
                    if pos[2] < min_z:
                        min_z = pos[2]
                # set ground offset so that the lowest point becomes z=0 in the first frame
                self.set_ground_offset(float(min_z))
            except Exception:
                print("Warning: Failed to compute constant ground offset.")
                # fallback: keep default ground_offset
                pass
            self._ground_offset_initialized = True
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
        
        #! keep root position in world unchanged (PHC-style): do NOT scale root
        scaled_root_pos = root_pos
        
        # scale other body parts in local frame
        for body_name in human_data.keys():
            if body_name not in human_scale_table:
                continue
            if body_name == human_root_name:
                # print(f"in scale_human_data(), human_root_pose: {human_data[human_root_name]}")
                continue
            else:
                # transform to local frame (only position)
                human_data_local[body_name] = (human_data[body_name][0] - root_pos) * human_scale_table[body_name]

                if body_name == "left_shoulder":
                    # print(f"in scale_human_data(), human_scale_table[left_shoulder]: {human_scale_table[body_name]}")
                    pass
            
        # transform the human data back to the global frame
        human_data_global = {human_root_name: (scaled_root_pos, root_quat)}
        for body_name in human_data_local.keys():
            human_data_global[body_name] = (human_data_local[body_name] + scaled_root_pos, human_data[body_name][1]) #notics: human_data is a dict, key is body name, value is the (pos, quat)

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
    
    def _verify_reverse_consistency(self, human_data):
        """Verify that scale_robot_data + offset_robot_data can recover the input human_data"""
        if getattr(self, "_reverse_verification_done", False):
            return
        self._reverse_verification_done = True
        
        try:
            from .reverse_motion_retarget import RobotToSMPLXRetargeting
            from .robot import BodyPose
        except Exception as exc:
            print(f"[GMR] Reverse verification skipped (import error): {exc}")
            return
        
        try:
            smplx_models = Path(__file__).resolve().parents[1] / "assets" / "body_models"
            reverse = RobotToSMPLXRetargeting(
                robot_type=self._tgt_robot,
                smplx_model_path=smplx_models,
                gender="neutral",
                verbose=False,
            )
        except Exception as exc:
            print(f"[GMR] Reverse verification skipped (initialization error): {exc}")
            return
        
        # Backup original human_data
        human_data_backup = copy.deepcopy(human_data)
        
        # Convert human_data (dict{body_name: [pos, quat]}) to robot_data (dict{robot_body_name: BodyPose})
        robot_data_input = {}
        for table in (self.ik_match_table1, self.ik_match_table2):
            for robot_name, entry in table.items():
                human_name = entry[0]
                if human_name not in human_data_backup:
                    continue
                pos, quat = human_data_backup[human_name]
                robot_data_input[robot_name] = BodyPose(
                    pos=np.array(pos, dtype=np.float64),
                    rot=np.array(quat, dtype=np.float64)
                )
        
        # Add root
        if self.human_root_name in human_data_backup:
            pos, quat = human_data_backup[self.human_root_name]
            robot_data_input[self.robot_root_name] = BodyPose(
                pos=np.array(pos, dtype=np.float64),
                rot=np.array(quat, dtype=np.float64)
            )
        
        # Apply reverse scale_robot_data + offset_robot_data
        robot_data_scaled = reverse.scale_robot_data(copy.deepcopy(robot_data_input))
        robot_data_final = reverse.offset_robot_data(copy.deepcopy(robot_data_scaled))
        
        # Convert back to human_data format for comparison
        human_data_recovered = {}
        for table in (self.ik_match_table1, self.ik_match_table2):
            for robot_name, entry in table.items():
                human_name = entry[0]
                if robot_name not in robot_data_final:
                    continue
                pose = robot_data_final[robot_name]
                human_data_recovered[human_name] = [pose.pos.copy(), pose.rot.copy()]
        
        if self.robot_root_name in robot_data_final:
            pose = robot_data_final[self.robot_root_name]
            human_data_recovered[self.human_root_name] = [pose.pos.copy(), pose.rot.copy()]
        
        # Compare
        max_pos_err = 0.0
        max_rot_err = 0.0
        for body_name in human_data_backup.keys():
            if body_name not in human_data_recovered:
                continue
            orig_pos = np.asarray(human_data_backup[body_name][0], dtype=np.float64)
            orig_quat = np.asarray(human_data_backup[body_name][1], dtype=np.float64)
            rec_pos = np.asarray(human_data_recovered[body_name][0], dtype=np.float64)
            rec_quat = np.asarray(human_data_recovered[body_name][1], dtype=np.float64)
            
            pos_err = float(np.linalg.norm(orig_pos - rec_pos))
            quat_direct = np.linalg.norm(orig_quat - rec_quat)
            quat_flipped = np.linalg.norm(orig_quat + rec_quat)
            quat_err = float(min(quat_direct, quat_flipped))
            
            max_pos_err = max(max_pos_err, pos_err)
            max_rot_err = max(max_rot_err, quat_err)
        
        print("[GMR] Reverse verification (offset_human_data -> scale_robot_data + offset_robot_data -> compare):")
        print(f"  max position error: {max_pos_err:.6e}")
        print(f"  max quaternion error: {max_rot_err:.6e}")
