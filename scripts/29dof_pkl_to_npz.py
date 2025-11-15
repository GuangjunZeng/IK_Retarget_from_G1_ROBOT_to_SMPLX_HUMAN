#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# include_base=False  这个选项有吗

import argparse, os
import numpy as np
import joblib
import xml.etree.ElementTree as ET

DESIRED_ORDER = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

def parse_joint_order_from_mjcf(xml_path: str):
    """
    读取 MJCF/XML 中 <actuator><motor joint="..."> 的顺序，作为 pkl 中 DOF 的当前顺序。
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    # 找 <actuator> 节点下的 <motor>
    motors = []
    for act in root.findall(".//actuator"):
        for m in act.findall("motor"):
            j = m.get("joint")
            if j is not None:
                motors.append(j)
    if not motors:
        raise RuntimeError(f"在 {xml_path} 中没有找到 <actuator><motor joint='...'> 定义")
    return motors

def axis_angle_to_quat_xyzw(axis_angle: np.ndarray) -> np.ndarray:
    """
    把根节点的 axis-angle(形状 Nx3) 转为四元数 (xyzw)。若失败则返回单位四元数。
    """
    try:
        from scipy.spatial.transform import Rotation as R
        q = R.from_rotvec(axis_angle).as_quat()  # SciPy 返回顺序是 (x,y,z,w)
        return q
    except Exception:
        # 兜底：单位四元数
        q = np.zeros((axis_angle.shape[0], 4), dtype=np.float64)
        q[:, 3] = 1.0
        return q

def reorder_and_export_one(data_dict, present_order, out_npz, include_base=False):
    """
    data_dict: 单条序列的字典，包含至少 data_dict['dof'] (NxJ)
    present_order: 当前 J 列对应的关节名顺序（长度应为 J)
    out_npz: 输出文件路径
    include_base: 是否在前面加 7 列 (base_x,y,z, qx,qy,qz,qw)
    """
    dof = np.asarray(data_dict["dof_pos"])      # (N, J_present)
    if dof.ndim != 2:
        raise ValueError(f"'dof' 应为二维 (N,J)，实际形状: {dof.shape}")
    N, J_present = dof.shape

    # --- 重排关节到目标顺序 ---
    out_joints = np.zeros((N, len(DESIRED_ORDER)), dtype=np.float64)
    name_to_idx = {n: i for i, n in enumerate(present_order)}
    missing = []
    for out_col, name in enumerate(DESIRED_ORDER):
        if name in name_to_idx:
            out_joints[:, out_col] = dof[:, name_to_idx[name]]
        else:
            missing.append(name)  # 缺失的用 0 填

    # 准备输出数据字典
    output_data = {}
    
    # 添加关节数据
    output_data['joints'] = out_joints
    output_data['joint_names'] = np.array(DESIRED_ORDER, dtype='U50')  # 关节名称
    
    if include_base:
        # ----------- 根位置 (N,3) -----------
        base_pos = None
        if "root_pos" in data_dict:
            base_pos = np.asarray(data_dict["root_pos"], dtype=np.float64)
            # 兼容 (3,) / (N,3)
            if base_pos.ndim == 1 and base_pos.shape[0] == 3:
                base_pos = np.repeat(base_pos[None, :], N, axis=0)
        if base_pos is None or base_pos.shape != (N, 3):
            base_pos = np.zeros((N, 3), dtype=np.float64)

        # ----------- 根姿态四元数 (N,4) -----------
        base_quat_xyzw = None
        # 1) 优先用 pkl 已给的四元数（SciPy as_quat() 默认 xyzw）
        if "root_rot" in data_dict:
            q = np.asarray(data_dict["root_rot"], dtype=np.float64)
            if q.ndim == 1 and q.shape[0] == 4:
                q = np.repeat(q[None, :], N, axis=0)
            if q.shape == (N, 4):
                base_quat_xyzw = q

        # 2) 退化：从 pose_aa 的根关节 (axis-angle) 还原
        if base_quat_xyzw is None:
            if "pose_aa" in data_dict:
                pose_aa = np.asarray(data_dict["pose_aa"], dtype=np.float64)
                if pose_aa.ndim == 3:            # (N, K, 3)
                    root_aa = pose_aa[:, 0, :]
                else:                             # 兼容 (N,*,3) 拉直
                    root_aa = pose_aa.reshape(N, -1, 3)[:, 0, :]
                base_quat_xyzw = axis_angle_to_quat_xyzw(root_aa)
            else:
                base_quat_xyzw = np.zeros((N, 4), dtype=np.float64); base_quat_xyzw[:, 3] = 1.0

        # 添加基础数据
        output_data['root_pos'] = base_pos
        output_data['root_quat'] = base_quat_xyzw
        
        # 为了兼容性，也提供合并的完整数据
        out_all = np.concatenate([base_pos, base_quat_xyzw, out_joints], axis=1)
        output_data['full_data'] = out_all  # 完整数据矩阵 (N, 7+29)
    else:
        output_data['full_data'] = out_joints  # 仅关节数据 (N, 29)

    # 添加元数据
    output_data['num_frames'] = N
    output_data['num_joints'] = len(DESIRED_ORDER)
    output_data['include_base'] = include_base
    
    # 保存 NPZ（保持原始精度）
    os.makedirs(os.path.dirname(out_npz) or ".", exist_ok=True)
    np.savez_compressed(out_npz, **output_data)
    
    print(f"[OK] 保存至: {out_npz}")
    print(f"     形状: {output_data['full_data'].shape}")
    print(f"     列数: {'7+' if include_base else ''}{len(DESIRED_ORDER)}")
    print(f"     精度: {output_data['full_data'].dtype}")
    print(f"     包含数据: {list(output_data.keys())}")

    if missing:
        print("[WARN] 下面这些关节在 pkl 中未找到，已用 0 填充：")
        for n in missing:
            print("   -", n)

def main():
    ap = argparse.ArgumentParser("Export pkl (retarget) to NPZ in a fixed joint order")
    ap.add_argument("--pkl", required=True, help="pkl 文件路径(joblib.dump 保存的）")
    ap.add_argument("--out", required=True, help="输出 NPZ 路径。如果 pkl 含多条序列，会在文件名里追加键名。")
    ap.add_argument("--mjcf", default=None, help="可选:MJCF/XML 路径，用于解析当前 DOF 顺序（推荐提供）")
    ap.add_argument("--include-base", action="store_true",
                    help="在关节列前添加根位置(xyz)与根姿态四元数(qx qy qz qw)共 7 列")
    args = ap.parse_args()

    # 加载 pkl
    obj = joblib.load(args.pkl)

    # 解析"当前顺序"
    present_order = None
    if args.mjcf:
        present_order = parse_joint_order_from_mjcf(args.mjcf)
        print(f"[INFO] 从 MJCF 读取到 {len(present_order)} 个关节（按 <actuator><motor> 顺序）")
    else:
        print("[INFO] 未提供 --mjcf,将假定 pkl 的列顺序已经与目标顺序一致（不重排）。")
        present_order = DESIRED_ORDER[:]  # 直接视为已对齐

    # # 处理单条 / 多条序列
    # if isinstance(obj, dict) and "dof" not in obj:
    #     # 多条序列：{key: data_dict, ...}
    #     for k, data_dict in obj.items():
    #         out_npz = args.out
    #         base, ext = os.path.splitext(out_npz)
    #         out_npz_k = f"{base}_{k}{ext or '.npz'}"
    #         reorder_and_export_one(data_dict, present_order, out_npz_k, include_base=args.include_base)
    # else:
    #     # 单条序列：data_dict
    reorder_and_export_one(obj, present_order, args.out, include_base=args.include_base)

if __name__ == "__main__":
    main()
