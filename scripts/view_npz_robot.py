#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# python3 scripts/view_npz_robot.py ../server3_data/locomotion/robot/ik_based/npz/000001.npz

# python3 scripts/view_npz_robot.py ../data/locomotion/robot/ik_based/npz/000001.npz

import argparse
import numpy as np
import os
from typing import Dict, Any


def print_separator(title="", char="=", width=80):
    """打印分隔线"""
    if title:
        side_len = (width - len(title) - 2) // 2
        print(f"{char * side_len} {title} {char * side_len}")
    else:
        print(char * width)


def print_array_stats(arr: np.ndarray, name: str, indent="  "):
    """打印数组的详细统计信息"""
    print(f"{indent}📊 统计信息:")
    print(f"{indent}   形状: {arr.shape}")
    print(f"{indent}   数据类型: {arr.dtype}")
    
    if np.issubdtype(arr.dtype, np.number):
        print(f"{indent}   最小值: {np.min(arr):.6f}")
        print(f"{indent}   最大值: {np.max(arr):.6f}")
        print(f"{indent}   平均值: {np.mean(arr):.6f}")
        print(f"{indent}   标准差: {np.std(arr):.6f}")
        print(f"{indent}   中位数: {np.median(arr):.6f}")


def print_per_joint_stats(joints: np.ndarray, joint_names: np.ndarray):
    """打印每个关节的详细统计"""
    print_separator("关节详细统计", "-", 80)
    print(f"共 {joints.shape[1]} 个关节，{joints.shape[0]} 帧数据\n")
    
    # 按身体部位分组
    body_parts = {
        "左腿": ["left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee", "left_ankle_pitch", "left_ankle_roll"],
        "右腿": ["right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee", "right_ankle_pitch", "right_ankle_roll"],
        "腰部": ["waist_yaw", "waist_roll", "waist_pitch"],
        "左臂": ["left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow", "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw"],
        "右臂": ["right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow", "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw"],
    }
    
    for part_name, keywords in body_parts.items():
        print(f"\n🔹 {part_name}:")
        print(f"{'关节名':<35} {'最小值':>10} {'最大值':>10} {'平均值':>10} {'标准差':>10} {'范围':>10}")
        print("-" * 95)
        
        for i, jname in enumerate(joint_names):
            jname_str = str(jname)
            # 检查关节是否属于当前身体部位
            if any(kw in jname_str for kw in keywords):
                joint_data = joints[:, i]
                min_val = np.min(joint_data)
                max_val = np.max(joint_data)
                mean_val = np.mean(joint_data)
                std_val = np.std(joint_data)
                range_val = max_val - min_val
                
                print(f"{jname_str:<35} {min_val:>10.4f} {max_val:>10.4f} {mean_val:>10.4f} {std_val:>10.4f} {range_val:>10.4f}")


def print_root_details(root_pos: np.ndarray, root_quat: np.ndarray):
    """打印根位置和旋转的详细信息"""
    print_separator("根节点详细信息", "-", 80)
    
    # 根位置
    print("\n🔹 根位置 (root_pos):")
    print(f"  形状: {root_pos.shape}")
    print(f"\n  {'轴':<10} {'最小值':>12} {'最大值':>12} {'平均值':>12} {'标准差':>12} {'位移范围':>12}")
    print("  " + "-" * 72)
    
    axes = ['X', 'Y', 'Z']
    for i, axis in enumerate(axes):
        col = root_pos[:, i]
        print(f"  {axis:<10} {np.min(col):>12.6f} {np.max(col):>12.6f} {np.mean(col):>12.6f} "
              f"{np.std(col):>12.6f} {np.max(col)-np.min(col):>12.6f}")
    
    # 运动轨迹分析
    print(f"\n  📍 起始位置: [{root_pos[0, 0]:.4f}, {root_pos[0, 1]:.4f}, {root_pos[0, 2]:.4f}]")
    print(f"  📍 结束位置: [{root_pos[-1, 0]:.4f}, {root_pos[-1, 1]:.4f}, {root_pos[-1, 2]:.4f}]")
    
    displacement = root_pos[-1] - root_pos[0]
    total_displacement = np.linalg.norm(displacement)
    print(f"  📍 总位移: {total_displacement:.4f} 米")
    print(f"  📍 位移向量: [{displacement[0]:.4f}, {displacement[1]:.4f}, {displacement[2]:.4f}]")
    
    # 速度分析（相邻帧的位移）
    if root_pos.shape[0] > 1:
        velocities = np.diff(root_pos, axis=0)
        speeds = np.linalg.norm(velocities, axis=1)
        print(f"\n  🚀 平均速度: {np.mean(speeds):.6f} 米/帧")
        print(f"  🚀 最大速度: {np.max(speeds):.6f} 米/帧")
    
    # 根旋转
    print("\n🔹 根旋转 (root_quat, xyzw格式):")
    print(f"  形状: {root_quat.shape}")
    print(f"\n  {'分量':<10} {'最小值':>12} {'最大值':>12} {'平均值':>12} {'标准差':>12}")
    print("  " + "-" * 60)
    
    quat_names = ['QX', 'QY', 'QZ', 'QW']
    for i, qname in enumerate(quat_names):
        col = root_quat[:, i]
        print(f"  {qname:<10} {np.min(col):>12.6f} {np.max(col):>12.6f} {np.mean(col):>12.6f} {np.std(col):>12.6f}")
    
    # 检查四元数归一化
    quat_norms = np.linalg.norm(root_quat, axis=1)
    print(f"\n  ✓ 四元数模长: 最小={np.min(quat_norms):.6f}, 最大={np.max(quat_norms):.6f}")
    if np.allclose(quat_norms, 1.0, atol=1e-4):
        print(f"  ✓ 四元数归一化检查: ✅ 通过")
    else:
        print(f"  ✓ 四元数归一化检查: ⚠️  部分四元数未归一化")


def print_frame_preview(data: Dict[str, Any], num_frames: int = 3):
    """打印第一帧的完整数据"""
    print_separator("第一帧完整数据", "-", 80)
    
    joints = data.get('joints')
    root_pos = data.get('root_pos')
    root_quat = data.get('root_quat')
    
    if joints is None or joints.shape[0] == 0:
        print("❌ 没有关节数据")
        return
    
    frame_idx = 0
    print(f"\n📍 第 {frame_idx} 帧 (完整数据):\n")
    
    # 打印根位置
    if root_pos is not None:
        print("  根位置 (root_pos):")
        print(f"    X: {root_pos[frame_idx, 0]:12.6f}")
        print(f"    Y: {root_pos[frame_idx, 1]:12.6f}")
        print(f"    Z: {root_pos[frame_idx, 2]:12.6f}")
        print()
    
    # 打印根旋转（四元数）
    if root_quat is not None:
        print("  根旋转 (root_quat, xyzw格式):")
        print(f"    X: {root_quat[frame_idx, 0]:12.6f}")
        print(f"    Y: {root_quat[frame_idx, 1]:12.6f}")
        print(f"    Z: {root_quat[frame_idx, 2]:12.6f}")
        print(f"    W: {root_quat[frame_idx, 3]:12.6f}")
        print()
    
    # 打印所有关节角度
    print(f"  关节角度 (joints, 共 {joints.shape[1]} 个DOF):")
    num_dofs = joints.shape[1]
    
    # 每行打印5个关节
    for i in range(0, num_dofs, 5):
        end_idx = min(i + 5, num_dofs)
        joint_values = joints[frame_idx, i:end_idx]
        
        # 打印索引号
        indices_str = "    " + "  ".join([f"DOF[{j:2d}]" for j in range(i, end_idx)])
        print(indices_str)
        
        # 打印数值
        values_str = "    " + "  ".join([f"{val:8.4f}" for val in joint_values])
        print(values_str)
        print()


def check_data_validity(data: Dict[str, Any]):
    """检查数据的有效性"""
    print_separator("数据有效性检查", "-", 80)
    
    checks_passed = 0
    total_checks = 0
    
    # 检查关节数据
    joints = data.get('joints')
    if joints is not None:
        total_checks += 1
        if not np.any(np.isnan(joints)) and not np.any(np.isinf(joints)):
            print("✅ 关节数据: 无NaN或Inf值")
            checks_passed += 1
        else:
            print("❌ 关节数据: 包含NaN或Inf值")
            print(f"   NaN数量: {np.sum(np.isnan(joints))}")
            print(f"   Inf数量: {np.sum(np.isinf(joints))}")
    
    # 检查根位置
    root_pos = data.get('root_pos')
    if root_pos is not None:
        total_checks += 1
        if not np.any(np.isnan(root_pos)) and not np.any(np.isinf(root_pos)):
            print("✅ 根位置数据: 无NaN或Inf值")
            checks_passed += 1
        else:
            print("❌ 根位置数据: 包含NaN或Inf值")
    
    # 检查根旋转
    root_quat = data.get('root_quat')
    if root_quat is not None:
        total_checks += 2
        if not np.any(np.isnan(root_quat)) and not np.any(np.isinf(root_quat)):
            print("✅ 根旋转数据: 无NaN或Inf值")
            checks_passed += 1
        else:
            print("❌ 根旋转数据: 包含NaN或Inf值")
        
        # 检查四元数归一化
        quat_norms = np.linalg.norm(root_quat, axis=1)
        if np.allclose(quat_norms, 1.0, atol=1e-3):
            print("✅ 根旋转归一化: 所有四元数已归一化")
            checks_passed += 1
        else:
            print("⚠️  根旋转归一化: 部分四元数未正确归一化")
            print(f"   模长范围: [{np.min(quat_norms):.6f}, {np.max(quat_norms):.6f}]")
    
    # 检查数据一致性
    if 'full_data' in data and 'joints' in data:
        total_checks += 1
        full_data = data['full_data']
        expected_cols = 7 + joints.shape[1] if data.get('include_base') else joints.shape[1]
        if full_data.shape[1] == expected_cols:
            print(f"✅ 数据维度一致: full_data列数 = {full_data.shape[1]}")
            checks_passed += 1
        else:
            print(f"❌ 数据维度不一致: full_data={full_data.shape[1]}, 期望={expected_cols}")
    
    print(f"\n总结: {checks_passed}/{total_checks} 项检查通过")
    return checks_passed == total_checks


def view_robot_npz(npz_path: str, verbose: bool = True, preview_frames: int = 3):
    """查看机器人NPZ文件的完整结构"""
    
    print_separator(f"机器人运动数据查看器", "=", 80)
    print(f"📂 文件路径: {npz_path}")
    print(f"📁 文件大小: {os.path.getsize(npz_path) / 1024:.2f} KB")
    print()
    
    # 加载数据
    data = np.load(npz_path, allow_pickle=True)
    
    # 基本信息
    print_separator("基本信息", "-", 80)
    print(f"📋 包含的键: {list(data.keys())}")
    print()
    
    # 解析所有数据
    data_dict = {}
    for key in data.keys():
        data_dict[key] = data[key]
        
    # 元数据
    print("📌 元数据:")
    if 'num_frames' in data_dict:
        print(f"  总帧数: {data_dict['num_frames']}")
    if 'num_joints' in data_dict:
        print(f"  关节数: {data_dict['num_joints']}")
    if 'include_base' in data_dict:
        print(f"  包含base: {data_dict['include_base']}")
    
    # 数据维度概览
    print_separator("数据维度概览", "-", 80)
    
    # 首先显示最重要的信息
    if 'full_data' in data_dict:
        full_data = data_dict['full_data']
        print(f"\n⭐ 核心数据矩阵: full_data")
        print(f"   📏 总共有 {full_data.shape[0]} 行（帧数）")
        print(f"   📏 总共有 {full_data.shape[1]} 列")
        
        if data_dict.get('include_base', False):
            print(f"\n   列结构分解:")
            print(f"      列 0-2   (3列):  根位置 XYZ")
            print(f"      列 3-6   (4列):  根四元数 XYZW") 
            print(f"      列 7-35  (29列): 关节角度")
            print(f"      ────────────────")
            print(f"      总计: 3 + 4 + 29 = 36 列")
        else:
            print(f"   列结构: 29列关节角度")
    
    print(f"\n   详细分解:")
    for key in ['joints', 'root_pos', 'root_quat', 'joint_names', 'full_data']:
        if key in data_dict:
            val = data_dict[key]
            if isinstance(val, np.ndarray):
                shape_str = f"{val.shape}"
                if val.ndim == 2:
                    shape_str = f"{val.shape} = {val.shape[0]}行 × {val.shape[1]}列"
                elif val.ndim == 1:
                    shape_str = f"{val.shape} = {val.shape[0]}个元素"
                print(f"   {key:<20} {shape_str:<35} 类型: {val.dtype}")
    
    print()
    
    # 关节名称列表
    if 'joint_names' in data_dict:
        print_separator("关节名称列表", "-", 80)
        joint_names = data_dict['joint_names']
        print(f"共 {len(joint_names)} 个关节:\n")
        
        # 分列显示
        for i in range(0, len(joint_names), 2):
            left = f"  [{i:2d}] {joint_names[i]}"
            if i + 1 < len(joint_names):
                right = f"[{i+1:2d}] {joint_names[i+1]}"
                print(f"{left:<45} {right}")
            else:
                print(left)
        print()
    
    # 详细统计
    if verbose:
        # 根节点详细信息
        if 'root_pos' in data_dict and 'root_quat' in data_dict:
            print_root_details(data_dict['root_pos'], data_dict['root_quat'])
            print()
        
        # 关节详细统计
        if 'joints' in data_dict and 'joint_names' in data_dict:
            print_per_joint_stats(data_dict['joints'], data_dict['joint_names'])
            print()
        
        # full_data统计
        if 'full_data' in data_dict:
            print_separator("完整数据矩阵 (full_data)", "-", 80)
            full_data = data_dict['full_data']
            print(f"形状: {full_data.shape}")
            
            if data_dict.get('include_base', False):
                print("结构: [root_pos(3) | root_quat(4) | joints(29)]")
                print("       列0-2: 根位置XYZ")
                print("       列3-6: 根四元数XYZW")
                print("       列7-35: 29个关节角度")
            else:
                print("结构: [joints(29)]")
                print("       列0-28: 29个关节角度")
            
            print_array_stats(full_data, "full_data")
            print()
    
    # 数据预览
    if preview_frames > 0:
        print_frame_preview(data_dict, preview_frames)
        print()
    
    # 数据有效性检查
    check_data_validity(data_dict)
    
    print()
    print_separator("", "=", 80)
    print("✅ 数据查看完成")
    
    data.close()
    return data_dict


def main():
    parser = argparse.ArgumentParser(
        description="查看机器人运动数据NPZ文件的完整结构和详细信息",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("npz_path", help="NPZ文件路径")
    parser.add_argument("--simple", action="store_true", help="简洁模式，不显示详细统计")
    parser.add_argument("--preview", type=int, default=3, help="预览的帧数 (默认: 3)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.npz_path):
        print(f"❌ 文件不存在: {args.npz_path}")
        return 1
    
    try:
        view_robot_npz(args.npz_path, verbose=not args.simple, preview_frames=args.preview)
        return 0
    except Exception as e:
        print(f"❌ 查看文件时出错: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())

