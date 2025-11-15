#!/usr/bin/env python3
"""可视化 SMPL-X 关节层级结构和 parents 数组的作用"""
#可以删除！！可以删除！！

import smplx
import numpy as np
from smplx.joint_names import JOINT_NAMES


def print_hierarchy_tree(parents, joint_names, max_depth=3):
    """打印关节树形结构"""
    
    def print_node(idx, depth=0, prefix=""):
        if depth > max_depth:
            return
        
        indent = "  " * depth
        if depth == 0:
            print(f"{prefix}[{idx:2d}] {joint_names[idx]} (ROOT)")
        else:
            print(f"{prefix}{indent}├─ [{idx:2d}] {joint_names[idx]}")
        
        # 找到所有以当前节点为父节点的子节点
        children = [i for i, p in enumerate(parents) if p == idx]
        for child in children:
            print_node(child, depth + 1, prefix)
    
    print("\n" + "="*70)
    print("SMPL-X 关节层级树 (树形结构)")
    print("="*70)
    print_node(0)


def explain_parents_usage():
    """演示 parents 数组在计算局部旋转中的作用"""
    
    print("\n" + "="*70)
    print("Parents 数组在 compute_local_rotations 中的作用")
    print("="*70)
    
    # 创建模型
    model = smplx.create('assets/body_models', 'smplx', gender='neutral', use_pca=False)
    parents = model.parents.detach().cpu().numpy().astype(int)
    joint_names = JOINT_NAMES[:len(parents)]
    
    print("\n示例：计算 left_knee 的局部旋转")
    print("-" * 70)
    
    # 找到 left_knee 的索引
    left_knee_idx = joint_names.index('left_knee')
    parent_idx = parents[left_knee_idx]
    
    print(f"1. 关节名称: {joint_names[left_knee_idx]}")
    print(f"2. 关节索引: i = {left_knee_idx}")
    print(f"3. 父节点索引: parents[{left_knee_idx}] = {parent_idx}")
    print(f"4. 父节点名称: {joint_names[parent_idx]}")
    
    print(f"\n在 compute_local_rotations 函数中:")
    print(f"   parent_idx = self.smplx_parents[{left_knee_idx}]  # = {parent_idx}")
    print(f"   parent_name = self.smplx_joint_names[{parent_idx}]  # = '{joint_names[parent_idx]}'")
    print(f"   joint_name = self.smplx_joint_names[{left_knee_idx}]  # = '{joint_names[left_knee_idx]}'")
    print(f"\n   parent_R = global_rots['{joint_names[parent_idx]}']  # 父节点的世界旋转")
    print(f"   joint_R = global_rots['{joint_names[left_knee_idx]}']  # 当前关节的世界旋转")
    print(f"   local_R = parent_R.inv() * joint_R  # 计算局部旋转")
    
    body_pose_start = (left_knee_idx - 1) * 3
    body_pose_end = body_pose_start + 3
    print(f"\n   存储位置: body_pose[{body_pose_start}:{body_pose_end}] = local_R.as_rotvec()")
    
    print("\n" + "="*70)
    print("为什么 body_pose 只有 63 维？")
    print("="*70)
    print("SMPL-X 身体模型有 22 个主要关节 (索引 0-21):")
    print("  - 关节 0 (pelvis): 全局旋转存储在 root_orient/global_orient")
    print("  - 关节 1-21: 局部旋转存储在 body_pose")
    print("  - 每个关节 3 个轴角参数: 21 * 3 = 63")
    print("\n手部、面部等关节 (索引 22-54) 的旋转存储在其他参数中:")
    print("  - left_hand_pose: 45 维 (15 个手指关节 * 3)")
    print("  - right_hand_pose: 45 维")
    print("  - jaw_pose, leye_pose, reye_pose: 各 3 维")


def show_body_pose_mapping():
    """显示 body_pose 数组与关节的对应关系"""
    
    print("\n" + "="*70)
    print("Body Pose 数组映射关系")
    print("="*70)
    
    model = smplx.create('assets/body_models', 'smplx', gender='neutral', use_pca=False)
    parents = model.parents.detach().cpu().numpy().astype(int)
    joint_names = JOINT_NAMES[:len(parents)]
    
    print(f"\n{'索引':^6} | {'关节名称':^20} | {'父节点':^20} | {'body_pose 位置':^20}")
    print("-" * 75)
    
    # 只显示身体关节 (1-21)
    for i in range(1, min(22, len(parents))):
        parent_idx = parents[i]
        body_pose_start = (i - 1) * 3
        body_pose_end = body_pose_start + 3
        print(f"{i:^6} | {joint_names[i]:^20} | {joint_names[parent_idx]:^20} | [{body_pose_start:2d}:{body_pose_end:2d}]")


def demonstrate_forward_kinematics():
    """演示如何用 parents 数组进行正向运动学"""
    
    print("\n" + "="*70)
    print("正向运动学：从局部旋转重建世界旋转")
    print("="*70)
    
    print("\n在 smplx_to_robot_batch.py 的 manual_downsample_smplx_data 中:")
    print("-" * 70)
    print("""
for i, joint_name in enumerate(joint_names):
    if i == 0:  # pelvis (根节点)
        rot = R.from_rotvec(single_global_orient)
    else:
        # 递归计算：世界旋转 = 父节点世界旋转 * 局部旋转
        rot = joint_orientations[parents[i]] * R.from_rotvec(
            single_full_body_pose[i].squeeze()
        )
    joint_orientations.append(rot)
    """)
    
    print("这里 parents[i] 告诉我们：")
    print("  - 当前关节 i 的父节点索引")
    print("  - 从 joint_orientations[parents[i]] 获取父节点的世界旋转")
    print("  - 用于递归计算整个骨骼链的世界姿态")


def main():
    """主函数"""
    
    print("\n" + "="*70)
    print("SMPL-X Parents 数组详解")
    print("="*70)
    
    # 创建模型
    model = smplx.create('assets/body_models', 'smplx', gender='neutral', use_pca=False)
    parents = model.parents.detach().cpu().numpy().astype(int)
    joint_names = JOINT_NAMES[:len(parents)]
    
    print(f"\nparents 是一个长度为 {len(parents)} 的整数数组")
    print("每个元素 parents[i] 表示关节 i 的父节点索引")
    print("\n特殊值:")
    print("  - parents[0] = -1: 表示 pelvis 是根节点，没有父节点")
    print("  - parents[i] >= 0: 表示关节 i 的父节点是 parents[i]")
    
    print(f"\n前 10 个元素: {parents[:10]}")
    
    # 打印层级树
    print_hierarchy_tree(parents, joint_names, max_depth=2)
    
    # 解释用法
    explain_parents_usage()
    
    # 显示映射关系
    show_body_pose_mapping()
    
    # 演示正向运动学
    demonstrate_forward_kinematics()
    
    print("\n" + "="*70)
    print("总结")
    print("="*70)
    print("""
self.smplx_parents 的作用:
1. 定义关节层级关系（父子关系）
2. 在逆向 IK (robot→SMPL-X) 中：用于计算局部旋转
   - local_R = parent_R.inv() * joint_R
3. 在正向 FK (SMPL-X→robot) 中：用于重建世界旋转
   - joint_R = parent_R * local_R
4. 确保关节按照正确的层级顺序处理（从根节点到叶子节点）

在 reverse_motion_retarget.py 第 82 行:
   self.smplx_parents = self.smplx_model.parents.detach().cpu().numpy().astype(int)

这行代码把 PyTorch tensor 转换成 NumPy 数组，方便后续索引和计算。
    """)


if __name__ == "__main__":
    main()

