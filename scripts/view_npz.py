#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import os


# python scripts/view_npz.py  assets/body_models/smplx/SMPLX_MALE.npz
# python scripts/view_npz.py  ../server3_data/locomotion/reference/000005.npz
# python scripts/view_npz.py  ../server3_data/locomotion/reference/000135.npz
# python scripts/view_npz.py  ../server3_data/locomotion/reference/006119.npz
# python scripts/view_npz.py  ../server3_data/locomotion/reference/009177.npz
## 显示第一帧完整数据: python scripts/view_npz.py  ../server3_data/locomotion/reference/000005.npz --first-frame

# 文件包含的键: ['gender', 'betas', 'pose_body', 'pose_hand', 'smpl_trans', 'smpl_quat_xyzw', 'pelvis_trans', 'pelvis_quat_xyzw', 'joints_local', 'fps']
#gender, betas, pose_body, 

# raw DATASET npz file: ['gender', 'surface_model_type', 'mocap_frame_rate', 'mocap_time_length', 'markers_latent', 'latent_labels', 'markers_latent_vids', 'trans', 'poses', 'betas', 'num_betas', 'root_orient', 'pose_body', 'pose_hand', 'pose_jaw', 'pose_eye', 'markers', 'labels', 'markers_obs', 'labels_obs', 'markers_sim', 'marker_meta', 'num_markers']

# python scripts/view_npz.py  ../server3_data/locomotion/human/ik_based/npz/000005.npz

def view_npz_data(npz_path, show_preview=True, save_csv=False, show_first_frame=False):
    """查看NPZ文件内容"""
    print(f"🔍 查看NPZ文件: {npz_path}")
    print("="*60)
    
    # 加载NPZ文件
    data = np.load(npz_path, allow_pickle=True)
    
    print(f"📁 文件包含的键: {list(data.keys())}")
    print()
    
    for key in data.keys():
        value = data[key]
        print(f"🔑 {key}:")
        
        if isinstance(value, np.ndarray):
            print(f"   形状: {value.shape}")
            print(f"   数据类型: {value.dtype}")

            if key == 'gender':
                if value.size == 1:
                    print(f"   值: {value.item()}")
                else:
                    print(f"   值列表: {value.tolist()}")
                print()
                continue
            
            # 处理对象类型（如字典、列表等）
            if value.dtype == object:
                print(f"   类型: 对象 (object)")
                if value.size == 1:
                    obj = value.item()
                    if isinstance(obj, dict):
                        print(f"   字典内容: {obj}")
                    else:
                        print(f"   值: {obj}")
                elif value.size > 0 and show_preview:
                    print(f"   前几个值: {value.flatten()[:5]}")
            else:
                # 数值类型才计算范围
                try:
                    print(f"   数值范围: [{np.min(value):.6f}, {np.max(value):.6f}]")
                except Exception:
                    print(f"   无法计算数值范围")

                if key == 'betas':
                    if value.size > 0:
                        pass
                        # print(f"   全部数值: {np.array2string(value, precision=6, separator=', ')}")
                if show_preview and value.size > 0:
                    if value.ndim == 1:
                        print(f"   全部数值: {value[:value.size]}")
                
                # 显示第一帧的完整数据
                if show_first_frame and value.size > 0:
                    if value.ndim >= 1:
                        print(f"   ━━━ 第一帧完整数据 ━━━")
                        if value.ndim == 1:
                            # 一维数组（如 betas）
                            print(f"   完整值: {value}")
                        elif value.ndim == 2:
                            # 二维数组（如 pose_body, pose_hand）
                            first_frame = value[0]
                            print(f"   第0帧数据 (长度={len(first_frame)}):")
                            # 每行打印10个值
                            for i in range(0, len(first_frame), 10):
                                chunk = first_frame[i:i+10]
                                indices = ", ".join([f"[{j:2d}]" for j in range(i, min(i+10, len(first_frame)))])
                                values = ", ".join([f"{v:8.4f}" for v in chunk])
                                print(f"     索引 {indices}")
                                print(f"     数值 {values}")
                        else:
                            print(f"   第0帧形状: {value[0].shape}")
                            print(f"   第0帧内容: {value[0]}")
    
            
            # 如果是关节名称
            if key == 'joint_names':
                print(f"   关节名称: {list(value)}")

            if key == 'pose_body':
                # print(f"    pose_body: {value}")
                np.set_printoptions(threshold=np.inf, linewidth=200, suppress=True)
                print("pose_body (first 5 frames):")
                print(value[:5])
            if key == "pelvis_trans":
                # print(f"    pelvis_trans: {value}")
                np.set_printoptions(threshold=np.inf, linewidth=200, suppress=True)
                # print("pelvis_trans (first 5 frames):")
                # print(value[:5])
                print("pelvis_trans (last 7 frames):")
                print(value[-7:])
            if key == "smpl_trans":
                # print(f"    smpl_trans: {value}")
                np.set_printoptions(threshold=np.inf, linewidth=200, suppress=True)
                # print("smpl_trans (first 5 frames):")
                # print(value[:5])
                print("smpl_trans (last 7 frames):")
                print(value[-7:])
            if key == "pelvis_quat_xyzw":
                # print(f"    pelvis_quat_xyzw: {value}")
                np.set_printoptions(threshold=np.inf, linewidth=200, suppress=True)
                # print("pelvis_quat_xyzw (first 5 frames):")
                # print(value[:5])
                print("pelvis_quat_xyzw (last 7 frames):")
                print(value[-7:])
            if key == "smpl_quat_xyzw":
                # print(f"    smpl_quat_xyzw: {value}")
                np.set_printoptions(threshold=np.inf, linewidth=200, suppress=True)
                # print("smpl_quat_xyzw (first 5 frames):")
                # print(value[:5])
                print("smpl_quat_xyzw (last 7 frames):")
                print(value[-7:])
            
        else:
            print(f"   值: {value}")
        print()
    
    # 保存为CSV（如果请求）
    if save_csv and 'full_data' in data:
        csv_path = npz_path.replace('.npz', '_extracted.csv')
        np.savetxt(csv_path, data['full_data'], fmt='%.6f', delimiter=',')
        print(f"💾 已保存为CSV: {csv_path}")
    
    data.close()

def main():
    ap = argparse.ArgumentParser("查看NPZ文件内容")
    ap.add_argument("npz_path", help="NPZ文件路径")
    ap.add_argument("--no-preview", action="store_true", help="不显示数据预览")
    ap.add_argument("--save-csv", action="store_true", help="保存为CSV文件")
    ap.add_argument("--first-frame", action="store_true", help="显示第一帧的完整数据")
    args = ap.parse_args()
    
    if not os.path.exists(args.npz_path):
        print(f"❌ 文件不存在: {args.npz_path}")
        return
    
    view_npz_data(args.npz_path, 
                  show_preview=not args.no_preview, 
                  save_csv=args.save_csv,
                  show_first_frame=args.first_frame)
    
    
    from smplx.joint_names import JOINT_NAMES
    # print("SMPL-X Body Joints (pose_body):")
    # for i in range(1, 22):  # 跳过索引 0 的 pelvis
    #     print(f"  pose_body[{(i-1)*3}:{(i-1)*3+3}] → {JOINT_NAMES[i]}")

    # print("\nSMPL-X Hand Joints (pose_hand):")
    # # 左手
    # for i in range(15):
    #     joint_idx = 22 + i  # 手指从索引 22 开始
    #     print(f"  pose_hand[{i*3}:{i*3+3}] → {JOINT_NAMES[joint_idx]} (左手)")
    # # 右手
    # for i in range(15):
    #     joint_idx = 37 + i
    #     print(f"  pose_hand[{45+i*3}:{45+i*3+3}] → {JOINT_NAMES[joint_idx]} (右手)")

if __name__ == "__main__":
    main()
