#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import matplotlib.pyplot as plt
import os

def load_csv_data(csv_path):
    """加载CSV数据"""
    data = np.loadtxt(csv_path, delimiter=',')
    print(f"[CSV] 形状: {data.shape}, 数据类型: {data.dtype}")
    return data

def load_npz_data(npz_path):
    """加载NPZ数据"""
    data = np.load(npz_path)
    print(f"[NPZ] 包含的键: {list(data.keys())}")
    
    # 获取主要数据
    if 'full_data' in data:
        full_data = data['full_data']
        print(f"[NPZ] full_data 形状: {full_data.shape}, 数据类型: {full_data.dtype}")
        return full_data, data
    else:
        raise ValueError("NPZ文件中没有找到 'full_data' 键")

def compare_data(csv_data, npz_data, tolerance=1e-6, start_row=None, end_row=None):
    """对比CSV和NPZ数据"""
    print("\n" + "="*60)
    print("数据对比分析")
    print("="*60)
    
    # 形状对比
    print(f"CSV形状: {csv_data.shape}")
    print(f"NPZ形状: {npz_data.shape}")
    
    if csv_data.shape != npz_data.shape:
        print("❌ 形状不匹配！")
        return False
    
    # 数值对比
    diff = np.abs(csv_data - npz_data)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    print(f"\n数值差异统计:")
    print(f"  最大差异: {max_diff:.2e}")
    print(f"  平均差异: {mean_diff:.2e}")
    print(f"  容差阈值: {tolerance:.2e}")
    
    if max_diff < tolerance:
        print("✅ 数据完全匹配！")
        return True
    else:
        print("❌ 数据存在差异！")
        return False

def visualize_comparison(csv_data, npz_data, output_dir="comparison_plots"):
    """可视化对比数据"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 选择几个关键列进行可视化
    num_cols_to_plot = min(6, csv_data.shape[1])
    cols_to_plot = np.linspace(0, csv_data.shape[1]-1, num_cols_to_plot, dtype=int)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, col in enumerate(cols_to_plot):
        if i >= len(axes):
            break
            
        ax = axes[i]
        ax.plot(csv_data[:, col], 'b-', label='CSV', alpha=0.7, linewidth=1)
        ax.plot(npz_data[:, col], 'r--', label='NPZ', alpha=0.7, linewidth=1)
        ax.set_title(f'列 {col}')
        ax.set_xlabel('帧数')
        ax.set_ylabel('数值')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 隐藏多余的子图
    for i in range(len(cols_to_plot), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'data_comparison.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 对比图保存至: {plot_path}")
    plt.close()

def print_statistics(data, name):
    """打印数据统计信息"""
    print(f"\n{name} 统计信息:")
    print(f"  形状: {data.shape}")
    print(f"  数据类型: {data.dtype}")
    print(f"  最小值: {np.min(data):.6f}")
    print(f"  最大值: {np.max(data):.6f}")
    print(f"  平均值: {np.mean(data):.6f}")
    print(f"  标准差: {np.std(data):.6f}")

def analyze_npz_structure(npz_data_dict):
    """分析NPZ文件结构"""
    print("\n" + "="*60)
    print("NPZ文件结构分析")
    print("="*60)
    
    for key, value in npz_data_dict.items():
        if isinstance(value, np.ndarray):
            print(f"{key}: 形状={value.shape}, 类型={value.dtype}")
            if key == 'joint_names':
                print(f"  关节名称: {value[:5]}...")  # 显示前5个关节名
        else:
            print(f"{key}: {value}")

def main():
    ap = argparse.ArgumentParser("验证NPZ和CSV文件的数据一致性")
    ap.add_argument("--csv", required=True, help="CSV文件路径")
    ap.add_argument("--npz", required=True, help="NPZ文件路径")
    ap.add_argument("--tolerance", type=float, default=1e-6, help="数值比较容差")
    ap.add_argument("--visualize", action="store_true", help="生成可视化对比图")
    ap.add_argument("--output-dir", default="comparison_plots", help="可视化输出目录")
    ap.add_argument("--start-row", type=int, default=3, help="局部对比起始行 (默认: 3)")
    ap.add_argument("--end-row", type=int, default=33, help="局部对比结束行 (默认: 33)")
    ap.add_argument("--local-only", action="store_true", help="只进行局部对比，不进行全局对比")
    args = ap.parse_args()
    
    print("🔍 开始验证NPZ和CSV文件数据一致性...")
    
    # 检查文件存在
    if not os.path.exists(args.csv):
        print(f"❌ CSV文件不存在: {args.csv}")
        return
    if not os.path.exists(args.npz):
        print(f"❌ NPZ文件不存在: {args.npz}")
        return
    
    try:
        # 加载数据
        print("\n📂 加载数据...")
        csv_data = load_csv_data(args.csv)
        npz_data, npz_dict = load_npz_data(args.npz)
        
        # 打印统计信息
        print_statistics(csv_data, "CSV")
        print_statistics(npz_data, "NPZ")
        
        # 分析NPZ结构
        analyze_npz_structure(npz_dict)
        
        # 对比数据
        if args.local_only:
            # 只进行局部对比
            is_match = compare_data(csv_data, npz_data, args.tolerance, args.start_row, args.end_row)
        else:
            # 进行完整对比（局部+全局）
            is_match = compare_data(csv_data, npz_data, args.tolerance, args.start_row, args.end_row)
        
        # 可视化（如果请求）
        if args.visualize:
            print("\n📊 生成可视化对比图...")
            visualize_comparison(csv_data, npz_data, args.output_dir)
        
        # 最终结果
        print("\n" + "="*60)
        if is_match:
            print("🎉 验证通过！NPZ和CSV数据完全一致！")
        else:
            print("⚠️  验证失败！数据存在差异，请检查转换过程！")
        print("="*60)
        
    except Exception as e:
        print(f"❌ 验证过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
