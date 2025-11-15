#TODO: 该文件可以删除！！！

"""
测试脚本：展示配置文件中的名称如何被解析为 MJCF 中的实际 body 名称
以及最终传入 mink.FrameTask 的 frame_name 是什么

Usage: python scripts/debug_frame_name_resolution.py
"""
import mujoco as mj
from pathlib import Path
import sys
import json

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from general_motion_retargeting.params import SMPLX_HUMANOID_XML, REVERSE_IK_CONFIG_DICT

# 加载 SMPL-X MJCF 模型
smplx_xml_path = Path(SMPLX_HUMANOID_XML)
model = mj.MjModel.from_xml_path(str(smplx_xml_path))

# 构建 body_name_to_id 和 body_alias_map（复制第 100-111 行的逻辑）
body_name_to_id = {}
body_alias_map = {}

print("=" * 80)
print("步骤 1: 构建 MJCF body 名称和别名映射")
print("=" * 80)

for body_id in range(model.nbody):
    name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, body_id)
    if name:
        body_name_to_id[name] = body_id
        lower = name.lower()
        body_alias_map[lower] = name
        
        # 创建 left_/right_ 别名
        if lower.startswith("l_"):
            alias = "left_" + lower[2:]
            body_alias_map[alias] = name
            print(f"  别名映射: '{alias}' → '{name}'")
        if lower.startswith("r_"):
            alias = "right_" + lower[2:]
            body_alias_map[alias] = name
            print(f"  别名映射: '{alias}' → '{name}'")

print(f"\n总计 {len(body_alias_map)} 个别名映射\n")

# 加载 g1_to_smplx 配置文件
ik_config_path = REVERSE_IK_CONFIG_DICT.get("unitree_g1")
with open(ik_config_path, 'r') as f:
    ik_config = json.load(f)

# resolve_smplx_body_name 函数（复制第 138-153 行的逻辑）
def resolve_smplx_body_name(candidate):
    custom_aliases = {
        "spine3": "chest",
        "left_foot": "l_toe",
        "right_foot": "r_toe",
    }
    
    key = candidate.lower()
    alias_key = custom_aliases.get(key, key)
    if alias_key in body_alias_map:
        return body_alias_map[alias_key]
    if candidate in body_name_to_id:
        return candidate
    raise KeyError(f"SMPL-X body '{candidate}' not found")

# 测试 ik_match_table1 中的前 15 个条目
print("=" * 80)
print("步骤 2: 解析配置文件中的名称 → frame_name (传入 mink.FrameTask)")
print("=" * 80)
config_header = "配置文件中的键 (smplx_body)"
frame_header = "frame_name (传入 mink)"
print(f"{config_header:<35} → {frame_header:<25}")
print("=" * 80)

count = 0
table_count = 0
should_break = False

for table_name in ("ik_match_table1", "ik_match_table2"):
    if should_break:
        break
        
    table = ik_config.get(table_name, {})
    
    if table_count > 0:
        print("-" * 80)
        print(f"切换到 {table_name}")
        print("-" * 80)
    table_count += 1
    
    for smplx_body, entry in table.items():
        if count >= 20:  # 只显示前 20 个
            remaining = sum(len(ik_config.get(t, {})) for t in ("ik_match_table1", "ik_match_table2"))
            print(f"\n... (省略剩余条目，两个表总计约 {remaining} 个条目)")
            should_break = True
            break
            
        if not entry:
            continue
        
        try:
            frame_name = resolve_smplx_body_name(smplx_body)
            robot_body, pos_weight, rot_weight, _, _ = entry
            
            # 只显示有效的任务（权重不为 0）
            if pos_weight != 0 or rot_weight != 0:
                print(f"{smplx_body:<35} → {frame_name:<25} (pos={pos_weight}, rot={rot_weight})")
                count += 1
        except KeyError as e:
            print(f"{smplx_body:<35} → ❌ 解析失败: {e}")
            count += 1

print("=" * 80)
print("\n✅ 结论:")
print("   1. 配置文件中使用小写名称 (如 'pelvis', 'left_hip')")
print("   2. resolve_smplx_body_name() 通过别名映射转换为 MJCF 实际名称")
print("   3. 传入 mink.FrameTask 的 frame_name 已经是正确的大写形式 (如 'Pelvis', 'L_Hip')")
print("   4. mink 不需要自己处理大小写问题，因为 frame_name 已经是正确的")
print("=" * 80)

