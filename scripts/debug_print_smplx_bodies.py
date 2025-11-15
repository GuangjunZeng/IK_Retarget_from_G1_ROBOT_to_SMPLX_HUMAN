#test file: print all body names in the SMPL-X MJCF model from "smplx_humanoid.xml" file

#usage： python scripts/debug_print_smplx_bodies.py

#totol: 52 bodies
# Body ID    | Body Name
# ============================================================
# 0          | world
# 1          | Pelvis  ✅
# 2          | L_Hip   ✅
# 3          | L_Knee  ✅
# 4          | L_Ankle ✅
# 5          | L_Toe   ✅
# 6          | R_Hip   ✅
# 7          | R_Knee  ✅
# 8          | R_Ankle ✅
# 9          | R_Toe   ✅
# 10         | Torso   ✅
# 11         | Spine   ✅
# 12         | Chest   ✅
# 13         | Neck    ✅
# 14         | Head    ✅
# 15         | L_Thorax✅
# 16         | L_Shoulder✅
# 17         | L_Elbow ✅
# 18         | L_Wrist ✅
# 19         | L_Index1✅
# 20         | L_Index2✅
# 21         | L_Index3✅
# 22         | L_Middle1✅
# 23         | L_Middle2✅
# 24         | L_Middle3✅
# 25         | L_Pinky1✅
# 26         | L_Pinky2✅
# 27         | L_Pinky3✅
# 28         | L_Ring1✅
# 29         | L_Ring2✅
# 30         | L_Ring3✅
# 31         | L_Thumb1✅
# 32         | L_Thumb2✅
# 33         | L_Thumb3✅
# 34         | R_Thorax✅
# 35         | R_Shoulder✅
# 36         | R_Elbow✅
# 37         | R_Wrist✅
# 38         | R_Index1✅
# 39         | R_Index2✅
# 40         | R_Index3✅
# 41         | R_Middle1✅
# 42         | R_Middle2✅
# 43         | R_Middle3✅
# 44         | R_Pinky1✅
# 45         | R_Pinky2✅
# 46         | R_Pinky3✅
# 47         | R_Ring1✅
# 48         | R_Ring2✅
# 49         | R_Ring3✅
# 50         | R_Thumb1✅
# 51         | R_Thumb2✅
# 52         | R_Thumb3✅
# ============================================================


import mujoco as mj
from pathlib import Path
import sys

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from general_motion_retargeting.params import SMPLX_HUMANOID_XML

# 加载 SMPL-X MJCF 模型 from xml file
smplx_xml_path = Path(SMPLX_HUMANOID_XML)
if not smplx_xml_path.exists():
    print(f"❌ SMPL-X MJCF not found: {smplx_xml_path}")
    sys.exit(1)

print(f"✅ Loading SMPL-X MJCF from: {smplx_xml_path}\n")

model = mj.MjModel.from_xml_path(str(smplx_xml_path))

print(f"Total number of bodies (model.nbody): {model.nbody}\n")
print("=" * 60)
print(f"{'Body ID':<10} | {'Body Name'}")
print("=" * 60)

# 遍历所有 body 并打印名称
for body_id in range(model.nbody):
    name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, body_id)
    if name:
        print(f"{body_id:<10} | {name}")   #<10 - 左对齐，宽度为 10
    else:
        print(f"{body_id:<10} | (unnamed/world)")

print("=" * 60)
print(f"\nTotal named bodies: {sum(1 for i in range(model.nbody) if mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, i))}")

