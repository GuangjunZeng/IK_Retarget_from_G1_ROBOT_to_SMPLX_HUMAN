import numpy as np
import viser.transforms as tf

# 计算 R_adjust 旋转矩阵
R_adjust = tf.SO3.from_rpy_radians(0.0, -np.pi/2, 0.0) @ tf.SO3.from_rpy_radians(-np.pi/2, 0.0, 0.0)
R_adjust_mat = R_adjust.as_matrix()  # (3, 3)

print("R_adjust 旋转矩阵:")
print(R_adjust_mat)
# [[ 0.00000000e+00  1.00000000e+00 -1.11022302e-16]
#  [ 0.00000000e+00  0.00000000e+00  1.00000000e+00]
#  [ 1.00000000e+00 -1.11022302e-16 -2.22044605e-16]]
R_adjust_mat_inv = R_adjust_mat.T
print("R_adjust_mat_inv 旋转矩阵:")
print(R_adjust_mat_inv)
print()

# 测试旋转矩阵的性质
# print("旋转矩阵的性质测试:")
# print(f"行列式: {np.linalg.det(R_adjust_mat):.6f}")
# print(f"正交性检查 (R * R^T):")
# ortho_check = R_adjust_mat @ R_adjust_mat.T
# print(ortho_check)
# print(f"是否接近单位矩阵: {np.allclose(ortho_check, np.eye(3))}")
# print()

# # 测试旋转矩阵对向量的影响
# test_vector = np.array([1.0, 0.0, 0.0])
# rotated_vector = R_adjust_mat @ test_vector
# print(f"测试向量: {test_vector}")
# print(f"旋转后向量: {rotated_vector}")
# print()

# # 转换为四元数查看
# quat_xyzw = R_adjust.as_quaternion_xyzw()
# quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
# print(f"四元数 (xyzw): {quat_xyzw}")
# print(f"四元数 (wxyz): {quat_wxyz}")
# print(f"四元数模长: {np.linalg.norm(quat_xyzw):.6f}")

# python3 scripts/test-smpl-pelvis.py