from scipy.spatial.transform import Rotation as R

#检查从 smplx_to_g1.json中reverse 计算得到的 g1_to_smplx_json的rot_offset是否正确

#python scripts/test_qua_compuation.py

#mark: smplx_to_g1.json中的quat是wxyz的顺序
 
# w, x, y, z = 0.4267755048530407,-0.5637931078484661,-0.5637931078484661,-0.4267755048530407
# w, x, y, z =  0.5, -0.5,-0.5, -0.5
# w, x, y, z =  -0.5, 0.5, 0.5, 0.5
# w, x, y, z =  0.70710678, 0.0, -0.70710678, 0.0
# w, x, y, z =  0, 0.70710678, 0, 0.70710678
# w, x, y, z =  1, 0, 0, 0
# w, x, y, z =  0, 0, 0, -1
# tx, ty, tz = 0.0, 0.0, 0.0
# tx, ty, tz = 0.0, 0.02, 0.0
# tx, ty, tz = 0.0, -0.02, 0.0

# rot_offset = R.from_quat([w, x, y, z], scalar_first=True)  # wxyz
# pos_offset = [tx, ty, tz]
# pos_offset_rev = - rot_offset.apply(pos_offset) #core calculation
# print(pos_offset_rev)
# print(-R_off.apply([tx, ty, tzx]))

#ik1 ✅
#[-0. -0. -0.]
#[-0. -0. -0.]
#[-0. -0. -0.]
#[-0.02 -0. -0.]
#[-0. -0. -0.]
#[-0. -0. -0.]
#[0.02 -0. -0.]
#[-0. -0. -0.]
#[-0. -0. -0.]
#[-0. -0. -0.]   left_elbow
#[-0. -0. -0.]
#[-0. -0. -0.]   right_shoulder_yaw_link
#[-0. -0. -0.]
#[-0. -0. -0.]


# ik2 (只有两个需要另外计算)
#left_toe_link  [-0. -0. -0.] ✅
#right_toe_link [-0. -0. -0.] ✅



# tx, ty, tz = 0.0, 0.0, 0.0
# w, x, y, z =  -0.5, 0.5, 0.5, 0.5
# rot_offset = R.from_quat([w, x, y, z], scalar_first=True)  # wxyz
# pos_offset = [tx, ty, tz]

# pos_smplx = [-0.3403,   0.2799,   0.9704]
# #! 注意归一化
# rot_smplx = [0.7281,  0.0446,  -0.0494,   0.6822]
# rot_offset = R.from_quat([w, x, y, z], scalar_first=True)
# rot_robot = R.from_quat(rot_smplx, scalar_first=True) * rot_offset  #[-0.70276547 -0.02405053  0.70756558  0.06995154]
# print("rot_robot: ", rot_robot.as_quat(scalar_first=True))
# pos_robot = pos_smplx + rot_robot.apply(pos_offset)
# print("pos_robot: ", pos_robot)  #[-0.3403  0.2799  0.9704]

# reverse_rot_offset = [-0.5, -0.5, -0.5, -0.5]
# reverse_rot_offset = R.from_quat(reverse_rot_offset, scalar_first=True)
# real_reverse_rot_offset = rot_offset.inv()  #[-0.5, -0.5, -0.5, -0.5]
# print("real_reverse_rot_offset: ", real_reverse_rot_offset.as_quat(scalar_first=True))
# rot_robot2 = [-0.70276547, -0.02405053,  0.70756558,  0.06995154]
# rot_smplx2 =  R.from_quat(rot_robot2, scalar_first=True) * real_reverse_rot_offset
# print("rot_smplx2: ", rot_smplx2.as_quat(scalar_first=True)) # [ 0.72811603  0.04460098 -0.04940109  0.68221502]
# pos_robot2 = [-0.3403,  0.2799,  0.9704]
# reverse_pos_offset = [-0.0, -0.0, -0.0]
# pos_smplx2 = pos_robot2 + rot_smplx2.apply(reverse_pos_offset) 
# print("pos_smplx2: ", pos_smplx2) #[-0.3403  0.2799  0.9704]




# test_rot_offset = (rot_robot.inv()) *  R.from_quat(rot_smplx, scalar_first=True) 
# print("test_rot_offset: ", test_rot_offset.as_quat(scalar_first=True))  #[-0.5 -0.5 -0.5 -0.5]




# tx, ty, tz = 0.0, -0.02, 0.0
# w, x, y, z =  0.5, -0.5, -0.5, -0.5

# w, x, y, z = 0.4267755048530407,-0.5637931078484661,-0.5637931078484661,-0.4267755048530407
# w, x, y, z =  0.5, -0.5,-0.5, -0.5
# w, x, y, z =  -0.5, 0.5, 0.5, 0.5
# w, x, y, z =  0.70710678, 0.0, -0.70710678, 0.0
# w, x, y, z =  0, 0.70710678, 0, 0.70710678
# w, x, y, z =  1, 0, 0, 0
w, x, y, z =  0, 0, 0, -1
tx, ty, tz = 0.0, 0.0, 0.0
# tx, ty, tz = 0.0, 0.02, 0.0
# tx, ty, tz = 0.0, 0.02, 0.0
# tx, ty, tz = 0.0, -0.02, 0.0
rot_offset = R.from_quat([w, x, y, z], scalar_first=True)  # wxyz
pos_offset = [tx, ty, tz]

pos_smplx = [-0.3403,   0.2799,   0.9704] 
rot_smplx = [0.7281,  0.0446,  -0.0494,   0.6822] #! 注意归一化
rot_offset = R.from_quat([w, x, y, z], scalar_first=True)
rot_robot = R.from_quat(rot_smplx, scalar_first=True) * rot_offset  # [ 0.70276547  0.02405053 -0.70756558 -0.06995154]
# print("rot_robot: ", rot_robot.as_quat(scalar_first=True))
pos_robot = pos_smplx + rot_robot.apply(pos_offset)
# print("pos_robot: ", pos_robot)  #[-0.34158569,  0.26011887,  0.96774411]

# reverse_rot_offset = [-0.5, -0.5, -0.5, -0.5]
# reverse_rot_offset = R.from_quat(reverse_rot_offset, scalar_first=True)
real_reverse_rot_offset = rot_offset.inv()  #[-0.5, -0.5, -0.5, -0.5]
print("real_reverse_rot_offset: ", real_reverse_rot_offset.as_quat(scalar_first=True))
rot_robot2 =  [ 0.70276547,  0.02405053, -0.70756558, -0.06995154]
rot_smplx2 =  R.from_quat(rot_robot2, scalar_first=True) * real_reverse_rot_offset
# print("rot_smplx2: ", rot_smplx2.as_quat(scalar_first=True)) #[ 0.72811603  0.04460098 -0.04940109  0.68221502]
pos_robot2 = [-0.34158569,  0.26011887,  0.96774411]
# reverse_pos_offset = [0.0, 0.0, 0.0]
reverse_pos_offset = -rot_offset.apply(pos_offset) #[ 0.02 -0.   -0.  ]
print("reverse_pos_offset: ", reverse_pos_offset)
pos_smplx2 = pos_robot2 + rot_smplx2.apply(reverse_pos_offset) 
# print("pos_smplx2: ", pos_smplx2) # [-0.34158569  0.26011887  0.96774411]
