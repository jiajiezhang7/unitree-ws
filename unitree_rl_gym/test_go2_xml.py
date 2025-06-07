#!/usr/bin/env python3
"""
测试 GO2 MuJoCo XML 文件是否能正确加载
"""

import mujoco
import numpy as np
from legged_gym import LEGGED_GYM_ROOT_DIR
import os

def test_xml_loading():
    """测试 XML 文件加载"""
    xml_path = f"{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/scene.xml"
    
    print(f"测试 XML 文件: {xml_path}")
    print(f"文件是否存在: {os.path.exists(xml_path)}")
    
    try:
        # 尝试加载模型
        print("正在加载 MuJoCo 模型...")
        m = mujoco.MjModel.from_xml_path(xml_path)
        print("✓ XML 文件加载成功!")
        
        # 创建数据对象
        d = mujoco.MjData(m)
        print("✓ MuJoCo 数据对象创建成功!")
        
        # 打印一些基本信息
        print(f"\n模型信息:")
        print(f"  关节数量: {m.nq}")
        print(f"  自由度数量: {m.nv}")
        print(f"  执行器数量: {m.nu}")
        print(f"  传感器数量: {m.nsensor}")
        print(f"  几何体数量: {m.ngeom}")
        print(f"  刚体数量: {m.nbody}")
        
        # 打印关节名称
        print(f"\n关节名称:")
        for i in range(m.njnt):
            joint_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i)
            if joint_name:
                print(f"  {i}: {joint_name}")
        
        # 打印传感器名称
        print(f"\n传感器名称:")
        for i in range(m.nsensor):
            sensor_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SENSOR, i)
            if sensor_name:
                print(f"  {i}: {sensor_name}")
        
        # 测试一步仿真
        print(f"\n测试仿真步骤...")
        mujoco.mj_step(m, d)
        print("✓ 仿真步骤执行成功!")
        
        return True
        
    except Exception as e:
        print(f"✗ 错误: {e}")
        return False

if __name__ == "__main__":
    success = test_xml_loading()
    if success:
        print("\n🎉 GO2 XML 文件测试通过!")
    else:
        print("\n❌ GO2 XML 文件测试失败!")
