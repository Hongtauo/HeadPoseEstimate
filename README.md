# HeadPoseEstimate 使用说明

## 1. 简介

`HeadPoseEstimate` 是一个基于 COCO 关键点格式的头部姿态估计类，支持头部朝向角度计算和示意图绘制。适用于人体姿态估计任务，关键点输入需为归一化坐标。

## 2. 依赖

- numpy
- matplotlib

## 3. 典型用法

```python
from HeadPoseEstimate.HeadPoseEstimate import HeadPoseEstimate
import numpy as np

# 初始化
head_pose_estimator = HeadPoseEstimate()

# 输入COCO格式关键点（归一化坐标，17个点）
keypoints = np.array([
    [0.058675, 0.11372],
    [0.057391, 0.10792],
    [0.055773, 0.11197],
    [0.052217, 0.10668],
    [0.048211, 0.1169],
    [0.055332, 0.1206],
    [0.044132, 0.14464],
    [0.066078, 0.13617],
    [0.052265, 0.17971],
    [0.068525, 0.13798],
    [0.065174, 0.15665],
    [0.065679, 0.17183],
    [0.057007, 0.18642],
    [0.081184, 0.19503],
    [0.066703, 0.21454],
    [0.091306, 0.22964],
    [0.079228, 0.24624]
])
head_pose_estimator.set_args(keypoints)

# 姿态估计
head_pose_estimator.estimate_head_pose()
print("头部姿态估计落点:", head_pose_estimator.F_pos)

# 绘制示意图
head_pose_estimator.plot()
```
## 4. 方法说明
set_args(keypoints)：设置输入关键点（COCO格式，归一化）。
estimate_head_pose()：计算头部姿态角和落点。
plot()：绘制骨架和头部朝向落点示意图。
F_pos：头部朝向落点坐标（归一化）。

## 5. 注意事项
输入关键点必须为 numpy 数组，且为 17 个点。
坐标需归一化（0~1），否则绘图和计算结果不准确。
支持直接在 Jupyter Notebook 或 Python 脚本中调用。
## 6. 参考文献
感谢论文提供的解决思路：

[1] 赵思源, 彭春蕾, 张云, 等. 基于AlphaPose模型的远距离行人头部姿态估计算法[J]. 陕西科技大学学报, 2023, 41(2): 191-198