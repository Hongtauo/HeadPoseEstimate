# 使用utf-8编码
import numpy as np
from matplotlib import pyplot as plt

# 定义一个计算类
class HeadPoseEstimate():
    def __init__(self):
        """
        参数说明：
        nose_pos: 鼻子关键点坐标 (tuple 或 list, 如 [x, y])
        left_eye_pos: 左眼关键点坐标 (tuple 或 list, 如 [x, y])
        right_eye_pos: 右眼关键点坐标 (tuple 或 list, 如 [x, y])
        left_ear_pos: 左耳关键点坐标 (tuple 或 list, 如 [x, y])
        right_ear_pos: 右耳关键点坐标 (tuple 或 list, 如 [x, y])
        left_shoulder_pos: 左肩关键点坐标 (tuple 或 list, 如 [x, y])
        right_shoulder_pos: 右肩关键点坐标 (tuple 或 list, 如 [x, y])
        shoulder_center: 左右肩中点坐标 (tuple 或 list, 如 [x, y])
        head_range: 头部姿态角区间列表，用于区分象限
        fixed_values: 头部姿态角的固定值列表
        L: 视线估计长度参数，需要人为设定
        F_pos: 视线估计落点坐标 (tuple 或 list, 如 [x, y])
        skeleton：coco格式的骨架
        """
        self.keypoints = None

        self.nose_pos = None

        self.left_eye_pos = None
        self.right_eye_pos = None

        self.left_ear_pos = None
        self.right_ear_pos = None

        self.left_shoulder_pos = None
        self.right_shoulder_pos = None

        self.shoulder_center = None

        self.head_range = [[-180, -90], [-90, 0], [0, 90], [90, 180]]
        self.fixed_values = [0, 180]

        self.L = 1

        # 视线估计落点
        self.F_pos = None

        self.skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13],  # 腿部连接
                        [6, 12], [7, 13],  # 躯干到臀部
                        [6, 7],   # 肩膀连接
                        [6, 8], [7, 9], [8, 10], [9, 11],  # 手臂连接
                        [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7] 
                          # 头部和面部连接
                        ]

    # 计算肩部中点
    def _cal_shoulder_center(self):
        if self.left_shoulder_pos is not None and self.right_shoulder_pos is not None:
            self.shoulder_center = (
                (self.left_shoulder_pos[0] + self.right_shoulder_pos[0]) / 2,
                (self.left_shoulder_pos[1] + self.right_shoulder_pos[1]) / 2
            )
        else:
            self.shoulder_center = None

    def _cal_l(self):
        if self.left_ear_pos is not None and self.right_ear_pos is not None and self.nose_pos is not None:
            l1 = self.nose_pos[0] - self.left_ear_pos[0]
            l2 = self.right_ear_pos[0] - self.nose_pos[0]
            return l1, l2
        return None, None
    
    def _get_head_pose_angle(self,theta1):
        """
        判断θ1属于哪个区间或是固定值，并返回对应的θ2
        """
        if theta1 == self.fixed_values[0]:
            return self.fixed_values[0]
        elif theta1 == self.fixed_values[1] or theta1 == -180:
            return self.fixed_values[1]
        else:
            for r in self.head_range:
                if r[0] < theta1 <= r[1]:
                    return (r[0] + r[1]) / 2
        return None

    def _cal_estimate_head_pose_direction_angle(self,theta2):
        """
        计算头部姿态方向角
        获得视线估计落点以鼻部为原点的相对坐标F′(x ,y)
        """
        x = self.L * np.sin(np.radians(theta2)) + self.nose_pos[0]
        y = self.L * np.cos(np.radians(theta2)) + self.nose_pos[1]
        return np.array([x, y])

    def estimate_head_pose(self):
        """
        1. 计算左耳到鼻子的距离 l1，右耳到鼻子的距离 l2：
            l1 = 鼻子的x坐标 - 左耳的x坐标
            l2 = 右耳的x坐标 - 鼻子的x坐标

        2. 判断 l1 和 l2 的大小，选择坐标原点 (x0, y0)：
            - x0 取左右肩部中点的 x 坐标
            - y0 根据耳部与眼部的 y 坐标选择
            - 若 l1 > l2，则 x0 = 肩部中点的x坐标, y0 = 左耳的y坐标
            - 若 l1 < l2，则 x0 = 肩部中点的x坐标, y0 = 右耳的y坐标

        3. 得到头部局部坐标原点 O(x0, y0)，最终计算头部姿态的象限角 θ
        """
        # 首先调用函数计算肩部中点
        self._cal_shoulder_center()
        # 计算l1，l2
        l1, l2 = self._cal_l()
        # 判断l1和l2的大小，选取局部坐标原点
        if l1 is not None and l2 is not None:
            x0 = self.shoulder_center[0] if self.shoulder_center is not None else 0
            y0 = self.left_ear_pos[1] if l1 > l2 else self.right_ear_pos[1]
            # 计算头部姿态的象限角 θ

        # theta1 = np.arctan(y0/(x0*((self.nose_pos[1] - y0) / (self.nose_pos[0]-x0))))
        # 进行平滑，避免除零错误
        denominator = x0 * ((self.nose_pos[1] - y0) / (self.nose_pos[0] - x0))
        denominator = denominator if abs(denominator) > 1e-8 else 1e-8  # 防止分母为零
        theta1 = np.arctan(y0 / denominator)
        
        # 转为角度
        theta1 = np.degrees(theta1)

        # 获取头部姿态的象限角
        theta2 = self._get_head_pose_angle(theta1)

        # 计算头部姿态方向角
        self.F_pos = self._cal_estimate_head_pose_direction_angle(theta2)

    def set_args(self,keypoints,L):
        # 传入一个coco格式的keypoint，要求索引从0开始
        self.keypoints = keypoints
        # 输入视线长度L
        self.L = L
        
        # 设置所需参数
        self.nose_pos = keypoints[0]
        self.left_eye_pos = keypoints[1]
        self.right_eye_pos = keypoints[2]
        self.left_ear_pos = keypoints[3]
        self.right_ear_pos = keypoints[4]
        self.left_shoulder_pos = keypoints[5]
        self.right_shoulder_pos = keypoints[6]

    def plot(self):
        # 绘图
        plt.figure(figsize=(8, 8))
        xy = self.keypoints  # 不乘以 img_w, img_h，直接用归一化坐标
        F_pixel = self.F_pos     # F 也是归一化坐标
        nose_pixel = self.nose_pos

        skeleton_0based = [[joint[0]-1, joint[1]-1] for joint in self.skeleton]

        # 画骨架
        for connection in skeleton_0based:
            point1_idx, point2_idx = connection
            if point1_idx < len(self.keypoints) and point2_idx < len(self.keypoints):
                x1, y1 = self.keypoints[point1_idx]
                x2, y2 = self.keypoints[point2_idx]
                if x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0:
                    plt.plot([x1, x2], [y1, y2], 'b-', linewidth=2, alpha=0.7)

        # 画关键点
        for i, (x, y) in enumerate(xy):
            if x > 0 and y > 0:
                plt.scatter(x, y, c='red', s=80, zorder=5)
                plt.annotate(str(i), (x, y), xytext=(5, 5), textcoords='offset points',
                            fontsize=10, color='black', weight='bold')

        # 画鼻子到F的连线
        plt.plot([nose_pixel[0], F_pixel[0]], [nose_pixel[1], F_pixel[1]], 'g-', linewidth=2, alpha=0.9, label='Nose to F')
        plt.scatter(F_pixel[0], F_pixel[1], c='green', s=100, marker='*', label='F point', zorder=10)
        plt.legend()

        # 自适应坐标轴
        valid = (xy[:, 0] > 0) & (xy[:, 1] > 0)
        xy_valid = xy[valid]
        if len(xy_valid) > 0:
            margin = 0.05  # 归一化坐标，边距可设为0.05
            min_x, max_x = xy_valid[:, 0].min() - margin, xy_valid[:, 0].max() + margin
            min_y, max_y = xy_valid[:, 1].min() - margin, xy_valid[:, 1].max() + margin
            plt.xlim(min_x, max_x)
            plt.ylim(max_y, min_y)
            
            plt.axis('off')
            plt.title('Pose Skeleton and Nose-F Point')
        else:
            plt.xlim(0, 1)
            plt.ylim(1, 0)

            plt.axis('off')
            plt.title('Pose Skeleton and Nose-F Point')

# 主函数，测试是否能够运行成功
if __name__ == "__main__":
    head_pose_estimator = HeadPoseEstimate()
    keypoints = [
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
    [0.079228, 0.24624]]
    keypoints = np.array(keypoints)

    head_pose_estimator.set_args(keypoints)


    # 估计头部姿态
    head_pose_estimator.estimate_head_pose()
    # 输出结果
    print("测试：头部姿态估计落点", head_pose_estimator.F_pos)
    # 绘图
    head_pose_estimator.plot()


