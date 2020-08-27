import cv2 as cv
import numpy as np
import math
from matplotlib import pyplot as plt

## 有如下功能：
# - 我们将学习另一个拐角检测器：Shi-Tomasi拐角检测器 
# - 我们将看到以下函数：cv.goodFeaturesToTrack()

#【原理】与Harris Harris Detector相比，显示了更好的结果。
# 哈里斯角落探测器的计分功能由下式给出：R=λ1λ2−k(λ1+λ2)^2
# 取而代之：R=min(λ1,λ2),从图中可以看到(图参考教程)，只有当λ1和λ2大于最小值λmin时，才将其视为拐角（绿色区域）
# void cv::goodFeaturesToTrack(
# 		cv::InputArray image, // 输入图像（CV_8UC1 CV_32FC1）
# 		cv::OutputArray corners, // 输出角点vector
# 		int maxCorners, // 最大角点数目
# 		double qualityLevel, // 质量水平系数（小于1.0的正数，一般在0.01-0.1之间）
# 		double minDistance, // 用于区分相邻两个角点的最小距离（小于这个距离得点将进行合并）
# 		cv::InputArray mask = noArray(), // 如果指定，它的维度必须和输入图像一致，且在mask值为0处不进行角点检测
# 		int blockSize = 3, // 使用的邻域数
# 		bool useHarrisDetector = false, // false ='Shi Tomasi metric' 默认不适用Harris检测
# 		double k = 0.04 // Harris角点检测时使用
# 	);
# 【测试结论】比harris角点检测效果好，在指定找很少个数的角点时，shi算法找的更精确
img_color = cv.imread('./pic/1.jpg')
img_gray = cv.imread('./pic/1.jpg', 0)
corners_shi = cv.goodFeaturesToTrack(img_gray, 4, 0.01, 10)
corners_harris = cv.goodFeaturesToTrack(img_gray, 4, 0.01, 10, useHarrisDetector=True)
for i in corners_shi:
    x,y = i.ravel()
    cv.circle(img_color,(x,y),3,255,-1)
cv.imshow('corners_shi', img_color)
img_color = cv.imread('./pic/1.jpg')
for i in corners_harris:
    x,y = i.ravel()
    cv.circle(img_color,(x,y),3,255,-1)
cv.imshow('corners_harris', img_color)

cv.waitKey(0)
cv.destroyAllWindows()