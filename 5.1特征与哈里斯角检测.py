import cv2 as cv
import numpy as np
import math
from matplotlib import pyplot as plt

## 有如下功能：
 # 【特征】：*如果我们对这种特征进行定义，可能会发现很难用语言来表达它，但是我们知道它们是什么*
 # ---*这些特征是什么？（答案对于计算机也应该是可以理解的。）*
 # ---*基本上，拐点被认为是图像中的良好特征。*
 # ---*** 我们以一种直观的方式回答了这一问题，即寻找图像中在其周围所有区域中移动（少量）变化最大的区域。***
 # - 特征值与特征向量 
 # - 我们将了解”Harris Corner Detection”背后的概念。 
 # - 我们将看到以下函数：cv.cornerHarris(),cv.cornerSubPix()

# 【特征值与特征向量】参考学习 https://www.matongxue.com/madocs/228
# 定义: Ax = cx, A为方阵,c为特征值，x为特征向量（）。

# 【哈里斯角检测】cv.cornerHarris(img,blockSize,ksize,k) 原理参考 https://blog.csdn.net/lwzkiller/article/details/54633670
#  - img - 输入图像，应为灰度和float32类型。 
#  - blockSize - 是拐角检测考虑的邻域大小 
#  - ksize - Sobel求导中使用的窗口大小。 
#  - k - 等式中的哈里斯检测器自由参数，取值参数为 [0,04,0.06]
#  - 返回值，存储R = λ1*λ2 -k(λ1+λ2)^2数值，和原始图像一样大
# 【原理】如文章所述，用数学公式去刻画的角点特征：
# E(u,v) = ∑w(x,y)[I(x+u, y+v)-I(x,y)]^2
# 其中(u,v)是窗口的偏移量，(x,y)是窗口内像素坐标，w(x,y)是窗口函数，已窗口原点为中心的高斯核，I为求像素数值
# 上述式子通过泰勒展开(只保留一阶导数)后，变形为[u,v]*M*[u,v]T
# 矩阵M为二维方阵，是Ix和Iy导数构成的（具体参考文章）
# 对M可以求出两个特征值λ1和λ2
# 最后利用公式 R = λ1*λ2 -k(λ1+λ2)^2来计算返回值，分三种情况：
#1、特征值都比较大时，即窗口中含有角点，R>0
#2、特征值一个较大，一个较小，窗口中含有边缘，R<0
#3、特征值都比较小，窗口处在平坦区域,此时R绝对值很小
# 最后可以画出文章中的λ1和λ2坐标图
img_gray = cv.imread('./pic/1.jpg', 0)
dist = cv.cornerHarris(img_gray, 2, 3, 0.04)
dist = cv.dilate(dist, None) #对数据膨胀一下，方便标记的清楚
img_color = cv.imread('./pic/1.jpg')
img_color[dist>0.1*dist.max()] = [255, 0, 0] #最佳阈值取出
cv.imshow('img_color_0.1', img_color)
img_color = cv.imread('./pic/1.jpg')
img_color[dist>0.01*dist.max()] = [255, 0, 0] #最佳阈值取出
cv.imshow('img_color_0.01', img_color)
img_color = cv.imread('./pic/1.jpg')
img_color[dist>0.05*dist.max()] = [255, 0, 0] #最佳阈值取出
cv.imshow('img_color_0.05', img_color)

# 【亚像素SubPixel角点检测】cv.cornerSubPix，通过对哈里斯检测到的一个角上有一堆像素，我们取它们的质心来细化它们


cv.waitKey(0)
cv.destroyAllWindows()