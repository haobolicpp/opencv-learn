import cv2 as cv
import numpy as np
import math
from matplotlib import pyplot as plt

## 在本教程中，您将学习简单阈值，自适应阈值和Otsu阈值。
## 你将学习函数**cv.threshold**和**cv.adaptiveThreshold**
## inRange()函数可实现二值化功能（这点类似threshold()函数），关键的是可以同时针对多通道进行操作，使用起来非常方便！
## inRange(原图像，下界阈值，上界阈值)，作用是将两个阈值之间的设置为白色255，之外的设置为黑色0，参数和threshold不太一样 参考4_1

# 【简单阈值】——主要是threshold，连接https://blog.csdn.net/weixin_42296411/article/details/80901080
img = cv.imread('./pic/3.bmp', 0)
img_roi = img[1063:1127, 1639:1733]
# img_roi = cv.imread('./pic/3-sub.png', 0)
cv.imshow('img_roi', img_roi)
# 目标是把黑色的轮廓显示出来，弄成白的
#THRESH_BINARY:转换为二值图(0,255),200为要比较的阈值，当像素值>200,设定为maxval(255),否则设置为0，操作完后，opencv的图标变为黑色，背景变为白色
#THRESH_BINARY_INV是对THRESH_BINARY求反
def OnThreshold(x):
    ret, img_binary = cv.threshold(img_roi, x, 255, cv.THRESH_BINARY)
    ret, img_binary_inv = cv.threshold(img_roi, x, 255, cv.THRESH_BINARY_INV)
    cv.imshow('img_binary', img_binary)
    cv.imshow('img_binary_inv', img_binary_inv)
cv.namedWindow("Threshold",cv.WINDOW_NORMAL)
cv.createTrackbar("value", "Threshold", 0, 255, OnThreshold)
cv.imshow('Threshold', img)

# 【自适应阈值】 https://blog.csdn.net/fabulousli/article/details/51487833
# 简单阈值使用一个全局值作为阈值，但这可能并非在所有情况下都很好，例如，如果图像在不同区域具有不同的光照条件。在这种情况下，自适应阈值阈值化可以提供帮助
# 自适应阈值中，!!!!!亮度较高的图像区域的二值化阈值通常会较高，而亮度较低的图像区域的二值化阈值则会相适应地变小!!!!!!!!!!!!!!!!!!!!!!!!!!!!!关键
# def adaptiveThreshold(src,maxValue,adaptiveMethod,thresholdType,blockSize,C,dst=None)
# maxval：Double类型的，阈值的最大值 
# adaptiveMethod：Int类型的，这里有两种选择 
# 1 —— ADAPTIVE_THRESH_MEAN_C（通过平均的方法取得平均值） 
# 2 —— ADAPTIVE_THRESH_GAUSSIAN_C(通过高斯取得高斯值) 
# 不过这两种方法最后得到的结果要减掉参数里面的C值
def OnAdaptive(blockSize):
    img_adp_mean = cv.adaptiveThreshold(img_roi, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, blockSize, 0)
    img_adp_gauss = cv.adaptiveThreshold(img_roi, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, blockSize, 0)
    cv.imshow('img_adp_mean', img_adp_mean)
    cv.imshow('img_adp_gauss', img_adp_gauss)
    img_gauss_adp_mean = cv.adaptiveThreshold(img_roi_gauss, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, blockSize, 0)
    img_gauss_adp_gauss = cv.adaptiveThreshold(img_roi_gauss, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, blockSize, 0)
    cv.imshow('img_gauss_adp_mean', img_gauss_adp_mean)
    cv.imshow('img_gauss_adp_gauss', img_gauss_adp_gauss)
img_roi_gauss = cv.GaussianBlur(img_roi, (3, 3), 1) #高斯滤波后,明显噪声更少了
cv.namedWindow("adaptiveThreshold",cv.WINDOW_NORMAL)
cv.createTrackbar("value", "adaptiveThreshold", 0, 255, OnAdaptive)
cv.imshow('adaptiveThreshold', img)

# 【OTSU二值化】利用图像的灰度直方图，确定最佳的全局阈值，就是该方法的全局阈值也是自动确定的    
# OTSU会找到位于两个峰值之间的t值，以使两个类别的差异最小。
# OTSU的是通过计算类间最大方差来确定分割阈值的阈值选择算法，OTSU算法对直方图有两个峰，中间有明显波谷的直方图对应图像二值化效果比较好，而对于只有一个单峰的直方图对应的图像分割效果没有双峰的好。
ret, img_binary_ostu = cv.threshold(img_roi, 0, 255, cv.THRESH_OTSU)
cv.imshow('img_binary_ostu', img_binary_ostu)
print(ret) #结果是186，这比自己设置好多了，但是最好先看看直方图是不是双峰

cv.waitKey(0)
cv.destroyAllWindows()