import cv2 as cv
import numpy as np
import math
from matplotlib import pyplot as plt

img = cv.imread('./pic/3-sub.png')
# 腐蚀 用一个5x5内核，取局部极小值替换中心元素的值
kernel = np.ones((5,5), np.uint8) #构建都是1的5*5矩阵
img_erode = cv.erode(img, kernel)
cv.imshow('img_erode', img_erode)

# 膨胀 取局部极大值替换中心元素的值
img_dilate = cv.dilate(img, kernel)
cv.imshow('img_dilate', img_dilate)

# 先腐蚀再膨胀
img_fs_pz = cv.dilate(img_erode, kernel)
cv.imshow('img_fs_pz', img_fs_pz)

# 先膨胀再腐蚀
img_pz_fs = cv.erode(img_dilate, kernel)
cv.imshow('img_pz_fs', img_pz_fs)

cv.imshow('img', img)

cv.waitKey(0)
cv.destroyAllWindows()