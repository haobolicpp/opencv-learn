import cv2 as cv
import numpy as np
import math
from matplotlib import pyplot as plt

# 图像金字塔在后面图像分割、Sift特征点等比较重要
# 主要包括高斯金字塔、拉普拉斯金字塔、DOG金字塔等等
# - cv.pyrUp()上采样，放大图像
# - cv.pyrDown()下采样，下采样缩小图像

## 【高斯金字塔】
#下降采样时，先要用高斯核对G1层图像进行卷积，然后删除所有偶数行和偶数列，得到缩小后的图像
#上采样，图像首先在每个维度上扩大为原来的两倍，新增的行（偶数行）以0填充。然后给指定的滤波器进行卷积
#上下采样是不可逆的，丢失了很多信息，丢失的信息即构成了拉普拉斯金字塔
img = cv.imread('./pic/3-sub.png',0)
G = img.copy()
img_gauss = [G]
for i in range(6):
    G = cv.pyrDown(G)
    img_gauss.append(G)
    imgname = 'gauss%d'%i
    cv.imshow(imgname, img_gauss[i])

## 【拉普拉斯金字塔】
#通过G层高斯金字塔图像-G+1层做UP操作后的图像得到的残差图像即可构成拉普拉斯金字塔
img_laplac = [img_gauss[5]]
for i in range(5,0,-1):
    G1 = img_gauss[i-1]
    G = cv.pyrUp(img_gauss[i]) #!!!!!TODO 未解决，大小不一致会崩溃！！！！！！！！！！！！！！！！！
    Glapl = cv.subtract(G, G1)
    img_laplac.append(Glapl)
    imgname = 'laplac%d'%i
    cv.imshow('laplac'+str(i), Glapl)

cv.waitKey(0)
cv.destroyAllWindows()