import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# 颜色空间--这里主要介绍HSV空间， 参考https://www.cnblogs.com/kekeoutlook/p/11123094.html
# 一般对颜色空间的图像进行有效处理都是在 HSV 空间进行的，HSV(色调 Hue,饱和度 Saturation,亮度 Value)六角锥体模型，其中H色调部分就对应了颜色的变化，
# 这样做的好处是只对该通道进行处理即可，不然RGB通道下蓝色的表达方式太多了不好处理。
# 取值范围：H [0, 179] S [0, 255] V [0, 255]，例如蓝色的H范围是100~124，S范围43~255，V范围46~255
# HSV图像的一个像素即为H、S、V
img = cv.imread('./pic/capture.png')
img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
cv.imshow('img_hsv', img_hsv)

# inRange()函数可实现二值化功能（这点类似threshold()函数），关键的是可以同时针对多通道进行操作，使用起来非常方便！
# inRange(原图像，下界阈值，上界阈值)，作用是将两个阈值之间的设置为白色255，之外的设置为黑色0，参数和threshold不太一样

# 利用HSV空间，对蓝色物体进行图像跟踪
# 【原理】利用inrange提取腌膜，将蓝色物体部分设置为白色，其他部分设置为黑色，这样在和原图进行与操作时，白色部分物体会在原图中保留，其他部分都设置为了黑色
blue_lower = np.array([100, 50, 50]) # 蓝色HSV的三个通道最低
blue_upper = np.array([130, 255, 255]) # 蓝色HSV的三个通道最高
img_3ch_binary_msk = cv.inRange(img_hsv, blue_lower, blue_upper) #二值化，蓝色物体部分设为白色,对三个通道进行
cv.imshow('img_3ch_binary_msk', img_3ch_binary_msk)
img_result = cv.bitwise_and(img, img, mask=img_3ch_binary_msk) #进行与运算，腌膜中白色部分对应的原图部分进行保留，其他部分与腌膜做与都变为黑色
cv.imshow('img_result', img_result)

cv.waitKey(0)
cv.destroyAllWindows()