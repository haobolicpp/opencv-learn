import cv2 as cv
import numpy as np
import math
from matplotlib import pyplot as plt

#学会： 
# - 使用各种低通滤镜模糊图像：平均、高斯、中位、双边、
# - 将定制的滤镜应用于图像（2D卷积）
# - 注意这里用的是卷积，不是矩阵相乘。

# 与一维信号一样，还可以使用各种低通滤波器（LPF），高通滤波器（HPF）等对图像进行滤波。
# LPF有助于消除噪声，使图像模糊等。
# HPF滤波器有助于在图像中找到边缘。

# 【平均】将图像与归一化框滤镜进行卷积来完成的。它仅获取内核区域下所有像素的平均值，并替换中心元素。
img = cv.imread('./pic/3-sub.png', 0)
img_blur = cv.blur(img, (5,5))
cv.imshow('img', img)
cv.imshow('img_blur', img_blur)
# plt.subplot(121),plt.imshow(img,cmap='gray'),plt.title('Original')
# plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(img_blur,cmap='gray'),plt.title('Blurred')
# plt.xticks([]), plt.yticks([])
# plt.show()

# 【高斯】代替盒式滤波器，使用了高斯核，核宽度和高度应为正数和奇数，还应指定X和Y方向的标准偏差，分别为sigmaX和sigmaY。如果仅指定sigmaX，则将sigmaY与sigmaX相同。
# 如果两个都为零，则根据内核大小进行计算
# 【结论】高斯滤波器采用像素周围的邻域并找到其高斯加权平均值。滤波时会考虑附近的像素。它不考虑像素是否具有几乎相同的强度。它不考虑像素是否是边缘像素。因此它也模糊了边缘，这个可以通过双边滤波解决
# 图像处理之前一般会进行高斯去噪，高斯去噪是为了防止把噪点也检测为边缘。
def OnGaussblur(x):
    kernel_size = cv.getTrackbarPos("kernel_size", "Gaussblur")
    sigmaX = cv.getTrackbarPos("sigmaX", "Gaussblur")
    img_gauss = cv.GaussianBlur(img, (kernel_size,kernel_size), sigmaX)
    cv.imshow('img_gauss', img_gauss)
cv.namedWindow("Gaussblur")
cv.createTrackbar("kernel_size", "Gaussblur", 0, 255, OnGaussblur) #创建多个滑条
cv.createTrackbar("sigmaX", "Gaussblur", 0, 255, OnGaussblur)
cv.imshow('Gaussblur', img)

## 【中位模糊】 消除图像中的椒盐噪声非常有效
# 【原理】取内核区域下所有像素的中值，并将中心元素替换为该中值
img_sault = cv.imread('./pic/sault.png', 1)
#img_sault = cv.imread('./pic/3-sub.png')
mg_m = cv.medianBlur(img_sault, 5)
cv.imshow('img_sault', img_sault)
cv.imshow('mg_m', mg_m)

## 【双边滤波】 去除噪声的同时保持边缘清晰锐利非常有效，与其他过滤器相比，该操作速度较慢，有美颜的效果
# 双边滤波器在空间中也采用高斯滤波器，但是又有一个高斯滤波器，它是像素差的函数。
# 空间的高斯函数确保仅考虑附近像素的模糊，而强度差的高斯函数确保仅考虑强度与中心像素相似的那些像素的模糊。由于边缘的像素强度变化较大，因此可以保留边缘。
# cv2.bilateralFilter(src,d,sigmaColor,sigmaSpace,borderType)
#         src: 输入图像对象矩阵,可以为单通道或多通道
#         d:用来计算卷积核的领域直径，如果d<=0，从sigmaSpace计算d
#         sigmaColor：颜色空间滤波器标准偏差值，决定多少差值之内的像素会被计算（构建灰度值模板）
#         sigmaSpace:坐标空间中滤波器标准偏差值。如果d>0，设置不起作用，否则根据它来计算d值（构建距离权重模板）
img_texture = cv.imread('./pic/3-sub.png')
#img_texture = cv.imread('./pic/texture.png', 1)
def Onbfblur(x):
    d = cv.getTrackbarPos("d", "bfblur")
    sigmaColor = cv.getTrackbarPos("sigmaColor", "bfblur")
    img_texture_bf = cv.bilateralFilter(img_texture, d, sigmaColor, 1)
    cv.imshow('img_texture_bf', img_texture_bf)
cv.namedWindow("bfblur")
cv.createTrackbar("d", "bfblur", 0, 255, Onbfblur) #创建多个滑条
cv.createTrackbar("sigmaColor", "bfblur", 0, 255, Onbfblur)
cv.imshow('img_texture', img_texture)

cv.waitKey(0)
cv.destroyAllWindows()