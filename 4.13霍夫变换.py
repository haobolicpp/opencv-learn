import cv2 as cv
import numpy as np
import math
from matplotlib import pyplot as plt

# 包含如下内容：
# 霍夫线变换-包括标准变换和累积概率变换,这里python分了两个函数(cv.HoughLines和HoughLinesP)，c++只有一个cvHonghLines2
# 霍夫圆变换

# 【霍夫线变换-标准】--在对图像进行边缘提取（梯度如sobel/laplacian/canny等），然后对所有的边缘像素点进行(ρ, θ)的矩阵累加计算,参考https://blog.csdn.net/qq_39861376/article/details/82119472
# 1、一条直线y=kx+b 可以表示成ρ=xcosθ+ysinθ，这里ρ为原点到直线的垂直距离，θ为该垂线与x轴的角度，范围0-180，逆时针
# 2、对边缘图像中的每个边缘像素点，计算[[ρ,θ]]的二维累加矩阵，如果精度为1°，那么计算θ在0-180之间的所有ρ值，并在矩阵中计数累加（即投票）
# 3、每个像素点都计算一遍二维矩阵，得到最终的累加结果，然后通过一个投票数阈值来确定多个[ρ,θ]直线
# HoughLines(image, rho, theta, threshold[, lines[, srn[, stn]]]) -> lines
# 第一个参数，输入图像应该是二进制图像
# 第二个参数，ρ精度，如1
# 第三个参数，θ精度，单位°，如1°
# 第四个参数，投票的阈值，这个决定了构成一条线的最少像素个数
# 返回找到的线[[ρ,θ]]矩阵
img_src = cv.imread('./pic/1.jpg', 0)
img = cv.GaussianBlur(img_src, (5,5), 0)
img = cv.Canny(img, 200, 255)
lines = cv.HoughLines(img, 1, np.pi/180, 200)
for line in lines:
    rho,theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(b)) #这里1000指从(x0,y0)点在线上延伸1000的距离，然后求x1,y1坐标
    y1 = int(y0 - 1000*(a))
    x2 = int(x0 - 1000*(b))
    y2 = int(y0 + 1000*(a))
    cv.line(img_src,(x1,y1),(x2,y2),(0,0,255),2)
cv.imshow('img_src1', img_src)
# 【霍夫线变换-概率】概率霍夫变换是我们看到的霍夫变换的优化。它没有考虑所有要点。取而代之的是，它仅采用随机的点子集，足以进行线检测。只是我们必须降低阈值
# 在HoughLines基础上，添加了两个参数：
# minLineLength：线段以像素为单位的最小长度，根据应用场景设置 
# maxLineGap：同一方向上两条线段判定为一条线段的最大允许间隔（断裂），超过了设定值，则把两条线段当成一条线段，值越大，允许线段上的断裂越大，越有可能检出潜在的直线段
# 返回值：lines：c++里是一个vector<Vec4i>,Vec4i是一个包含4个int数据类型的结构体，[x1,y1,x2,y2],可以表示一个线段
lines = cv.HoughLinesP(img, 1, np.pi/180, 100, 50, 50)
for line in lines:
    x1,y1,x2,y2 = line[0]
    cv.line(img_src,(x1,y1),(x2,y2),(0,255,0),2)
cv.imshow('img_src2', img_src)


# plt.subplot(144),plt.imshow(img_back, cmap = 'gray'),plt.title('img_back_low'),plt.xticks([]), plt.yticks([])
# plt.show()


cv.waitKey(0)
cv.destroyAllWindows()