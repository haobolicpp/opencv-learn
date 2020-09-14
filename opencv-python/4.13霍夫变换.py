import cv2 as cv
import numpy as np
import math
from matplotlib import pyplot as plt

# 包含如下内容：
# 霍夫线变换-包括标准变换和累积概率变换,这里python分了两个函数(cv.HoughLines和HoughLinesP)，c++只有一个cvHonghLines2
# 霍夫圆变换-cv.HoughCircles()使用边缘的梯度信息的**Hough梯度方法**

# 【霍夫线变换-标准】--在对图像进行边缘提取（梯度如sobel/laplacian/canny等）后，然后对所有的边缘像素点进行(ρ, θ)的矩阵累加计算,参考https://blog.csdn.net/qq_39861376/article/details/82119472
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


# 【霍夫圆变换】利用降维的思想，先找圆心，再算半径。原理参考：https://blog.csdn.net/qq_37059483/article/details/77916655
# 【大概描述】：1、首先用canny找到边缘像素的二值图像
# 2、利用sobel计算梯度，沿着像素的梯度方向画线（实际没画）
# 3、线的交点位置进行累加计数，然后用阈值卡计数，得到所有圆心
# 4、根据所有边缘点到圆心的距离，用阈值卡出半径，得到圆
# C++: void HoughCircles(InputArray image,OutputArray circles, int method, double dp, double minDist, double param1=100,double param2=100, int minRadius=0, int maxRadius=0 )
# 第一个参数，InputArray类型的image，输入图像，即源图像，需为8位的灰度单通道图像。
# 第二个参数，OutputArray类型的circles，经过调用HoughCircles函数后此参数存储了检测到的圆的输出矢量，每个矢量由包含了3个元素的浮点矢量(x, y, radius)表示。
# 第三个参数，int类型的method，即使用的检测方法，目前OpenCV中就霍夫梯度法一种可以使用，它的标识符为CV_HOUGH_GRADIENT，在此参数处填这个标识符即可。
# 第四个参数，double类型的dp，用来检测圆心的累加器图像的分辨率于输入图像之比的倒数，且此参数允许创建一个比输入图像分辨率低的累加器。上述文字不好理解的话，来看例子吧。例如，如果dp= 1时，累加器和输入图像具有相同的分辨率。如果dp=2，累加器便有输入图像一半那么大的宽度和高度。
# 第五个参数，double类型的minDist，为霍夫变换检测到的圆的圆心之间的最小距离，即让我们的算法能明显区分的两个不同圆之间的最小距离。这个参数如果太小的话，多个相邻的圆可能被错误地检测成了一个重合的圆。反之，这个参数设置太大的话，某些圆就不能被检测出来了。
# 第六个参数，double类型的param1，有默认值100。它是第三个参数method设置的检测方法的对应的参数。对当前唯一的方法霍夫梯度法CV_HOUGH_GRADIENT，它表示传递给canny边缘检测算子的高阈值，而低阈值为高阈值的一半。
# 第七个参数，double类型的param2，也有默认值100。它是第三个参数method设置的检测方法的对应的参数。对当前唯一的方法霍夫梯度法CV_HOUGH_GRADIENT，它表示在检测阶段圆心的累加器阈值。它越小的话，就可以检测到更多根本不存在的圆，而它越大的话，能通过检测的圆就更加接近完美的圆形了。
# 第八个参数，int类型的minRadius,有默认值0，表示圆半径的最小值。
# 第九个参数，int类型的maxRadius,也有默认值0，表示圆半径的最大值。
img_src = cv.imread('./pic/1.jpg', 0)
img = cv.GaussianBlur(img_src, (5,5), 0)
circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT, 1, 20, param1=50,param2=30)#结果是[[[250 202  24][ 74  70  11]]]，1行n列三个像素点
circles = np.uint16(np.around(circles)) #around四舍五入，最后转为整形
print(circles)
for i in circles[0,:]: #circles[0,:]取后面[[250 202  24][ 74  70  11]]……
    # 绘制外圆
    cv.circle(img,(i[0],i[1]),i[2],(0, 255, 0),2)
    # 绘制圆心
    cv.circle(img,(i[0],i[1]),2,(0, 255, 0),3)
cv.imshow('HoughCircles',img)


# plt.subplot(144),plt.imshow(img_back, cmap = 'gray'),plt.title('img_back_low'),plt.xticks([]), plt.yticks([])
# plt.show()


cv.waitKey(0)
cv.destroyAllWindows()