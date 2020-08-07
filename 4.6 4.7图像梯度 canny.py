import cv2 as cv
import numpy as np
import math
from matplotlib import pyplot as plt

# 在本章中，我们将学习： 
# - 查找图像梯度，边缘等 
# - 我们将看到以下函数：cv.Sobel()，cv.Scharr()，cv.Laplacian()等
# - 最后会看到Canny()的应用
# OpenCV提供三种类型的梯度滤波器或高通滤波器，即Sobel，Scharr和Laplacian。我们将看到他们每一种。
# 

# 【拉普拉斯】原理使用参考https://blog.csdn.net/cyf15238622067/article/details/87859887
# 利用离散二次求导，计算出像素值，二次求导可以通过卷积矩阵进行运算
# 离散函数f(x)一次求导：f' = f(x+1)-f(x)
# 二次求导：f'' = f(x+1)+f(x-1)-2f(x)
# 对于拉普拉斯来说，f(x,y)对x二次求导：f''x = f(x+1,y)+f(x-1,y)-2f(x,y)
# f(x,y)对y二次求导：f''y = f(x,y+1)+f(x,y-1)-2f(x,y)
# 拉普拉斯结果：f''x + f''y = f(x+1,y)+f(x-1,y)+f(x,y+1)+f(x,y-1)-4f(x,y)
# 卷积矩阵：
# |0 1  0|
# |1 -4 1|
# |0 1  0|
# 【结论】Laplace算子是二阶导数操作，其在强调图像素中灰度不连续的部分的同时也不在强调灰度值连续的部分。这样会产生一个具有很明显的灰度边界
# 一般用来对图像边缘进行锐化（原图像-拉普拉斯图）。
# 
# Laplacian( src_gray, dst, ddepth, kernel_size, scale, delta, BORDER_DEFAULT );
# ddepth，输出图像深度，因为输入图像一般为CV_8U，为了避免数据溢出，输出图像深度应该设置为CV_16S或CV_64F，这是因为白色到黑色的过渡斜率为负值！！
# kernel_size，filter mask的规模，我们的mask时3x3的，所以这里应该设置为3
# scale,delta,BORDER_DEFAULT，默认设置就好


img = cv.imread('./pic/3-sub.png',0)
#img = cv.imread('./pic/wb.png',0)
cv.imshow('img', img)
#img_laplac = cv.Laplacian(img, cv.CV_64F)#不加高斯预处理
#cv.imshow('img_laplac_no_gaus', img_laplac)
# 加上高斯预处理
img = cv.GaussianBlur(img, (5,5), 1) # 图像处理之前一般会进行高斯去噪，高斯去噪是为了防止把噪点也检测为边缘。
cv.imshow('GaussianBlur', img)
img_laplac = cv.Laplacian(img, cv.CV_64F)
#cv.imshow('img_laplac_gaus', img_laplac) #imshow偷着转了，所以显示的图像不准确
img_laplac = cv.convertScaleAbs(img_laplac) #取绝对值
img_laplac = cv.subtract(img, img_laplac) #原图像减去拉普拉斯值，进行锐化
cv.imshow('img_laplac_gaus', img_laplac) #显示锐化后的图像，效果是显示的图像边缘更黑了

#imshow的转换规则：
#1，如果原始图片是8位无符号整数，就按照原来的数字进行显示。也就是数字范围是[0,255]
#2，如果原始图片是16位无符号整数或者32位整数，就除以256进行显示。也就是说0到256*256的范围被压缩到0到255。
#3，如果图片是32位或者64位的浮点类型数据，那么像素值就会乘以255。也就是说，0到1的范围被映射到0到255.

# 【sobel算子】https://www.cnblogs.com/sdu20112013/p/11608469.html
# sobel算子模拟一阶求导
# 卷积矩阵（x方向）：
# |-1 0 1|
# |-2 0 2| 即右边像素减去左边像素和为中间像素的数值
# |-1 0 1|
#
# Sobel(src, ddepth, dx, dy[, dst[, ksize[, scale[, delta[, borderType]]]]]) -> dst dx dy：0不起作用1起作用； ksize：只能取1/3/5/7
sobel_x = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=3)
sobel_y = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=3)
cv.imshow('sobel_x', sobel_x)
cv.imshow('sobel_y', sobel_y)
#合并x y
#addWeighted(InputArray src1, double alpha, InputArray src2,double beta, double gamma, OutputArray dst, int dtype=-1);
# 第2个参数，图片1的融合比例
# 第3个参数，输入图片2
# 第4个参数，图片2的融合比例
# 第5个参数，偏差
img_add = cv.addWeighted(sobel_x, 1, sobel_y, 1, 0)
cv.imshow('img_add', img_add)


## 【Scharr滤波器】核大小为3，本滤波器比sobel更好
img_scharr_x = cv.Scharr(img, cv.CV_64F, 1, 0)
img_scharr_y = cv.Scharr(img, cv.CV_64F, 0, 1)
cv.imshow('img_scharr_x', img_scharr_x)
cv.imshow('img_scharr_y', img_scharr_y)

# CV_64F 转 CV_8U 最终结果图像很淡，因为取得是绝对值，负值部分都比较小
# 【推测】在上面拉普拉斯变换的结果中，负值部分在调用imshow的时候，直接被转为了255！！！
# 拉普拉斯毕竟不是二值图，找边缘的话还得用二值图
abs_64f = np.absolute(img_laplac)
img_8u = np.uint8(abs_64f)
cv.imshow('img_8u', img_8u)

# 【canny边缘检测】内部完成如下步骤： https://blog.csdn.net/duwangthefirst/article/details/79971212
# 滤掉噪声 比如高斯滤波
# 计算梯度 比如用sobel算子算出梯度
# 非极大值抑制，就是要在局部像素点中找到变换最剧烈的一个点,这样就得到了更细的边缘。抑制是设置为0
# 双阈值检测和连接边缘：关于2个阈值参数：
#  低于阈值1的像素点会被认为不是边缘；
#  高于阈值2的像素点会被认为是边缘；
#  在阈值1和阈值2之间的像素点,若与第2步得到的边缘像素点相邻，则被认为是边缘，否则被认为不是边缘。
# oid cv::Canny	(	InputArray	image,	（输入图像：8-bit）
# OutputArray	edges,	（输出边缘图像：单通道，8-bit，size与输入图像一致）
# double	threshold1,	（阈值1）
# double	threshold2,	（阈值2）
# int	apertureSize=3,	（Sober算子大小）
# bool	L2gradient=false	（是否采用更精确的方式计算图像梯度）默认false,true更精确
def OnCanny(x):
    threshold1 = cv.getTrackbarPos("threshold1", "canny")
    threshold2 = cv.getTrackbarPos("threshold2", "canny")
    img_canny = cv.Canny(img, threshold1, threshold2)
    cv.imshow('img_canny', img_canny)
cv.namedWindow("canny")
cv.createTrackbar("threshold1", "canny", 0, 255, OnCanny) #创建多个滑条
cv.createTrackbar("threshold2", "canny", 0, 255, OnCanny)
cv.setTrackbarPos("threshold1", "canny",100)
cv.setTrackbarPos("threshold2", "canny",200)
OnCanny(1)

cv.waitKey(0)
cv.destroyAllWindows()