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
# 【缺点】角点是像素级别的，速度较慢
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

# 【亚像素SubPixel角点检测】通常情况下，角点的真实位置并不一定在整数像素位置，因此为了获取更为精确的角点位置坐标，需要角点坐标达到亚像素（subPixel）精度
# cv.cornerSubPix ，通过对哈里斯检测到的一个角上有一堆像素，我们取它们的质心来细化它们
	# c++:void cv::cornerSubPix(
	# 	cv::InputArray image, // 输入图像
	# 	cv::InputOutputArray corners, // 角点（既作为输入也作为输出）
	# 	cv::Size winSize, // 窗口大小为NXN，是角点像素周围计算亚像素的窗口
	# 	cv::Size zeroZone, // 类似于winSize，但是总具有较小的范围，Size(-1,-1)表示忽略
	# 	cv::TermCriteria criteria // 停止优化的标准，可选的值有cv::TermCriteria::MAX_ITER 、cv::TermCriteria::EPS（可以是两者其一，或两者均选）
    #               前者表示迭代次数达到了最大次数时停止，后者表示角点位置变化的最小值已经达到最小时停止迭代
	# );
# inline  CvTermCriteria  cvTermCriteria( int type, int max_iter, double epsilon );
    # type：/* CV_TERMCRIT_ITER 和 CV_TERMCRIT_EPS 二值之一，或者二者的组合 */
    # max_iter：/* 最大迭代次数 */
    # epsilon：/* 结果的精确性 */
# 【大概原理】参考 https://www.cnblogs.com/riddick/p/8476456.html
# 目的是在窗口winSize内，求出一个亚像素精度的点(x,y)。这个点通过最小二乘法（公式怎么列的还没搞懂，大概是根据角的性质利用向量相乘为0列出的公式 TODO）求出，
# 当求出一个点后，已该点为中心，重新构建窗口，进行最小二乘计算，直到满足退出条件（迭代次数以及精度为退出条件）
# 【哈里斯角检测结果的处理】在调用亚像素角点检测前，需要对cornerHarris结果进行处理，因为它返回的一个角的检测结果中，可能对应了多个角结果，需要计算它们的质心作为最终的角检测结果。
# 如果为了简单，可采用下面的Shi-Tomas角检测算法直接算出的角点坐标。
# 这里需要对结果首先进行膨胀，然后二值化，再区域连通后返回质心坐标。
#1、膨胀--已经做了
#2、二值化
r = 1; c = 3; i = 1
ret, dist = cv.threshold(dist, 0.01*dist.max(), 255, 0)
plt.subplot(r,c,i),plt.imshow(dist, cmap = 'gray'),plt.title('dist_b'),plt.xticks([]), plt.yticks([]);i+=1
#3、区域连通
dist = np.uint8(dist)
retval, labels, stats, centroids = cv.connectedComponentsWithStats(dist) #centroids中记录了坐标
#4、创建停止条件
stopcriteria = (cv.TERM_CRITERIA_EPS+cv.TERM_CRITERIA_MAX_ITER, 50, 0.01)
#5、亚像素计算
corners = cv.cornerSubPix(img_gray, np.float32(centroids), (5,5), (-1,-1), stopcriteria)
#6、绘制亚像素点（略）
print('before subpix',centroids)
print('after subpix',corners)

plt.show()

cv.waitKey(0)
cv.destroyAllWindows()