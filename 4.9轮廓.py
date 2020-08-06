import cv2 as cv
import numpy as np
import math
from matplotlib import pyplot as plt

## 查找轮廓 cv.findContours()
## 绘制轮廓 cv.drawContours()
## 轮廓特征 面积，周长，质心，边界框等

## 【轮廓】轮廓可以简单地解释为连接具有相同颜色或强度的所有连续点（沿边界）的曲线
# 为了获得更高的准确性，请使用二进制图像。因此，在找到轮廓之前，请应用阈值或canny边缘检测。
# 从OpenCV 3.2开始，findContours()不再修改源图像。
# 在OpenCV中，找到轮廓就像从【【黑色背景】】中找到【【白色物体】】。因此请记住，【【要找到的对象应该是白色，背景应该是黑色】】。


# 【查找轮廓】
# findContours( InputOutputArray image, OutputArrayOfArrays contours,OutputArray hierarchy, int mode,int method, Point offset=Point());
# 第一个参数：image，单通道图像矩阵，可以是灰度图，但更常用的是二值图像，一般是经过Canny、拉普拉斯等边缘检测算子处理过的二值图像；
# 第二个参数：contours，是一个向量，并且是一个双重向量，向量内每个元素保存了一组由连续的Point点构成的点的集合的向量，每一组Point点集就是一个轮廓。vector<vector<Point>> contours
# 第三个参数：vector<Vec4i> hierarchy，向量内每一个元素包含了4个int型变量；typedef Vec<int, 4> Vec4i; 这四个变量，分别表示当前轮廓的后一个轮廓、前一个轮廓、父轮廓、内嵌轮廓的索引编号
# 第四个参数：CV_RETR_EXTERNAL 只检测最外围轮廓，包含在外围轮廓内的内围轮廓被忽略
#           CV_RETR_LIST   检测所有的轮廓，包括内围、外围轮廓，但是检测到的轮廓不建立等级关系，彼此之间独立，没有等级关系，这就意味着这个检索模式下不存在父轮廓或内嵌轮廓
#           CV_RETR_CCOMP  检测所有的轮廓，但所有轮廓只建立两个等级关系，外围为顶层，若外围内的内围轮廓还包含了其他的轮廓信息，则内围内的所有轮廓均归属于顶层
#           CV_RETR_TREE， 检测所有轮廓，所有轮廓建立一个等级树结构
# 第五个参数：
#           取值一：CV_CHAIN_APPROX_NONE 保存物体边界上所有连续的轮廓点到contours向量内
#           取值二：CV_CHAIN_APPROX_SIMPLE 仅保存轮廓的拐点信息，把所有轮廓拐点处的点保存入contours向量内，拐点与拐点之间直线段上的信息点不予保留
#           取值三和四：CV_CHAIN_APPROX_TC89_L1，CV_CHAIN_APPROX_TC89_KCOS使用teh-Chinl chain 近似算法
# 第六个参数：Point偏移量，所有的轮廓信息相对于原始图像对应点的偏移量，相当于在每一个检测出的轮廓点上加上该偏移量，并且Point还可以是负值

img_rgb = cv.imread('./pic/3.bmp')
img_rgb = img_rgb[1063:1127, 1639:1733]
img = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
img = cv.GaussianBlur(img, (3,3), 1)
#ret, img_binary = cv.threshold(img, 0, 255, cv.THRESH_OTSU) #ostu
#img_binary = cv.Laplacian(img, cv.CV_32F),img_binary = cv.convertScaleAbs(img_binary) #TODO 拉普拉斯不会用
img_binary = cv.Canny(img, 100, 200) #canny
contours, hierachy = cv.findContours(img_binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

# 【绘制轮廓】
img_src_copy1 = img_rgb.copy()
img_src_copy2 = img_rgb.copy()
cv.drawContours(img_src_copy1, contours, -1, (0,255,0), 1) #绘制所有轮廓
cv.drawContours(img_src_copy2, contours[1], -1, (0,255,0), 1) #绘制下标4的轮廓

plt.subplot(131),plt.imshow(img_binary,cmap='gray'),plt.title('img_binary')
plt.subplot(132),plt.imshow(img_src_copy1),plt.title('img_src_copy1')
plt.subplot(133),plt.imshow(img_src_copy2),plt.title('img_src_copy2')
plt.show()

## 【轮廓特征】
# 【特征矩】

cv.waitKey(0)
cv.destroyAllWindows()