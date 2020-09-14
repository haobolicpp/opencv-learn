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
# 轮廓分层hierarchy：参考http://woshicver.com/FifthSection/4_9_5_轮廓分层/

img_rgb = cv.imread('./pic/3.bmp')
img_rgb = img_rgb[1063:1127, 1639:1733]
img = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
img = cv.GaussianBlur(img, (3,3), 1)
img_binary=cv.Laplacian(img, cv.CV_32F)
img_binary=cv.convertScaleAbs(img_binary)
img_binary = cv.subtract(img, img_binary) #图像锐化
cv.imshow('img_binary_锐化', img_binary)
#下面三种预处理选一
#ret,img_binary=cv.threshold(img_binary, 200, 255, cv.THRESH_BINARY)# 普通二值化
#img_binary = cv.adaptiveThreshold(img_binary, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 5, 0) #自适应阈值，效果很烂
ret, img_binary = cv.threshold(img, 0, 255, cv.THRESH_OTSU) #ostu自动全局阈值
#找轮廓
contours, hierachy = cv.findContours(img_binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

# 【绘制轮廓】
# drawContours:参数2为轮廓集合，参数3为指定轮廓集合的哪个，参数4为绘制轮廓的颜色，参数5为画笔粗细,如果为负值或CV_FILLED表示填充轮廓内部
img_src_copy1 = img_rgb.copy()
img_src_copy2 = img_rgb.copy()
cv.drawContours(img_src_copy1, contours, 3, (0,255,0), 1) #绘制所有轮廓(参数3为-1)，也可以指定哪一个
cv.drawContours(img_src_copy2, contours[7], -1, (0,255,0), 1) #绘制下标4的轮廓

plt.subplot(131),plt.imshow(img_binary,cmap='gray'),plt.title('img_binary')
plt.subplot(132),plt.imshow(img_src_copy1),plt.title('img_src_copy1')
plt.subplot(133),plt.imshow(img_src_copy2),plt.title('img_src_copy2')
plt.show()

## 【轮廓特征】
# 【图像的矩】--参考 https://blog.csdn.net/weixin_42464187/article/details/105700055   https://blog.csdn.net/qq_37207090/article/details/83986950
# 矩的离散公式：∑∑x^p*y^q*f(x,y),pq的取值决定了几阶矩
# 0阶矩，m00,即p=q=0，公式为∑∑f(x,y)表示全局灰度累加值,但它和轮廓面积相同，轮廓面积是轮廓内像素的个数，这样是不是说零阶矩是灰度为1后的累加？？？TODO
# 1阶矩,m01和m00，可以用来计算质心
# 【获得轮廓的矩】--moments接口，返回0~3阶矩的数值
cnt0 = contours[5]
M = cv.moments(cnt0) #0
print(M) #返回的是map，m00': 0.0, 'm10': 0.0, 'm01': 0.0, 'm20': 0.0,............

# 【轮廓面积】 -- 从contourArea或0阶矩给出
area1 = cv.contourArea(cnt0)
area2 = M['m00']
print(area1, area2) # 结果是相同的

# 【轮廓周长】
perimeter1 = cv.arcLength(cnt0, True) #闭合轮廓
perimeter2 = cv.arcLength(cnt0, False) #曲线
print(perimeter1, perimeter2) #54.6 52.6

# 【轮廓近似】它是Douglas-Peucker算法的实现，参考https://www.cnblogs.com/bjxqmy/p/12347265.html
img_src_copy3 = img_rgb.copy()
cntt = cv.approxPolyDP(contours[3], 20, False)
cv.drawContours(img_src_copy3, cntt, -1, (0,0,255), 1)
plt.subplot(111),plt.imshow(img_src_copy3),plt.title('img_src_copy3')
plt.show() #问题 只绘制了点，没有连接

# 【轮廓凸包】-凸曲线是始终凸出或至少平坦的曲线。如果在内部凸出，则称为凸度缺陷
#    void convexHull(InputArray points,OutputArray hull,bool clockwise =  false, bool returnPoints = true)，，最后一个参数表示返回点还是点在轮廓中的的下标索引
hull = cv.convexHull(contours[3])
img_src_copy4 = img_rgb.copy()
cv.polylines(img_src_copy4, [hull], True, (0, 255, 0)) #画多边形
plt.subplot(111),plt.imshow(img_src_copy4),plt.title('img_src_copy4')
plt.show()

# 【检查凸度】-
isconvex = cv.isContourConvex(contours[3])
print(isconvex)

# 【边界矩形】 - 两种，正矩阵、最小面积矩形 boundingRect 和 minAreaRect
x,y,w,h = cv.boundingRect(contours[3]) #直角矩形，不考虑物体旋转，它面积不是最小的
cv.rectangle(img_src_copy4,(x,y),(x+w,y+h),(0,0,255),1)
rect = cv.minAreaRect(contours[3]) #旋转矩形，最小面积矩形，返回值中包含位置及旋转角度
boxpt = cv.boxPoints(rect) #返回的是个4*2的矩阵，且是浮点型
boxpt = np.array([boxpt], np.int32) #转整形，需添加[]包裹
cv.polylines(img_src_copy4, boxpt, True, (0, 255, 0)) #画多边形
plt.subplot(111),plt.imshow(img_src_copy4),plt.title('bounding-rect')
plt.show()

# 【最小闭合圆】
img_src_copy5 = img_rgb.copy()
(x,y),radius = cv.minEnclosingCircle(contours[3])
cv.circle(img_src_copy5, (int(x),int(y)), int(radius), (0,0,255), 1)

# 【拟合椭圆】
ellipse = cv.fitEllipse(contours[7])
cv.ellipse(img_src_copy5,ellipse,(0,255,0),1)
plt.subplot(111),plt.imshow(img_src_copy5),plt.title('min-circle ellipse')
plt.show()

# 【拟合直线】 - https://blog.csdn.net/lovetaozibaby/article/details/99482973
# cv2.fitLine(InputArray  points, distType, param, reps, aeps)
# distType: 距离类型，表示点到拟合直线的距离和最小
#       cv2.DIST_USER : 用户自定义
#       cv2.DIST_L1: distance = |x1-x2| + |y1-y2|
#       cv2.DIST_L2: 欧式距离，此时与最小二乘法相同
#       cv2.DIST_C:distance = max(|x1-x2|,|y1-y2|)
#       cv2.DIST_L12:L1-L2 metric: distance = 2(sqrt(1+x*x/2) - 1))
#       cv2.DIST_FAIR:distance = c^2(|x|/c-log(1+|x|/c)), c = 1.3998
#       cv2.DIST_WELSCH: distance = c2/2(1-exp(-(x/c)2)), c = 2.9846
#       cv2.DIST_HUBER:distance = |x|<c ? x^2/2 : c(|x|-c/2), c=1.345
# param： 距离参数，跟所选的距离类型有关，值可以设置为0。
# reps, aeps： 第5/6个参数用于表示拟合直线所需要的径向和角度精度，通常情况下两个值均被设定为0.01
# 【返回值】点斜式直线，返回4*1的矩阵，[0]和[1]存放方向向量，即cosΘ和sinΘ,[2]和[3]存放直线上的一个点
# 直线 x*conΘ + y*sinΘ + b = 0，斜率为k=-cosΘ/sinΘ
rows,cols = img_src_copy5.shape[:2]
[vx,vy,x,y] = cv.fitLine(contours[1],cv.DIST_L2, 0, 0.01, 0.01)
lefty = int((-x*vy/vx) + y) #没研究 TODO
righty = int(((cols-x)*vy/vx)+y)
cv.line(img_src_copy5,(cols-1,righty),(0,lefty),(255,0,0),1)
plt.subplot(111),plt.imshow(img_src_copy5),plt.title('min-circle ellipse line')
plt.show()


cv.waitKey(0)
cv.destroyAllWindows()