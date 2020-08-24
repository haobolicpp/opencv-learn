import cv2 as cv
import numpy as np
import math
from matplotlib import pyplot as plt

## 有如下功能：
 # - 距离变换--distanceTransform
 # - 连通组件--connectedComponents
 # - 我们将学习使用分水岭算法实现基于标记的图像分割 
 # - 我们将看到：cv.watershed()
 # 直接参考https://blog.csdn.net/yukinoai/article/details/88575861?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-4.channel_param&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-4.channel_param
 # 讲的很好
 

# 【连通组件】 connectedComponents  参考 https://zhuanlan.zhihu.com/p/66362447
# 【原理】算法的实质是扫描二值图像的每个像素点，对于像素值相同的而且相互连通分为相同的组(group),最终得到图像中所有的像素连通组件
# 【函数】retval, labels =cv2.connectedComponents(image, connectivity, ltype)  
#image, // 输入二值图像，黑色背景
#connectivity = 8, // 连通域，默认是8连通
#ltype = CV_32, // 输出的labels类型，默认是CV_32S 输出
#retval, // 索引个数
#labels, // 输出的标记图像，背景index=0,其他连通区域index>0
# 【函数2】 retval, labels, stats, centroids=cv2.connectedComponentsWithStats(image, connectivity, ltype )
# stats结构信息：
# CC_STAT_LEFT: 连通组件外接矩形左上角坐标的X位置信息
# CC_STAT_TOP: 连通组件外接左上角坐标的Y位置信息
# CC_STAT_WIDTH: 连通组件外接矩形宽度
# CC_STAT_HEIGHT: 连通组件外接矩形高度
# CC_STAT_AREA: 连通组件的面积大小，基于像素多少统计
# Centroids输出的是每个连通组件的中心位置坐标(x, y)
r = 1; c = 3; i = 1
img_conn = cv.imread('./pic/opencv.png', 0)
plt.subplot(r,c,i),plt.imshow(img_conn, cmap = 'gray'),plt.title('img_conn_OTSU'),plt.xticks([]), plt.yticks([]);i+=1
ret, img_conn = cv.threshold(img_conn, 200, 255, cv.THRESH_BINARY_INV) #二值化
plt.subplot(r,c,i),plt.imshow(img_conn, cmap = 'gray'),plt.title('img_conn_OTSU'),plt.xticks([]), plt.yticks([]);i+=1
num_labels,labels = cv.connectedComponents(img_conn, connectivity=8, ltype=cv.CV_32S) #进行连通
# 构造颜色colors数组
colors = []
for index in range(num_labels):
    bc = np.random.randint(0, 256) #随机色值
    gc = np.random.randint(0, 256)
    rc = np.random.randint(0, 256)
    colors.append((bc, gc, rc))
colors[0] = (0, 0, 0)
# 画出连通图
h, w = img_conn.shape
image_color = np.zeros((h,w,3), dtype=np.uint8) #黑色图像
for row in range(h):
    for col in range(w):
        image_color[row,col] = colors[labels[row,col]]
plt.subplot(r,c,i),plt.imshow(cv.cvtColor(image_color, cv.COLOR_BGR2RGB)),plt.title('image_color'),plt.xticks([]), plt.yticks([]);i+=1
plt.show()

 # 【距离变换】distanceTransform 用于计算图像中每一个非零点距离离自己最近的零点的距离，图像上越亮的点，代表了离零点的距离越远。
 # 【原理】大概是只沿着xy方向找，不考虑斜着的方式。
 # cv2.distanceTransform(src, distanceType, maskSize)
#  distanceType：计算距离的方式，自带7种
#                         DIST_L1      = 1,   //!< distance = |x1-x2| + |y1-y2|
#                         DIST_L2      = 2,   //!< the simple euclidean distance
#                         DIST_C       = 3,   //!< distance = max(|x1-x2|,|y1-y2|)
#                         DIST_L12     = 4,   //!< L1-L2 metric: distance = 2(sqrt(1+x*x/2) - 1))
#                         DIST_FAIR    = 5,   //!< distance = c^2(|x|/c-log(1+|x|/c)), c = 1.3998
#                         DIST_WELSCH  = 6,   //!< distance = c^2/2(1-exp(-(x/c)^2)), c = 2.9846
#                         DIST_HUBER   = 7    //!< distance = |x|<c ? x^2/2 : c(|x|-c/2), c=1.345
# maskSize：蒙板尺寸，3种
#                         DIST_MASK_3 = 3, //!< mask=3
#                         DIST_MASK_5 = 5, //!< mask=5
#                         DIST_MASK_PRECISE = 0 //!< mask=0
r = 1; c = 3; i = 1
img_src = cv.imread('./pic/coin.png')
img_src_gray = cv.cvtColor(img_src, cv.COLOR_BGR2GRAY)
ret, img_ostu = cv.threshold(img_src_gray, 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU) #ostu
plt.subplot(r,c,i),plt.imshow(img_ostu, cmap = 'gray'),plt.title('img_ostu'),plt.xticks([]), plt.yticks([]);i+=1
img_dis1 = cv.distanceTransform(img_ostu, cv.DIST_L1, cv.DIST_MASK_3)
img_dis1 = cv.convertScaleAbs(img_dis1) #取绝对值
img_dis2 = cv.distanceTransform(img_ostu, cv.DIST_L1, cv.DIST_MASK_5)
img_dis2 = cv.convertScaleAbs(img_dis2) #取绝对值
plt.subplot(r,c,i),plt.imshow(img_dis1, cmap = 'gray'),plt.title('img_dis1'),plt.xticks([]), plt.yticks([]);i+=1
plt.subplot(r,c,i),plt.imshow(img_dis2, cmap = 'gray'),plt.title('img_dis2'),plt.xticks([]), plt.yticks([]);i+=1
plt.show()

 # 分水岭算法的基本原理 基于标记图像的分水岭算法较好的实现了复杂背景下前景目标分割 TODO
 # cv.watershed需要输入一个标记图像，图像的像素值为32位有符号正数（CV_32S类型）,每个非零像素代表一个标签(可以通过区域连通构建)
 # 该标记图像中0值代表unknown区域，是需要漫水填充计算的区域。
 # 漫水过程【推测】：对原图像求梯度，这样边界的位置像素值最高。从水平线1开始，遍历unknown区域看看是否两个边界接触了，是的话筑起边界壁垒，水平线+1后继续。
r = 2; c = 4; i = 1
plt.subplot(r,c,i),plt.imshow(img_src, cmap = 'gray'),plt.title('img_src'),plt.xticks([]), plt.yticks([]);i+=1
 #1、二值化--上面做了
plt.subplot(r,c,i),plt.imshow(img_ostu, cmap = 'gray'),plt.title('img_ostu'),plt.xticks([]), plt.yticks([]);i+=1
#2、去噪声
kernel = np.ones((3,3), np.uint8)
img_opening = cv.morphologyEx(img_ostu, cv.MORPH_OPEN, kernel, iterations=2)
plt.subplot(r,c,i),plt.imshow(img_opening, cmap = 'gray'),plt.title('img_opening'),plt.xticks([]), plt.yticks([]);i+=1
#3、进行距离变换--这里要进行膨胀操作，因为不膨胀的话距离变换后图像白色地方亮度很低，因为距离太小了，膨胀可以增加白色部分中心的距离
img_dilat = cv.dilate(img_opening, kernel, iterations=1)
plt.subplot(r,c,i),plt.imshow(img_dilat, cmap = 'gray'),plt.title('img_dilat'),plt.xticks([]), plt.yticks([]);i+=1
img_dis = cv.distanceTransform(img_dilat, cv.DIST_L1, cv.DIST_MASK_3)
plt.subplot(r,c,i),plt.imshow(img_dis, cmap = 'gray'),plt.title('img_dis'),plt.xticks([]), plt.yticks([]);i+=1
#4、距离变换后二值化，得到前景图像
ret, img_dis_b = cv.threshold(img_dis, 0.7*img_dis.max(), 255, cv.THRESH_BINARY)
plt.subplot(r,c,i),plt.imshow(img_dis_b, cmap = 'gray'),plt.title('img_dis_b'),plt.xticks([]), plt.yticks([]);i+=1
#5、找到硬币边缘所在的区域为未知区域--膨胀后的减距离变换后二值化的
img_dis_b = np.uint8(img_dis_b)
img_unknown = cv.subtract(img_dilat, img_dis_b)
plt.subplot(r,c,i),plt.imshow(img_unknown, cmap = 'gray'),plt.title('img_unknown'),plt.xticks([]), plt.yticks([]);i+=1
#6、对硬币前景区域进行连通标记，得到mark图像
numbers, img_mark = cv.connectedComponents(img_dis_b)
img_mark = img_mark + 1 #对所有index值+1，因为背景是0，分水岭算法会认为无效区域
img_mark[img_unknown==255] = 0 # 将位置区域部分设置为0 np的一种用法
#7、最后，使用分水岭算法,结果中边界区域被标记为-1
img_water = cv.watershed(img_src, img_mark)
img_src[img_water==-1] = [255,0,0] #边界像素标记255,B G R
cv.imshow('img_src_end',img_src)

plt.show()
cv.waitKey(0)
cv.destroyAllWindows()