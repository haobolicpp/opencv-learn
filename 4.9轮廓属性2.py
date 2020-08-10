import cv2 as cv
import numpy as np
import math
from matplotlib import pyplot as plt

#坚实度，等效直径，掩模图像，平均强度等

# 【长宽比】轮廓边界矩形的宽度与高度之比
# 【范围】轮廓区域与边界矩形区域的比值（面积）
# 【坚实度】 轮廓区域面积/轮廓凸包面积
# 【等效直径】 与轮廓面积相同的圆的直径 equi_diameter = np.sqrt(4*area/np.pi)
# 【取向】物体指向的角度，例如拟合椭圆找出长轴与短轴长度？？？

# 【构建轮廓点的腌膜图像】--查找轮廓点后，返回的是轮廓的坐标，我们可以根据坐标构建腌膜图像(0和255的二值图)，来利用腌膜进行一些操作
img_src = cv.imread('./pic/3-sub.png', 0)
img = cv.GaussianBlur(img_src, (3,3), 1)
ret, img = cv.threshold(img, 200, 255, cv.THRESH_BINARY)
contours, hierachy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
img_mask = np.zeros(img.shape, np.uint8) #全黑
cv.drawContours(img_mask, contours, 67, 255, -1) #绘制轮廓并填充轮廓内部白色，这就是腌膜图像
cv.imshow('img_mask', img_mask)

# 【利用腌膜-像素点】轮廓坐标
pixelpts = cv.findNonZero(img_mask) 

# 【利用腌膜-查找原轮廓区域内的最大最小像素点】
min_val, max_val, min_loc, max_loc = cv.minMaxLoc(img_src, mask=img_mask)

# 【利用腌膜-计算轮廓区域内的平均颜色强度】
mean_val = cv.mean(img_src, mask=img_mask)

# 【极端点】？？

# 【凸性缺陷convexity defects】-https://blog.csdn.net/sufangqi123/article/details/47664021
# convexity defects如链接中所述，凸性缺陷指的是手边缘外和凸轮廓之间的部分，有四个特征量：起始点、结束点、距离凸包(convexity hull)最远的点以及到最远点的近似距离
# cpp返回值是vector<vector<Vec4i>>，python返回的是三维矩阵(实际是二维)
hull = cv.convexHull(contours[67], returnPoints=False) # 返回在轮廓中的索引
defects = cv.convexityDefects(contours[67], hull)
print(defects)

#【图像中某个点到轮廓的最短距离】它返回的距离，点在轮廓线外时为负，点在轮廓线内时为正，点在轮廓线上时为零
# 函数第二个参数：某个点
# 函数第三个参数：measureDist。如果它是真的，它会找到有符号的距离。如果为假，则查找该点是在轮廓线内部还是外部(分别返回+1、-1和0)
dist = cv.pointPolygonTest(contours[67], (0,0), True)
print(dist)

# 【形状匹配】--能够比较两个形状或两个轮廓，是根据Hu矩的数值计算出来的，并返回一个显示相似性的度量
# 返回的度量越低匹配越好，因为是通过Hu矩匹配的，Hu矩具有旋转、平移、缩放的七个矩，因此图形发生旋转缩放等也可以匹配出来
# 【使用】由Hu矩组成的特征量对图片进行识别，优点就是速度很快，缺点是识别率比较低。因此Hu不变矩一般用来识别图像中大的物体，对于物体的形状描述得比较好，图像的纹理特征不能太复杂。
# ret = cv.matchShapes(cnt1,cnt2,1,0.0) 参数一和二为轮廓数据，参数三是使用的方法，默认即可，参数四默认0即可
# 例子步骤：
# 1.在模板图像中二值化后找到轮廓
# 2.轮廓拟合圆，找到轮廓面积等效直径和圆直径在误差范围内的轮廓，作为模板轮廓
# 3.在目标图像中二值化，查找轮廓，进行拼配，找到最小的那个轮廓画出来
# 略 TODO
# img_template = img_src[1063:1127, 1639:1733]
# plt.subplot(131),plt.imshow(img_template),plt.title('img_template')
# plt.show()
# # ret, img_template = cv.threshold(img_template, 0, 255, cv.THRESH_OTSU)
# # contours, hierachy = cv.findContours(img_template, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

# # img_template_contours = np.zeros(img_template.shape, np.uint8)
# # cv.drawContours()
# # plt.show()


cv.waitKey(0)
cv.destroyAllWindows()