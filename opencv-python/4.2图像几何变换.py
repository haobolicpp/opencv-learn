import cv2 as cv
import numpy as np
import math
from matplotlib import pyplot as plt

# 讲解图像的平移/旋转/仿射变换/透视变换等

# 【缩放】 resize函数，缩放时需指定图像插值的方法
# 函数原型：cv2.resize(InputArray src, OutputArray dst, Size, fx, fy, interpolation)Size：输出图片尺寸；fx, fy:沿x轴，y轴的缩放系数;interpolation插入方式：
# INTER_NEAREST 最近邻插值,就是用离得最近的像素值作为结果；
# INTER_LINEAR 双线性插值（默认设置）,是在 x和 y 方向根据临近的两个像素的位置进行线性插值；
# INTER_CUBIC 4x4像素邻域的双三次插值,是用某种3次方函数差值，三次样条插值；
# INTER_LANCZOS4 8x8像素邻域的Lanczos插值,是跟傅立叶变换有关的三角函数的方法。
# INTER_AREA 使用像素区域关系进行重采样。神神秘秘
# 最佳实践：INTER_AREA用于缩小，INTER_CUBIC和INTER_LINEAR用于放大
img = cv.imread('./pic/3.bmp')
img_zoom_in = cv.resize(img, None, fx=1.5, fy=1.5, interpolation = cv.INTER_CUBIC) #放大
img_zoom_out = cv.resize(img, None, fx=0.2, fy=0.2, interpolation = cv.INTER_CUBIC) #缩小
cv.imshow('img_zoom_in', img_zoom_in)
cv.imshow('img_zoom_out', img_zoom_out)

# 【放射变换】warpAffine--利用输入的变换矩阵进行变换，各种矩阵参考https://blog.csdn.net/FadeFarAway/article/details/54970189
# cv2.warpAffine(src, M, dsize[, dst[, flags[, borderMode[, borderValue]]]]) → dst
# 【原理介绍】--注意这里是对图像位置的矩阵相乘，不是卷积，最后算出来的是X/Y的实际位置
# |X|   |m00 m01 m02|  |x|   |m00*x+m01*y+m02|
# |Y| = |m10 m11 m12|* |y| = |m10*x+m11*y+m12| 最后一行是为了齐次坐标添加的，m00等构成的矩阵即为变换矩阵，注意xy指定是坐标
# |1|   |0    0   1 |  |1|   |       1       |

## 【平移】矩阵为
# |1 0 tx|
# |0 1 ty|
M = np.float32([[1,0,100],[0, 1, 200]]) #x平移100， y平移200
rows,cols = img_zoom_out.shape[0:2]
 cv.imshow('img_trans', img_trans)

## 【旋转】矩阵为
# |cosθ -sinθ 0|
# |sinθ  cosθ 0|
sita = math.pi/4
M = np.float32([[math.cos(sita), -math.sin(sita),0],[math.sin(sita), math.cos(sita), 0]]) #旋转45°
rows,cols = img_zoom_out.shape[0:2]
img_rotate1 = cv.warpAffine(img_zoom_out, M, (cols,rows), flags=cv.INTER_LINEAR,borderValue=(0,255,255)) #需指定输出图像得大小(cols,rows)
cv.imshow('img_rotate1', img_rotate1)
# 【旋转API方式】opencv提供了可缩放的旋转以及可调整的旋转中心的API  getRotationMatrix2D ：Mat getRotationMatrix2D(Point2f center, double angle, double scale)
M = cv.getRotationMatrix2D(((cols-1)/2.0,(rows-1)/2.0), 45, 1) # opencv中旋转角度是逆时针的！！
img_rotate2 = cv.warpAffine(img_zoom_out, M, (cols,rows), flags=cv.INTER_LINEAR,borderValue=(0,255,255)) #需指定输出图像得大小(cols,rows)
cv.imshow('img_rotate2', img_rotate2)

## 【仿射变换】在原图像取三个点，目标图像三个点，可以获得一个2*3变换矩阵（通过getAffineTransform），然后将图像进行放射变换，变换后原先平行的线还是平行的，
img_net = cv.imread('./pic/net.png')
cv.imshow('img_net', img_net)
src_pt = np.float32([[50, 50], [50, 100], [100, 50]])
des_pt = np.float32([[10, 100], [100, 150], [200, 10]])
rows,cols = img_net.shape[0:2]
M = cv.getAffineTransform(src_pt, des_pt)
img_affine = cv.warpAffine(img_net, M, (cols,rows), flags=cv.INTER_LINEAR,borderValue=(0,255,255)) #需指定输出图像得大小(cols,rows)
cv.imshow('img_affine', img_affine)

## 【透视变换】和放射变换类似，但需要四个点，构建3*3矩阵来进行变换，参考https://blog.csdn.net/ganbelieve/article/details/91996326
# 透视变换中原先不平行的线可能变为平行,比如本例中斜着拍摄的书，转为正的
img_per_src = cv.imread('./pic/perspective.png')
cv.imshow('img_per_src', img_per_src)
rows,cols = img_per_src.shape[0:2]
src_pt = np.float32([[67,48],[282,63],[291,363],[14,343]]) #通过画图打开得到书的四个角点坐标,从左上角顺时针
des_pt = np.float32([[0,0],[cols-1,0],[cols-1,rows-1],[0,rows-1]])
M = cv.getPerspectiveTransform(src_pt, des_pt) #创建变换矩阵
img_perspective = cv.warpPerspective(img_per_src,  M, (cols,rows)) #透视变换
cv.imshow('img_perspective', img_perspective)

cv.waitKey(0)
cv.destroyAllWindows()