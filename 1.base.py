import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('./pic/1.jpg', 0)  #0灰度
cv.imshow('1', img)
plt.imshow(img, cmap='gray',
           interpolation='bicubic')  #interpolation插值方法，bicubic：三次插值
plt.xticks([]), plt.yticks([])  # 隐藏 x 轴和 y 轴上的刻度值
plt.show()

cv.imwrite('./pic/2.jpg', img)

##绘图
#创建全黑图像
img_black = np.zeros((512, 512, 3), np.uint8)  #shape为3个轴，每个轴的元素个数为512,512,3
print(img_black.shape)
#画线
cv.line(img_black, (0, 0), (511, 511), (0, 255, 0), 5)
#画多边形
pts = np.array([[10, 5], [20, 30], [70, 200], [50, 10]], np.int32)
#pts = pts.reshape((-1,1,2))
cv.polylines(img_black, [pts], True, (0, 255, 0))
cv.imshow('paint', img_black)

##圖像基本操作
#访问像素——【opencv中图像读进来的排布是BGR】
img_color = cv.imread('./pic/4.jpg')
print(img_color[0, 0])  #[0,0,45]
#访问R分量蓝色
print(img_color[0, 0, 2])  # 45
#修改像素值
img_color[0, 0] = [0, 0, 0]
# 更好的方式————用numpy的方式,item()和itemset()接口，效率更高
img_color.itemset((0, 0, 1), 255)
img_color.item(0, 0, 1)  # [0, 255, 0]
# 图像属性
print('shape:', img_color.shape)  # 形状，轴及元素个数 [x,x,x]
print('size像素数', img_color.size)  # 像素数
print('数据类型', img_color.dtype)  # 图像数据类型
# 【图像ROI区域】
img_roi = img[
    150:200,
    150:200]  #切片，150:200是说行从150到199，第二个参数是列从150到199！！！！！！！！！！！！！！！！！！！！！！！！！
print(img_roi.shape)
cv.imshow('roi', img_roi)
# 【拆分/合并通道】
b, g, r = cv.split(img_color)  # 耗时，建议改为下面的numpy
b = img_color[:, :, 0]
g = img_color[:, :, 1]
r = img_color[:, :, 2]
img_merge = cv.merge((b, g, r))
img_merge[:, :, 2] = 0  #r通道批量置0
cv.imshow('img_merge', img_merge)

#【图像融合--通过腌膜】图例参考https://zhuanlan.zhihu.com/p/36656952
#opencv融合到leg图片上，位and：0和1与后为0，即黑色与其他色与后为黑色；位add:0和1相加后还为1，也就是黑色部分和其他颜色相加还是其他颜色
imgopencv = cv.imread('./pic/opencv.png')
imgleg = cv.imread('./pic/leg.png')
rows, cols, channels = imgopencv.shape
imgleg_roi = imgleg[0:rows, 0:cols] #要融合的ROI区域
imgopencv_gray = cv.cvtColor(imgopencv, cv.COLOR_BGR2GRAY) #转灰度
ret, imgopencv_whitebgMsk = cv.threshold(imgopencv_gray, 200, 255, cv.THRESH_BINARY) #THRESH_BINARY:转换为二值图(0,255),200为要比较的阈值，当像素值>200,设定为maxval(255),否则设置为0，操作完后，opencv的图标变为黑色，背景变为白色
imgopencv_blackbgMsk = cv.bitwise_not(imgopencv_whitebgMsk) #求反后的腌膜
imgleg_black_opencv = cv.bitwise_and(imgleg_roi, imgleg_roi, mask=imgopencv_whitebgMsk) #【关键】眼膜中>0的部分使用原图的值（白色部分），0值部分要参与运算，腌膜最后也是要参与与运算的
cv.imshow('imgleg_black_opencv', imgleg_black_opencv)
imgopencv_blackbg_colorfg = cv.bitwise_and(imgopencv, imgopencv, mask=imgopencv_blackbgMsk)#logo部分彩色原图，背景黑色
cv.imshow('imgopencv_blackbg_colorfg', imgopencv_blackbg_colorfg)
imgadd = cv.add(imgleg_black_opencv, imgopencv_blackbg_colorfg) #黑色部分add后还是原图
cv.imshow('imgadd', imgadd)
imgleg[0:rows, 0:cols] = imgadd #像素替换
cv.imshow('imgleg', imgleg)

# 计算代码执行时间
e1 = cv.getTickCount()
# 你的执行代码
e2 = cv.getTickCount()
time = (e2 - e1)/ cv.getTickFrequency() #单位秒

#python 性能优化技术
# 有几种技术和编码方法可以充分利用 Python 和 Numpy 的最大性能。这里只注明相关信息，并提供重要信息来源的链接。这里要注意的主要事情是，首先尝试以一种简单的方式实现算法。一旦它运行起来，分析它，找到瓶颈并优化它们。
# ->尽量避免在Python中使用循环，尤其是双/三重循环等。它们本来就很慢。
# ->由于Numpy和OpenCV已针对向量运算进行了优化，因此将算法/代码向量化到最大程度。
# ->利用缓存一致性。
# ->除非需要，否则切勿创建数组的副本。尝试改用视图。数组复制是一项昂贵的操作。
# ->即使执行了所有这些操作后，如果你的代码仍然很慢，或者不可避免地需要使用大循环，请使用Cython等其他库来使其更快。

cv.waitKey(0)
cv.destroyAllWindows()
