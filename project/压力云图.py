import cv2 as cv
import numpy as np
import math
from matplotlib import pyplot as plt

## 创建压力云图

# 创建高斯核 g(x, y, sigma) = exp(-(x**2+y**2)/(2*sigma**2))/(2*pi*sigma**2) 这里x是核坐标与中心坐标的差值，参考:http://www.ruanyifeng.com/blog/2012/11/gaussian_blur.html
# https://blog.csdn.net/nlite827109223/article/details/90697377 https://www.cnblogs.com/elliottzheng/p/6616971.html
# 计算出的高斯核，总和为1
def create_gauss_kernel(size, sigma):
    kernel = np.zeros((size, size))
    center = size // 2 #中心
    sigma2 = 2*sigma**2

    for i in range(size):
        for j in range(size):
            x = i - center
            y = j - center
            kernel[i, j] = np.exp(-(x**2+y**2)/sigma2)
    kernel = kernel/(2*np.pi*sigma**2) #最后做了除法
    sumval = np.sum(kernel) #总和为1的计算量
    kernel = kernel/sumval
    return kernel

#【方法一】：创建大的高斯核，将核心放到压力点上，然后计算核内其他位置的像素数值
# size = 301
# gauss_kernel = create_gauss_kernel(size, size/3)
# print(gauss_kernel)
# print(np.sum(gauss_kernel))

# # np.set_printoptions(threshold=np.inf)
# # print(gauss_kernel1)

# def OnCall(x):
#     global size
#     offset = cv.getTrackbarPos("pic2_offset", "press")
#     H1 = cv.getTrackbarPos("gauss_kernel1_H", "press")

#     # 除以中心点的值
#     gauss_kernel1 = gauss_kernel/gauss_kernel[size//2, size//2] *H1
#     gauss_kernel1 = np.uint8(gauss_kernel1)
#     gauss_kernel2 = gauss_kernel/gauss_kernel[size//2, size//2] *100
#     gauss_kernel2 = np.uint8(gauss_kernel2)

#     #对高斯进行反向 比如从中心130到外围120，变为中心120到外围130

    
#     gauss_kernel_white1 = gauss_kernel1.min()+(gauss_kernel1.max()-gauss_kernel1.min())/5
#     gauss_kernel_white2 = gauss_kernel2.min()+(gauss_kernel2.max()-gauss_kernel2.min())/5

#     # 创建图像
#     img = np.zeros((512, 512, 3), np.uint8)
#     img[:, :, 0]= 0 #H
#     img[:, :, 1]= 0 #S 0为白色图像
#     img[:, :, 2]= 255 #V 

#     #一个压力点
#     img[0:size, 0:size, 0]= gauss_kernel1 #H
#     img[0:size, 0:size, 1]= 255 #S
#     #外圈白色
#     img[0:size, 0:size, 1][img[0:size, 0:size,0]<gauss_kernel_white1] = 0 #这个可以这么理解，前面img[]返回的是二维数组，后面条件也是二维数组，两个数组必须大小一致.
#     img[0:size, 0:size, 0][img[0:size, 0:size,0]<gauss_kernel_white1] = 0

#     #两个压力点重合
#     #【叠加方法】
#     # img[offset:size+offset, offset:size+offset, 0]= img[offset:size+offset, offset:size+offset, 0]+gauss_kernel2 #H
#     # img[offset:size+offset, offset:size+offset, 1]=255
#     #【取最值方法】
#     # for i in range(size):
#     #     for j in range(size):
#     #         if gauss_kernel2[i,j] >= img[i+offset, j+offset, 0]: #高斯交汇处，取较大数值
#     #             img[i+offset, j+offset, 0] = gauss_kernel2[i,j]
#     #             if (img[i+offset, j+offset, 0] > gauss_kernel_white2): #外圈白色
#     #                 img[i+offset, j+offset, 1] = 255

#     #颜色顺序反向 
#     #img[:,:,0] = 170 - img[:,:,0]
#     img = cv.cvtColor(img, cv.COLOR_HSV2BGR)
#     cv.imshow('press', img)

# cv.namedWindow("press",cv.WINDOW_NORMAL)
# cv.createTrackbar("pic2_offset", "press", 0, 255, OnCall)
# cv.createTrackbar("gauss_kernel1_H", "press", 0, 170, OnCall)

# 【方法二】创建高斯核，长宽分别为图像长宽的两倍，然后按中心压力值向四周衰减，然后和原始图像进行叠加，得到最终的压力值图像
# 最后对所有压力值进行颜色映射
# 是否插值等验证后看效果
image_w = 512
image_h = 512
size = image_w * 2 +1
gauss_kernel = create_gauss_kernel(size, size/10)
def OnCall(x):
    global size
    pt1_x = cv.getTrackbarPos("pt1_x", "press")
    pt1_y = cv.getTrackbarPos("pt1_y", "press")
    pt2_x = cv.getTrackbarPos("pt2_x", "press")
    pt2_y = cv.getTrackbarPos("pt2_y", "press")
    press_v1 = cv.getTrackbarPos("press_value1", "press")
    press_v2 = cv.getTrackbarPos("press_value2", "press")

    # 除以中心点的值 乘以压力值
    gauss_kernel1 = gauss_kernel/gauss_kernel[size//2, size//2] * press_v1
    gauss_kernel1 = np.uint8(gauss_kernel1)
    gauss_kernel2 = gauss_kernel/gauss_kernel[size//2, size//2] * press_v2
    gauss_kernel2 = np.uint8(gauss_kernel2)

    # 创建图像
    img = np.zeros((image_w, image_h, 3), np.uint8)
    img[:, :, 0]= 0 #H
    img[:, :, 1]= 255 #S 0为白色图像
    img[:, :, 2]= 255 #V 

    #高斯图形
    img_gauss = np.zeros((size, size, 3), np.uint8)
    img_gauss[:, :, 0]= gauss_kernel1 #H
    img_gauss[:, :, 1]= 255 #S 0为白色图像
    img_gauss[:, :, 2]= 255 #V
    img_gauss = cv.cvtColor(img_gauss, cv.COLOR_HSV2BGR)
    cv.imshow('gauss', img_gauss)

    #一个压力点
    ilefttop_x = size//2 - pt1_x
    ilefttop_y = size//2 - pt1_y
    for i in range(image_w):
        for j in range(image_h):
            ivalue = gauss_kernel1[i+ilefttop_x, j+ilefttop_y]
            img[i, j, 0] = img[i, j, 0] + ivalue
            if img[i, j, 0] > 120:
                img[i, j, 0] = 120
    
    #一个压力点
    ilefttop_x = size//2 - pt2_x
    ilefttop_y = size//2 - pt2_y
    for i in range(image_w):
        for j in range(image_h):
            img[i, j, 0] = img[i, j, 0] + gauss_kernel2[i+ilefttop_x, j+ilefttop_y]
            if img[i, j, 0] > 120:
                img[i, j, 0] = 120
   
    img[:,:,0] = 120 - img[:,:,0]
    img = cv.cvtColor(img, cv.COLOR_HSV2BGR)
    cv.imshow('press', img)

cv.namedWindow("press")
cv.createTrackbar("pt1_x", "press", 0, image_w, OnCall)
cv.createTrackbar("pt1_y", "press", 0, image_w, OnCall)
cv.createTrackbar("pt2_x", "press", 0, image_w, OnCall)
cv.createTrackbar("pt2_y", "press", 0, image_w, OnCall)
cv.createTrackbar("press_value1", "press", 0, 120, OnCall)
cv.createTrackbar("press_value2", "press", 0, 120, OnCall)


cv.waitKey(0)
cv.destroyAllWindows()