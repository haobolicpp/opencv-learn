import cv2 as cv
import numpy as np
import math
from matplotlib import pyplot as plt

#本部分内容：
## 直方图histogram
## 直方图查找
## 直方图绘制
## 直方图均衡 equalizeHist
## 二维直方图 从HSV空间中提取H和S对应的像素个数
## 直方图反投影
## cv.calcHist()，np.histogram()

# 【直方图】它是在X轴上具有像素值（不总是从0到255的范围），在Y轴上具有图像中相应像素个数的图。 
# 【术语BINS】opencv中用histSize表示，针对x轴，当bins为256则表示每个像素统计一个x，bins为16时表示16个像素的个数总和表示一个刻度
# 【术语DIMS】收集数据的参数的数量，这里是1？？？TODO
# 【术语RANGE】要测量的强度的范围，通常是[0,256]
# cv.calcHist（images，channels，mask，histSize，ranges [，hist [，accumulate]]）
# 参数一images：它是uint8或float32类型的源图像。它应该放在方括号中，即“ [img]”。
# 参数二channels：也以方括号给出。它是我们计算直方图的通道的索引。例如，如果输入为灰度图像，则其值为[0]。对于彩色图像，您可以传递[0]，[1]或[2]分别计算蓝色，绿色或红色通道的直方图。
# 参数三ask：图像掩码。为了找到完整图像的直方图，将其指定为“无”。但是，如果要查找图像特定区域的直方图，则必须为此创建一个掩码图像并将其作为掩码。
# histSize：这表示我们的BIN计数。需要放在方括号中。对于全尺寸，设置[256]。
# ranges：这是我们的RANGE。通常为[0,256]。
# 返回值：256*1的数组
img = cv.imread('./pic/1.jpg', 0)
hist = cv.calcHist([img], [0], None, [256], [0, 256])
print(hist)

# 【绘制直方图】--使用matplotlib--它直接找到直方图并将其绘制,无需调用calcHist
plt.hist(img.ravel(),256,[0,256]); plt.show()

# 【绘制直方图--使用opencv】 使用cv.line或pylonline  略

# 【掩码的应用】，可创建掩码图像，掩码中白色为要统计直方图的部分，黑色为忽略的部分。 略

# 【直方图均衡】--参考https://blog.csdn.net/qq_15971883/article/details/88699218
# 大概思路：
# ①求直方图，得到各个像素个数的概率（像素n的个数/总个数）
# ②依靠s = T(r)得到最终的像素，这里r为原始像素值，s为均衡化变换后的像素值
# T的公式参考链接，这里不推导了
# 【适用场景】如果一幅图像整体偏暗或者偏亮，那么直方图均衡化的方法很适用。
# 【缺点】：如果图像某些区域对比度很好，而另一些区域对比度不好，那采用直方图均衡化就不一定适用；
# 均衡化后图像的灰度级减少，某些细节将会消失；某些图像（如直方图有高峰），经过均衡化后对比度不自然的过分增强
img = cv.imread('./pic/his1.png', 0)
img_equalizeHist = cv.equalizeHist(img)
plt.subplot(121),plt.imshow(img,cmap='gray'),plt.title('img')
plt.subplot(122),plt.imshow(img_equalizeHist,cmap='gray'),plt.title('img_equalizeHist')
plt.show()

# 【对比度受限的自适应直方图均衡-(CLAHE算法)】--为了解决上面的缺点，这里用了一个图像，某些区域直方图过高导致
# 大概原理：图像被分成称为“tiles”的小块（在OpenCV中，tileSize默认为8x8）。然后，像往常一样对这些块中的每一个进行直方图均衡。
img = cv.imread('./pic/his2.png', 0)
img_equalizeHist = cv.equalizeHist(img) #普通的全局均衡，效果不好
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
img_clahe = clahe.apply(img)
plt.subplot(131),plt.imshow(img,cmap='gray'),plt.title('img')
plt.subplot(132),plt.imshow(img_equalizeHist,cmap='gray'),plt.title('img_equalizeHist')
plt.subplot(133),plt.imshow(img_clahe,cmap='gray'),plt.title('img_clahe')
plt.show()

# 【二维直方图-颜色直方图】考虑两个特征：每个像素的色相和饱和度值（即HSV空间中的H和S，取值范围：H [0, 179] S [0, 255] V [0, 255]）
# 一维直方图是将图像从RGB转为灰度
# 二维直方图试将图像从RGB转为HSV
img = cv.imread('./pic/1.jpg')
img = cv.cvtColor(img,cv.COLOR_BGR2HSV)
hist = cv.calcHist([img], [0,1], None, [180, 256], [0, 180, 0, 256]) #shape[180, 256],x轴代表S饱和度，y轴代表H色相
cv.imshow('hist', hist) #是一副灰度图，代表不同饱和度和色相对应的像素个数，不是很直观

# 【直方图反向投影】--例子讲的很明白https://blog.csdn.net/keith_bb/article/details/70154219
# 反向投影：简单来说，如果当前图像一个像素值为10，那么就用直方图中该像素10对应的像素个数来代替当前的像素值！
# calcBackProject（const Mat *     images,输入图像，图像深度必须位CV_8U,CV_16U或CV_32F中的一种，尺寸相同，每一幅图像都可以有任意的通道数 
# int     nimages, 输入图像数量
# const int *     channels,用于计算反向投影的通道列表，通道数必须与直方图维度相匹配，第一个数组的通道是从0到image[0].channels()-1,第二个数组通道从图像image[0].channels()到image[0].channels()+image[1].channels()-1计数 
# InputArray      hist,输入的直方图，直方图的bin可以是密集(dense)或稀疏(sparse) 
# OutputArray     backProject,目标反向投影输出图像，是一个单通道图像，与原图像有相同的尺寸和深度 
# const float **      ranges,直方图中每个维度bin的取值范围 
# double      scale = 1,可选输出反向投影的比例因子 
# bool    uniform = true ）直方图是否均匀分布(uniform)的标识符，有默认值true
# 【用途】比方说提取背景、目标追踪等等
# 1、用一个ROI来包括目标的一部分，计算该ROI的直方图（二维的最好）
# 2、将直方图进行归一化处理，归一化到[0,255]范围内（因为像素个数可能会超过255）
# 3、将该直方图在目标图像上反投影，得到投影图像
#图像采用capture.png，我们将分别提取墙壁、手和瓶盖进行测试
#结果墙壁效果不好，因为三面墙颜色都不太一样
#手也不太好，颜色差别明显，可能把手整个阔起来好点
#瓶盖效果较好
img_src = cv.imread('./pic/capture.png')
img_src_hsv = cv.cvtColor(img_src, cv.COLOR_BGR2HSV)
img_roi_wall = img_src[22:36, 12:23] # 
img_roi_hand = img_src[75:96, 84:104]
img_roi_battle = img_src[25:42, 54:70]
plt.subplot(431),plt.imshow(cv.cvtColor(img_roi_wall, cv.COLOR_BGR2RGB)),plt.title('img_roi_wall') # 显示有色差，是因为plt.imshow是按RGB显示，cv.imread是bgr格式 参考https://blog.csdn.net/u013608424/article/details/80117178
plt.subplot(432),plt.imshow(cv.cvtColor(img_roi_hand, cv.COLOR_BGR2RGB)),plt.title('img_roi_hand')
plt.subplot(433),plt.imshow(cv.cvtColor(img_roi_battle, cv.COLOR_BGR2RGB)),plt.title('img_roi_battle')
img_roi_wall_hsv = cv.cvtColor(img_roi_wall, cv.COLOR_BGR2HSV)
img_roi_hand_hsv = cv.cvtColor(img_roi_hand, cv.COLOR_BGR2HSV)
img_roi_battle_hsv = cv.cvtColor(img_roi_battle, cv.COLOR_BGR2HSV)
#————提取墙壁————
his_wall = cv.calcHist([img_roi_wall_hsv], [0,1], None, [180, 256], [0, 180, 0, 256]) #二维直方图
plt.subplot(434),plt.imshow(his_wall,cmap='gray'),plt.title('his_wall')
cv.normalize(his_wall,his_wall,0,255,cv.NORM_MINMAX)
dist_wall = cv.calcBackProject([img_src_hsv], [0,1], his_wall, [0, 180, 0, 256], 1) #反向投影
#————提取手————
his_hand = cv.calcHist([img_roi_hand_hsv], [0,1], None, [180, 256], [0, 180, 0, 256]) #二维直方图
plt.subplot(435),plt.imshow(his_hand,cmap='gray'),plt.title('his_hand')
cv.normalize(his_hand,his_hand,0,255,cv.NORM_MINMAX)
dist_hand = cv.calcBackProject([img_src_hsv], [0,1], his_hand, [0, 180, 0, 256], 1) #反向投影
#————提取瓶盖————
his_battle = cv.calcHist([img_roi_battle_hsv], [0,1], None, [180, 256], [0, 180, 0, 256]) #二维直方图
plt.subplot(436),plt.imshow(his_battle,cmap='gray'),plt.title('his_battle')
cv.normalize(his_battle,his_battle,0,255,cv.NORM_MINMAX)
dist_battle = cv.calcBackProject([img_src_hsv], [0,1], his_battle, [0, 180, 0, 256], 1) #反向投影
plt.subplot(437),plt.imshow(dist_wall,cmap='gray'),plt.title('dist_wall')
plt.subplot(438),plt.imshow(dist_hand,cmap='gray'),plt.title('dist_hand')
plt.subplot(439),plt.imshow(dist_battle,cmap='gray'),plt.title('dist_battle')
#---最终二值化--
ret, img_binary = cv.threshold(dist_wall, 0, 255, cv.THRESH_BINARY)
plt.subplot(4,3,10),plt.imshow(img_binary,cmap='gray'),plt.title('dist_wall_binary')
ret, img_binary = cv.threshold(dist_hand, 0, 255, cv.THRESH_BINARY)
plt.subplot(4,3,11),plt.imshow(img_binary,cmap='gray'),plt.title('dist_hand_binary')
ret, img_binary = cv.threshold(dist_battle, 0, 255, cv.THRESH_BINARY)
plt.subplot(4,3,12),plt.imshow(img_binary,cmap='gray'),plt.title('dist_battle_binary')

plt.show()
cv.waitKey(0)
cv.destroyAllWindows()