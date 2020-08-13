import cv2 as cv
import numpy as np
import math
from matplotlib import pyplot as plt

#本部分内容：
## 直方图histogram
## 直方图查找
## 直方图绘制
## 直方图均衡 equalizeHist
## 二维直方图
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

cv.waitKey(0)
cv.destroyAllWindows()