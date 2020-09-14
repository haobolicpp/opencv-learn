import cv2 as cv
import numpy as np
import math
from matplotlib import pyplot as plt
import time

#本部分内容：（还是没理解深）
# 快速傅里叶变换FFT
# opencv傅里叶变换及应用
# 傅里叶变换性能优化--针对图像尺寸 cv.getOptimalDFTSize() cv.copyMakeBorder
 
#图像x,y,灰度可以在三维空间中描述为一个曲面，【傅里叶变换】就是找到一堆sin型曲面，这个sin型曲面包含了幅值、相位、频率和方向，图像中opencv-dft后得到的即是各个中sin型曲面的信息（用复数描述，实部+i虚部）
#在FFT或DFT后的频谱图像中，点到中心的距离是频率，点的值是幅值，幅值需根据opencv返回的值单独计算，因为DFT返回的是复数。复数的幅值，实部和虚部平方和开方
#【结论】离中心点近的点频率越低，越亮表示幅值越大，可以这样考虑：表示变化比较低的点如背景，比较多导致中心位置一般幅值高所以很亮。
#参考https://zhuanlan.zhihu.com/p/99605178，讲的非常明白
#https://blog.csdn.net/yukinoai/article/details/88362051, 实践

# 【傅里叶变换】
img = cv.imread('./pic/3-sub.png',0)
dft = cv.dft(np.float32(img),flags = cv.DFT_COMPLEX_OUTPUT) #返回双通道，一个通道是实部，一个通道是虚部
dft_shift = np.fft.fftshift(dft) #中心化
magnitude_spectrum = 20*np.log(cv.magnitude(dft_shift[:,:,0],dft_shift[:,:,1])) #cv.magnitude计算复数的幅值，实部和虚部平方和开方。为了显示进行进行对数变换
plt.subplot(141),plt.imshow(img, cmap = 'gray'),plt.title('img')
plt.subplot(142),plt.imshow(magnitude_spectrum, cmap = 'gray'),plt.title('magnitude_spectrum')

#【频率图像滤波-低通滤波】--模糊图像
rows, cols = img.shape
crow,ccol = int(rows/2) , int(cols/2)
mask = np.zeros((rows,cols,2),np.uint8)# 首先创建一个掩码，
mask[(crow-30):(crow+30), (ccol-30):(ccol+30)] = 1#中心正方形区域为1，其余全为零
fshift = dft_shift*mask #相乘后只保留中心正方形的数值，即低频部分
f_ishift = np.fft.ifftshift(fshift) #去中心化
img_back = cv.idft(f_ishift) #反变换
img_back = cv.magnitude(img_back[:,:,0],img_back[:,:,1]) #计算幅值作为像素值
plt.subplot(143),plt.imshow(img_back, cmap = 'gray'),plt.title('img_back_low')

#【频率图像滤波-高通滤波】--删除低频信息，保留边缘信息，采用高通滤波器【拉普拉斯】
# 方形框会出现“振铃”效应，应该采用高斯框，这里没再实验了 TODO
mask = np.ones((rows,cols,2),np.uint8)# 首先创建一个掩码，
mask[(crow-30):(crow+30), (ccol-30):(ccol+30)] = 0#中心正方形区域为0，其余全为1
fshift = dft_shift*mask 
f_ishift = np.fft.ifftshift(fshift) #去中心化
img_back = cv.idft(f_ishift) #反变换
img_back = cv.magnitude(img_back[:,:,0],img_back[:,:,1]) #计算幅值作为像素值
plt.subplot(144),plt.imshow(img_back, cmap = 'gray'),plt.title('img_back_low')
plt.show()

#【频率图像滤波-高通滤波】-使用高斯窗口（略）

#【性能优化】对于某些数组尺寸，DFT的计算性能较好。当数组大小为2的幂时，速度最快
#如果您担心代码的性能，可以在找到DFT之前将数组的大小修改为任何最佳大小(通过填充零)。对于OpenCV，您必须手动填充零。
img = cv.imread('./pic/1.jpg',0)
(rows,cols) = img.shape
print(rows,cols)
nrows = cv.getOptimalDFTSize(rows)
ncols = cv.getOptimalDFTSize(cols)
print(nrows,ncols)
# 填充边界 void copyMakeBorder(InputArray src, OutputArray dst, int top, int bottom, int left, int right, int borderType, const Scalar& value=Scalar());
# top，bottom，left，right，分别表示在原图像的四个方向上扩充多少像素。
img2 = cv.copyMakeBorder(img,0,nrows-rows,0,ncols-cols,cv.BORDER_CONSTANT, value = 0) # 只扩充了下边界和右边界
cv.imshow('img2', img2)
time_start=time.time()
cv.dft(np.float32(img),flags = cv.DFT_COMPLEX_OUTPUT)
time_end=time.time()
print('img:',1000*(time_end-time_start), 'ms') #3ms
time_start=time.time()
cv.dft(np.float32(img2),flags = cv.DFT_COMPLEX_OUTPUT)
time_end=time.time()
print('img2:',1000*(time_end-time_start), 'ms') #<1ms 时间相差三倍

cv.waitKey(0)
cv.destroyAllWindows()