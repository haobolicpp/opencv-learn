import cv2 as cv
import numpy as np
import math
from matplotlib import pyplot as plt

# 内容如下：
# - 使用模板匹配在图像中查找对象 
# - cv.matchTemplate()，cv.minMaxLoc()
# - 但对象匹配，多对象匹配
# 参考链接https://blog.csdn.net/holybin/article/details/40541933

# matchTemplate() 只是将模板图​​像滑动到输入图像上（就像在2D卷积中一样），然后在模板图像下比较模板和输入图像的拼图，
# 因此旋转/缩放的物体是匹配不出来的。
# 【匹配的方法】：
# 1、cv::TM_SQDIFF：该方法使用平方差进行匹配，因此最佳的匹配结果在结果为0处，值越大匹配结果越差。
# 2、cv::TM_SQDIFF_NORMED：该方法使用归一化的平方差进行匹配，最佳匹配也在结果为0处。
# 3、cv::TM_CCORR：相关性匹配方法，该方法使用源图像与模板图像的卷积结果进行匹配，因此，最佳匹配位置在值最大处，值越小匹配结果越差。
# 4、cv::TM_CCORR_NORMED：归一化的相关性匹配方法，与相关性匹配方法类似，最佳匹配位置也是在值最大处。
# 5、cv::TM_CCOEFF：相关性系数匹配方法，该方法使用源图像与其均值的差、模板与其均值的差二者之间的相关性进行匹配，最佳匹配结果在值等于1处，最差匹配结果在值等于-1处，值等于0直接表示二者不相关。
# 6、cv::TM_CCOEFF_NORMED：归一化的相关性系数匹配方法，正值表示匹配的结果较好，负值则表示匹配的效果较差，也是值越大，匹配效果也好。
# 【匹配结果result】模板在待测图像上每次在横向或是纵向上移动一个像素，并作一次比较计算，由此，横向比较W-w+1次，纵向比较H-h+1次，从而得到一个（W-w+1）×（H-h+1）维的结果矩阵，
# result即是用图像来表示这样的矩阵,因而图像result的大小为（W-w+1）×（H-h+1）。匹配结果图像与原图像之间的大小关系，他们之间差了一个模板大小。
# 【如何从result中获得最佳匹配区域】使用函数cvMinMaxLoc(result,&min_val,&max_val,&min_loc,&max_loc,NULL);从result中提取最大值（最小值）以及最大值的位置（即模板滑行时左上角的坐标）


# plt.subplot(144),plt.imshow(img_back, cmap = 'gray'),plt.title('img_back_low')
# plt.show()

cv.waitKey(0)
cv.destroyAllWindows()