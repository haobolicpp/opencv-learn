#include <opencv.hpp>
using namespace cv;
using namespace std;
//利用SimpleBlobDetector，参考链接 https://www.cnblogs.com/ybqjymy/p/12826445.html
// thresholdStep = 10;   //二值化的阈值步长，即公式1的t
// minThreshold = 50;  //二值化的起始阈值，即公式1的T1
// maxThreshold = 220; //二值化的终止阈值，即公式1的T2
// minRepeatability = 2;//重复的最小次数，只有属于灰度图像斑点的那些二值图像斑点数量大于该值时，该灰度图像斑点才被认为是特征点
// minDistBetweenBlobs = 10;//最小的斑点距离，不同二值图像的斑点间距离小于该值时，被认为是同一个位置的斑点，否则是不同位置上的斑点
// filterByColor = true;        //斑点颜色的限制变量
// blobColor = 0;                  //表示只提取黑色斑点；如果该变量为255，表示只提取白色斑点
// filterByArea = true;         //斑点面积的限制变量
// minArea = 25;                   //斑点的最小面积
// maxArea = 5000;                 //斑点的最大面积
// filterByCircularity = false; //斑点圆度的限制变量，默认是不限制
// minCircularity = 0.8f;          //斑点的最小圆度
// maxCircularity = std::numeric_limits<float>::max();//斑点的最大圆度，所能表示的float类型的最大值
//                             //斑点惯性率的限制变量
// minInertiaRatio = 0.1f;                          //斑点的最小惯性率
// maxInertiaRatio = std::numeric_limits<float>::max(); //斑点的最大惯性率
// filterByConvexity = true;                         //斑点凸度的限制变量
// minConvexity = 0.95f;                            //斑点的最小凸度
// maxConvexity = std::numeric_limits<float>::max();    //斑点的最大凸度

int main(){

    Mat output_img;
    cv::Mat img = cv::imread("./pic/wudian.png", IMREAD_GRAYSCALE);
    output_img = img.clone();
    namedWindow("img", WINDOW_NORMAL);
    cv::imshow("img", img);
    imwrite("./img.bmp", img);

    //拉普拉斯变换
    // cv::Laplacian(img, output_img, CV_64F, 3);
    // cv::convertScaleAbs(output_img, output_img);
    // imwrite("./Laplacian.bmp", output_img);
    // cv::subtract(img, output_img, output_img);
    // imwrite("./sub_Laplacian.bmp", output_img);

    // //膨胀后腐蚀（针对亮色部分）
    // Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3), Point(-1, -1));
    // cv::dilate(output_img, output_img, kernel);
    // cv::erode(output_img, output_img, kernel);
    // imwrite("./morph.bmp", output_img);

    //sobel变换

    //高斯滤波
    cv::GaussianBlur(img, output_img, cv::Size(5, 5), 3);
    imwrite("./GaussianBlur.bmp", output_img);

    //二值化
    cv::threshold(output_img, output_img, 88, 255, THRESH_BINARY);
    //cv::threshold(output_img, output_img, 0, 255, THRESH_OTSU);
    imwrite("./threshold.bmp", output_img);


    SimpleBlobDetector::Params params;
    params.blobColor = 255;
    params.minThreshold = 80;
    params.maxThreshold = 105;
    params.thresholdStep = 5;
    params.minArea = 10;
    params.maxArea = 50;
    params.filterByConvexity = false; //不检测凸度
    //params.minConvexity = .05f;
    params.filterByInertia = false; //不检测惯性率
    //params.minInertiaRatio = .05f; //斑点的最小惯性率，越接近1越圆
    params.filterByCircularity = false; //圆度不检测
    //params.minCircularity = 0.5f; //斑点的最小圆度，接近1越接近圆
    

    Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);
    vector<KeyPoint> key_points;
    detector->detect(output_img, key_points);
    printf("key_points:%d\n", (int)key_points.size());

    drawKeypoints(output_img, key_points, output_img, Scalar(0, 255, 0), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    imwrite("./SimpleBlobDetector.bmp", output_img);

    waitKey(0);

    return 0;
}