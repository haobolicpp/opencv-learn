#include <opencv.hpp>

int main(){

    cv::Mat img = cv::imread("./pic/wudian.png");
    cv::imshow("img", img);

    cv::waitKey(0);

    return 0;
}