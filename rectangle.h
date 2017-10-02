#include "BoundaryDissimilarityMap.h"
using namespace std;
using namespace cv;

int getThreshold(cv::Mat img,int width,int height);//get threshold by otsu's method

cv::Rect getrectangular(cv::Mat img);//get rectangular from binary map

cv::Mat CombinationandPostProcessing(cv::Mat salientimg,int width, int height);//get binary map from M's distance

cv::Rect getSalientMap(cv::Mat img,int width,int height);//get the final rectangular

int getoptimizerect(cv::Mat img,cv::Point2i &tk_pt,cv::Size &tk_sz);
