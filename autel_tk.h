#ifndef __AUTEL_TK_H__
#define __AUTEL_TK_H__

#include <iostream>
#include <stdio.h>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <string>
#include <opencv2/opencv.hpp>

#include "tk_api.h"

void tk_init(AutelMat img, AutelPoint2i tk_pt, AutelSize init_sz, tk_data *ptk_data, tk_params &params);

void tk_init(cv::Mat img, cv::Point2i tk_pt, cv::Size init_sz, tk_data *ptk_data, tk_params &params);

void tk_track(AutelMat img, AutelPoint2i &tk_pt, AutelSize &tk_sz, tk_data *ptk_data, tk_params &params);

void tk_track(cv::Mat img, cv::Point2i &tk_pt, cv::Size &tk_sz, tk_data *ptk_data, tk_params &params);

void tk_detect(cv::Mat img, cv::Point2i &tk_pt, cv::Size &tk_sz, tk_data *ptk_data, tk_params &params);

void tk_release(tk_data *ptk_data, tk_params &params);

#endif
