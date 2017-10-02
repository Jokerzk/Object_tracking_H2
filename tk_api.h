#ifndef __TK_API_H__
#define __TK_API_H__

#include <iostream>
#include <stdio.h>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <string>
#include <opencv2/opencv.hpp>
#include <sys/time.h>
#include "arm_neon.h"

#define h_psr_th 8.0
#define l_psr_th 6.0


//#define h_psr_th 8.5
//#define l_psr_th 7.0
#define PROFLING
#define RGB
#define FHOG
#define SCALE_NUM 3
#define PADDING

typedef struct
{
	float padding;
	float output_sigma_factor;
	float scale_sigma_factor;
	float lambda;
	float learning_rate;
	int number_of_scales;
	float scale_step;
	int scale_model_max_area;
}tk_params;

typedef struct
{
	float score;
	cv::Point2f coor;
	cv::Rect dect_rect;
}detect_candidate;

typedef struct
{
	cv::Size trans_model_sz;
	cv::Size scale_model_sz;
	float curr_scale;
	float scale_fator;
	float* scale_factors;
	cv::Mat hann_trans_2D;
	cv::Mat hann_scale_1D;
	cv::Mat haan_color_2D;

	int sizeX;
	int sizeY;

	int ilost_cnt;

	int img_width;
	int img_height;

	cv::Mat input_img;

	cv::Point2f tk_ptf;
	cv::Point2f tk_color_ptf;

	cv::Mat gauss_trans_2D;
	cv::Mat gauss_scale_2D;

	cv::Mat trans_roi_mat;
	std::vector<cv::Mat> vec_feature_2D;
	std::vector<cv::Mat> vec_hf_num;
	std::vector<cv::Mat> vec_detect_2D;
	cv::Mat hf_den;

	cv::Mat scale_feature_2D;
	cv::Mat sf_num;
	cv::Mat sf_den;
	cv::Mat gray;
	int scale_feature_length;
	float tk_psr;
	float dt_psr;

	cv::MatND color_model;
	cv::MatND color_foreground;
	cv::MatND color_background;
}tk_data;

typedef struct
{
	float x;  
	float y;  
	float width; 
	float height; 

}AutelRect;

typedef struct
{
	int x;
	int y;
}AutelPoint2i;

typedef struct
{
	int width;
	int height;
}AutelSize;

typedef struct
{
	int width;
	int height;
	unsigned char *buffer;
}AutelMat;

int YuvToRgb(uchar *rgbBuf, uchar *yBuf, uchar *uvBuf, int width, int height, int *rgbLen);

cv::Mat fftd(cv::Mat img, bool backwards, bool byRow);

void init_api(cv::Mat img, tk_data *ptk_data, cv::Size init_sz, tk_params params);

void update_color_model(tk_data *ptk_data, float learning_rate);

cv::Mat calc_color_response(cv::Mat roi_mat, tk_data *ptk_data);

void sort_by_color_score(cv::Mat img, tk_data *ptk_data, std::vector<detect_candidate> &vec_candidate);

void get_scale_response(tk_data *ptk_data, tk_params params, cv::Size &tk_sz);

float calc_target_score(cv::Mat response, cv::Point2f &tk_pt);

float get_trans_response(tk_data *ptk_data, tk_params params, cv::Point2i &tk_pt);

float get_detect_response(tk_data *ptk_data, tk_params params, cv::Point2i &tk_pt);

cv::Mat  get_space_model(tk_data *ptk_data);

void get_space_model_new(tk_data *ptk_data, cv::Mat &space_model);

float calc_psr(cv::Mat response, cv::Point2f tk_pt);

void rearrange(cv::Mat &img);

void RGB2Lab(cv::Mat rgbMat, cv::Mat &labMat);

void get_trans_sample(cv::Mat img, cv::Point2f tk_pt, cv::Size init_sz, tk_data *ptk_data, tk_params &params);

void get_detect_sample(cv::Mat img, cv::Point2f tk_pt, cv::Size detect_sz, tk_data *ptk_data, tk_params &params);

void get_scale_sample(cv::Mat img, cv::Point2f tk_pt, cv::Size init_sz, tk_data *ptk_data, tk_params &params);

void update_trans_model(tk_data *ptk_data, float learning_rate);

void update_scale_model(tk_data *ptk_data, float learning_rate);

void get_local_peak_candidate(cv::Mat response, std::vector<detect_candidate> &vec_candidate, int winsize);

int getcurtime(void);

#endif