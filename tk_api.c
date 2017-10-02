#include "tk_api.h"
#include "fhog.hpp"

#ifdef HAVE_TBB
#include <tbb/tbb.h>
#include "tbb/parallel_for.h"
#include "tbb/blocked_range.h"
#endif


//#define SCALE_MAG

int bin_map[256] = {  0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4,
					  5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9,
                      10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13,
					  13, 13, 14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 16, 16, 16, 17, 17, 17, 17, 
					  17, 17, 17, 17, 18, 18, 18, 18, 18, 18, 18, 18, 19, 19, 19, 19, 19, 19, 19, 19, 20, 20, 20, 20, 20, 20, 20, 20, 21, 21, 
					  21, 21, 21, 21, 21, 21, 22, 22, 22, 22, 22, 22, 22, 22, 23, 23, 23, 23, 23, 23, 23, 23, 24, 24, 24, 24, 24, 24, 24, 24, 
					  25, 25, 25, 25, 25, 25, 25, 25, 26, 26, 26, 26, 26, 26, 26, 26, 27, 27, 27, 27, 27, 27, 27, 27, 28, 28, 28, 28, 28, 28,
					  28, 28, 29, 29, 29, 29, 29, 29, 29, 29,30, 30, 30, 30, 30, 30, 30, 30, 31, 31, 31, 31, 31, 31, 31, 31};

cv::Mat fftd(cv::Mat img, bool backwards, bool byRow)
{
	if (img.channels() == 1)
	{
		cv::Mat planes[] = { cv::Mat_<float>(img), cv::Mat_<float>::zeros(img.size()) };
		cv::merge(planes, 2, img);
	}
	if (byRow)
		cv::dft(img, img, (cv::DFT_ROWS | cv::DFT_COMPLEX_OUTPUT));
	else
		cv::dft(img, img, backwards ? (cv::DFT_INVERSE | cv::DFT_SCALE) : 0);
	return img;
}

cv::Mat fft2(cv::Mat img, bool backwards, bool byRow)
{
	cv::Mat complex_mat;
	if(byRow)
	{
		cv::dft(img, complex_mat, (cv::DFT_ROWS | cv::DFT_COMPLEX_OUTPUT));
	}
	else
	{
		cv::dft(img, complex_mat, cv::DFT_COMPLEX_OUTPUT);
	}
	return complex_mat;
}

cv::Mat real(cv::Mat img)
{
	std::vector<cv::Mat> planes;
	cv::split(img, planes);
	return planes[0];
}

cv::Mat imag(cv::Mat img)
{
	std::vector<cv::Mat> planes;
	cv::split(img, planes);
	return planes[1];
}

cv::Mat magnitude(cv::Mat img)
{
	cv::Mat res;
	std::vector<cv::Mat> planes;
	cv::split(img, planes); // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
	if (planes.size() == 1) res = cv::abs(img);
	else if (planes.size() == 2) cv::magnitude(planes[0], planes[1], res); // planes[0] = magnitude
	else assert(0);
	return res;
}

cv::Mat complexMultiplication(cv::Mat a, cv::Mat b)
{
	std::vector<cv::Mat> pa;
	std::vector<cv::Mat> pb;
	cv::split(a, pa);
	cv::split(b, pb);

	std::vector<cv::Mat> pres;
	pres.push_back(pa[0].mul(pb[0]) - pa[1].mul(pb[1]));
	pres.push_back(pa[0].mul(pb[1]) + pa[1].mul(pb[0]));

	cv::Mat res;
	cv::merge(pres, res);

	return res;
}

cv::Mat complexMultiplication_ex(cv::Mat a, cv::Mat b)
{
	cv::Mat res = cv::Mat(a.rows, a.cols, a.type());

	float *pa = (float*)a.data;
	float *pb = (float*)b.data;
	float *pres = (float*)res.data;

	float pa1 = 0;
	float pa2 = 0;
	float pb1 = 0;
	float pb2 = 0;
	for (int i = 0; i < a.rows; ++i)
	{
		for (int j = 0; j < a.cols; j++)
		{
			pa1 = *pa++;
			pa2 = *pa++;
			pb1 = *pb++;
			pb2 = *pb++;
			*pres++ = pa1 * pb1 - pa2 * pb2;
			*pres++ = pa1 * pb2 + pa2 * pb1;
		}
	}
	return res;
}

cv::Mat complexDivisionReal(cv::Mat a, cv::Mat b)
{
	std::vector<cv::Mat> pa;
	cv::split(a, pa);

	std::vector<cv::Mat> pres;

	cv::Mat divisor = 1. / b;

	pres.push_back(pa[0].mul(divisor));
	pres.push_back(pa[1].mul(divisor));

	cv::Mat res;
	cv::merge(pres, res);
	return res;
}

#ifndef WIN_32
int getcurtime(void)
{
	struct timeval tv;
	struct timezone tz;
	int timetmp = 0;
	gettimeofday(&tv, &tz);
	timetmp = tv.tv_sec * 1000 + tv.tv_usec / 1000;

	return timetmp;
}
#else
int getcurtime(void)
{
	return cv::getTickCount()*1000 / cv::getTickFrequency();
}
#endif

cv::Mat complexDivision(cv::Mat a, cv::Mat b)
{
	std::vector<cv::Mat> pa;
	std::vector<cv::Mat> pb;
	cv::split(a, pa);
	cv::split(b, pb);

	cv::Mat divisor = 1. / (pb[0].mul(pb[0]) + pb[1].mul(pb[1]));

	std::vector<cv::Mat> pres;

	pres.push_back((pa[0].mul(pb[0]) + pa[1].mul(pb[1])).mul(divisor));
	pres.push_back((pa[1].mul(pb[0]) + pa[0].mul(pb[1])).mul(divisor));

	cv::Mat res;
	cv::merge(pres, res);
	return res;
}
void normalizedLogTransform(cv::Mat &img)
{
	img = cv::abs(img);
	img += cv::Scalar::all(1);
	cv::log(img, img);
	// cv::normalize(img, img, 0, 1, CV_MINMAX);
}

void rearrange(cv::Mat &img)
{
	int cx = img.cols / 2;
	int cy = img.rows / 2;

	cv::Mat q0(img, cv::Rect(0, 0, cx, cy)); // Top-Left - Create a ROI per quadrant
	cv::Mat q1(img, cv::Rect(cx, 0, cx, cy)); // Top-Right
	cv::Mat q2(img, cv::Rect(0, cy, cx, cy)); // Bottom-Left
	cv::Mat q3(img, cv::Rect(cx, cy, cx, cy)); // Bottom-Right

	cv::Mat tmp; // swap quadrants (Top-Left with Bottom-Right)
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);
	q1.copyTo(tmp); // swap quadrant (Top-Right with Bottom-Left)
	q2.copyTo(q1);
	tmp.copyTo(q2);
//	cv::rotate(img, img, 1);
}

bool sort_by_score(detect_candidate &d1, detect_candidate &d2)
{
	return d1.score > d2.score;
}

void init_api(cv::Mat img, tk_data *ptk_data, cv::Size init_sz, tk_params params)
{
	int padded_w = init_sz.width *(1 + params.padding);
	int padded_h = init_sz.height *(1 + params.padding);


	ptk_data->input_img = cv::Mat(img.rows, img.cols, CV_8UC3);

	ptk_data->img_width = img.cols;
	ptk_data->img_height = img.rows;

	int tmpl_size = cv::max(padded_h, padded_w);
	int trans_sz = 0;
	if (tmpl_size >64)
	{
		trans_sz = 64;
	}
	else
	{
		trans_sz = 64;
	}
	int cell_size = 4;
	ptk_data->curr_scale = (float)tmpl_size / trans_sz;
	ptk_data->scale_fator = 1.0;

	ptk_data->trans_model_sz.width = (int)(padded_w / ptk_data->curr_scale + 0.5);
	ptk_data->trans_model_sz.height = (int)(padded_h / ptk_data->curr_scale + 0.5);

	ptk_data->trans_model_sz.width = (((int)(ptk_data->trans_model_sz.width / (2 * cell_size))) * 2 * cell_size) + cell_size * 2;
	ptk_data->trans_model_sz.height = (((int)(ptk_data->trans_model_sz.height / (2 * cell_size))) * 2 * cell_size) + cell_size * 2;

	//printf("trans model size: %d, %d\n", ptk_data->trans_model_sz.width, ptk_data->trans_model_sz.height);

	if (init_sz.width * init_sz.height > params.scale_model_max_area)
	{
		float scale_ratio = std::sqrt(params.scale_model_max_area / (float)(init_sz.width * init_sz.height));
		ptk_data->scale_model_sz.width = cv::max((int)(init_sz.width *scale_ratio+0.5), 12);
		ptk_data->scale_model_sz.height = cv::max((int)(init_sz.height*scale_ratio+0.5), 12);
	}
	else
	{
		ptk_data->scale_model_sz = init_sz;
	}
	int n_scales = params.number_of_scales;
	float scale_step = params.scale_step;
	ptk_data->scale_factors = new float[n_scales];
	float ceilS = std::ceil(n_scales / 2.0f);
	for (int i = 0; i < n_scales; i++)
	{
		ptk_data->scale_factors[i] = std::pow(scale_step, ceilS - i - 1);
	}

#ifdef FHOG
	ptk_data->sizeX = ptk_data->trans_model_sz.width / cell_size -2;
	ptk_data->sizeY = ptk_data->trans_model_sz.height / cell_size -2;
#else
	ptk_data->sizeX = ptk_data->trans_model_sz.width;
	ptk_data->sizeY = ptk_data->trans_model_sz.height;
#endif

	cv::Mat hann1t = cv::Mat(cv::Size(ptk_data->sizeX, 1), CV_32FC1);
	cv::Mat hann2t = cv::Mat(cv::Size(1, ptk_data->sizeY), CV_32FC1);

	for (int i = 0; i < hann1t.cols; i++)
		hann1t.at<float >(0, i) = 0.5 * (1 - std::cos(2 * 3.14159265358979323846 * i / (hann1t.cols - 1)));
	for (int i = 0; i < hann2t.rows; i++)
		hann2t.at<float >(i, 0) = 0.5 * (1 - std::cos(2 * 3.14159265358979323846 * i / (hann2t.rows - 1)));

	ptk_data->hann_trans_2D = cv::Mat(cv::Size(ptk_data->sizeX, ptk_data->sizeY), CV_32FC1, cv::Scalar(0));
	for (int i = 0; i < ptk_data->sizeY; ++i)
	{
		for (int j = 0; j < ptk_data->sizeX; j++)
		{
			ptk_data->hann_trans_2D.at<float>(i, j) = hann2t.at<float>(i, 0)*hann1t.at<float>(0, j);
		}
	}
	cv::Mat hann1t_color = cv::Mat(cv::Size(ptk_data->trans_model_sz.width, 1), CV_32FC1);
	cv::Mat hann2t_color = cv::Mat(cv::Size(1, ptk_data->trans_model_sz.height), CV_32FC1);

	for (int i = 0; i < hann1t_color.cols; i++)
		hann1t_color.at<float >(0, i) = 0.5 * (1 - std::cos(2 * 3.14159265358979323846 * i / (hann1t_color.cols - 1))) ;//******//
	for (int i = 0; i < hann2t_color.rows; i++)
		hann2t_color.at<float >(i, 0) = 0.5 * (1 - std::cos(2 * 3.14159265358979323846 * i / (hann2t_color.rows - 1))) ;//******//

	ptk_data->haan_color_2D = cv::Mat(cv::Size(ptk_data->trans_model_sz.width, ptk_data->trans_model_sz.height), CV_32FC1, cv::Scalar(0));
	for (int i = 0; i < ptk_data->haan_color_2D.rows; ++i)
	{
		for (int j = 0; j < ptk_data->haan_color_2D.cols; j++)
		{
			ptk_data->haan_color_2D.at<float>(i, j) = hann2t_color.at<float>(i, 0)*hann1t_color.at<float>(0, j);
		}
	}

	ptk_data->hann_scale_1D = cv::Mat(cv::Size(n_scales, 1), CV_32FC1, cv::Scalar(0));
	for (int i = 0; i < ptk_data->hann_scale_1D.cols; i++)
		ptk_data->hann_scale_1D.at<float >(0, i) = 0.5 * (1 - std::cos(2 * 3.14159265358979323846 * i / (ptk_data->hann_scale_1D.cols - 1)));

	cv::Mat gauss_mask(ptk_data->sizeY, ptk_data->sizeX, CV_32FC1, cv::Scalar(0));
	int syh = ptk_data->sizeY / 2;
	int sxh = ptk_data->sizeX / 2;
	float output_sigma = std::sqrt((float)ptk_data->sizeY*ptk_data->sizeX) * params.output_sigma_factor / (1 + params.padding);
	float mult = -0.5 / (output_sigma * output_sigma);

	for (int i = 0; i < ptk_data->sizeY; i++)
	{
		for (int j = 0; j < ptk_data->sizeX; j++)
		{
			int ih = i - syh;
			int jh = j - sxh;
			gauss_mask.at<float>(i, j) = std::exp(mult * (float)(ih * ih + jh * jh));
		}
	}
	ptk_data->gauss_trans_2D = fft2(gauss_mask, false, false);

	int sizeX = ptk_data->scale_model_sz.width / cell_size - 2;
	int sizeY = ptk_data->scale_model_sz.height / cell_size - 2;
	ptk_data->scale_feature_length = sizeX*sizeY*31;

	float scale_sigma2 = n_scales / std::sqrt(n_scales) * params.scale_sigma_factor;
	scale_sigma2 = scale_sigma2 * scale_sigma2;
	cv::Mat gauss_scale(1, n_scales, CV_32FC1, cv::Scalar(0));
	ceilS = std::ceil(n_scales / 2.0f);
	for (int i = 0; i < n_scales; i++)
	{
		gauss_scale.at<float>(0, i) = std::exp(-0.5 * std::pow(i + 1 - ceilS, 2) / scale_sigma2);
	}
	ptk_data->gauss_scale_2D = fft2(gauss_scale, false, false);
	ptk_data->gauss_scale_2D = cv::repeat(ptk_data->gauss_scale_2D, ptk_data->scale_feature_length, 1);

	ptk_data->sf_num = cv::Mat(ptk_data->scale_feature_length, n_scales, CV_32FC2, cv::Scalar(0));
	ptk_data->sf_den = cv::Mat(1, n_scales, CV_32F, cv::Scalar(0));

	const int size[3] = { 32, 32, 32 };
	ptk_data->color_model = cv::Mat(3, size, CV_32F, float(0));
	ptk_data->color_foreground = cv::Mat(3, size, CV_32F, float(0));
	ptk_data->color_background = cv::Mat(3, size, CV_32F, float(0));
}

cv::Mat get_space_model(tk_data *ptk_data)
{
	cv::Mat new_hf_den;
	cv::mulSpectrums(ptk_data->vec_hf_num[0], ptk_data->vec_hf_num[0], new_hf_den, 0, true);
	cv::Mat temp = real(new_hf_den);
	cv::Mat space_mat = complexDivisionReal(ptk_data->vec_hf_num[0], temp + 1e-2);
	cv::dft(space_mat, space_mat, cv::DFT_INVERSE | cv::DFT_SCALE);
	space_mat = real(space_mat);
	cv::normalize(space_mat, space_mat, 1, 0, CV_MINMAX);
	rearrange(space_mat);
	return space_mat;
}

void get_space_model_new(tk_data *ptk_data, cv::Mat &space_model)
{
	cv::Mat new_hf_den;
	cv::mulSpectrums(ptk_data->vec_feature_2D[0], ptk_data->vec_feature_2D[0], new_hf_den, 0, true);
	//cv::Mat temp = real(new_hf_den);
	//cv::Mat space_mat = complexDivisionReal(ptk_data->vec_feature_2D[0], temp + 1e-2);
	cv::Mat space_mat;
	cv::dft(ptk_data->vec_feature_2D[0], space_mat, cv::DFT_INVERSE | cv::DFT_SCALE);
	space_mat = real(space_mat);
	cv::normalize(space_mat, space_mat, 1, 0, CV_MINMAX);
	//rearrange(space_mat);
	//cv::imshow("space_model", space_mat);
	//space_mat.copyTo(space_model);
}

static inline void* _my_memcpy(void *dest, void *src, size_t count)
{
	char *to = (char*)dest;
	char *from = (char*)src;
	size_t align_length = 0l;
	long head_align_offset = 0l, tail_align_offset = 0l;
	head_align_offset = (((unsigned long)from + 0x7) & ~0x7) - (unsigned long)from;
	switch (head_align_offset) 
	{
		case 7: 
			*to++ = *from++;
		case 6: *to++ = *from++;
		case 5: *to++ = *from++;
		case 4: *to++ = *from++;
		case 3: *to++ = *from++;
		case 2: *to++ = *from++;
		case 1: *to++ = *from++;
		default:
			break;
	}
	align_length = count - head_align_offset;
	tail_align_offset = align_length & 0x7;
	align_length &= (~0x7);
	memcpy(to, from, align_length);
	to += align_length;
	from += align_length;
	switch (tail_align_offset) {
	case 7: *to++ = *from++;
	case 6: *to++ = *from++;
	case 5: *to++ = *from++;
	case 4: *to++ = *from++;
	case 3: *to++ = *from++;
	case 2: *to++ = *from++;
	case 1: *to++ = *from++;
	default: break;
	}
	return dest;
}

cv::Mat getsubwindow(cv::Mat img, cv::Rect roi_rect)
{
	cv::Mat roi_mat = cv::Mat(roi_rect.height, roi_rect.width, img.type());
	int start_y = roi_rect.y;
	int end_y = roi_rect.y + roi_rect.height;
	int start_x = roi_rect.x;
	int end_x = roi_rect.x + roi_rect.width;
	int iy = 0;
	int ix = 0;
	int irow = 0;
	int icol = 0;
	if (img.channels() == 1)
	{
		for (int i = start_y; i < end_y; i++, irow++)
		{
			iy = i;
			if (iy < 0)
			{
				iy = 0;
			}
			if (iy >= img.rows)
			{
				iy = img.rows - 1;
			}
			uchar *ptr_src = img.data + iy * img.step1();
			uchar *ptr_dst = roi_mat.data + irow * roi_mat.step1();
			for (int j = start_x, icol = 0; j < end_x; j++, icol++)
			{
				ix = j;
				if (j < 0)
				{
					ix = 0;
				}
				if (j >= img.cols)
				{
					ix = img.cols - 1;
				}
				*ptr_dst++ = ptr_src[ix];
			}
		}
	}
	else
	{	
		int nchannel = 3;
		if (start_y >= 0 && end_y<img.rows && start_x >= 0 && end_x<img.cols)
		{
			irow = 0;
			for (int i = start_y; i < end_y; i++)
			{
				uchar *ptr_src = img.data + i* img.step1() + start_x * nchannel;
				uchar *ptr_dst = roi_mat.data + irow * roi_mat.step1();
				_my_memcpy(ptr_dst, ptr_src, sizeof(uchar)*roi_rect.width*nchannel);
				irow++;
			}
		}
		else
		{
			irow = 0;
			for (int i = start_y; i < end_y; i++)
			{
				iy = i;
				if (i<0)
				{
					iy = 0;
				}
				if (i >= img.rows)
				{
					iy = img.rows - 1;
				}
				uchar *ptr_src = img.data + iy * img.step1();
				uchar *ptr_dst = roi_mat.data + irow * roi_mat.step1();
				for (int j = start_x; j < end_x; j++)
				{
					ix = j;
					if (j<0)
					{
						ix = 0;
					}
					if (j >= img.cols)
					{
						ix = img.cols - 1;
					}
					uchar *ptemp = ptr_src + nchannel*ix;
					*ptr_dst++ = ptemp[0];
					*ptr_dst++ = ptemp[1];
					*ptr_dst++ = ptemp[2];
				}
				irow++;
			}
		}
	}
	return roi_mat;
}

void RGB2Lab(cv::Mat rgbMat, cv::Mat &labMat)
{
	int width = rgbMat.cols;
	int height = rgbMat.rows;
	uchar *rgbdata = rgbMat.data;
	float *labdata = (float*)labMat.data;
	for (int i = 0; i < height; ++i)
	{
		for (int j = 0; j < width; ++j)
		{
			
			double R = *rgbdata++ / 255.0;
			double G = *rgbdata++ / 255.0;
			double B = *rgbdata++ / 255.0;

			double r = 0;
			double g = 0;
			double b = 0;

			if (R <= 0.04045)	
				r = R / 12.92;
			else				
				r = pow((R + 0.055) / 1.055, 2.4);

			if (G <= 0.04045)	
				g = G / 12.92;
			else				
				g = pow((G + 0.055) / 1.055, 2.4);

			if (B <= 0.04045)	
				b = B / 12.92;
			else				
				b = pow((B + 0.055) / 1.055, 2.4);

			double X = r*0.4124564 + g*0.3575761 + b*0.1804375;
			double Y = r*0.2126729 + g*0.7151522 + b*0.0721750;
			double Z = r*0.0193339 + g*0.1191920 + b*0.9503041;
			//------------------------
			// XYZ to LAB conversion
			//------------------------
			double epsilon = 0.008856;	//actual CIE standard
			double kappa = 903.3;		//actual CIE standard

			double Xr = 0.950456;	//reference white
			double Yr = 1.0;		//reference white
			double Zr = 1.088754;	//reference white

			double xr = X / Xr;
			double yr = Y / Yr;
			double zr = Z / Zr;

			double fx = 0;
			double fy = 0; 
			double fz = 0;

			if (xr > epsilon)	
				fx = pow(xr, 1.0 / 3.0);
			else				
				fx = (kappa*xr + 16.0) / 116.0;
			if (yr > epsilon)	
				fy = pow(yr, 1.0 / 3.0);
			else				
				fy = (kappa*yr + 16.0) / 116.0;
			if (zr > epsilon)	
				fz = pow(zr, 1.0 / 3.0);
			else				
				fz = (kappa*zr + 16.0) / 116.0;

			*labdata++ = 116.0*fy - 16.0;
			*labdata++ = 500.0*(fx - fy);
			*labdata++ = 200.0*(fy - fz);
		}
	}
}
//#define LAB
void get_trans_sample(cv::Mat img, cv::Point2f tk_pt, cv::Size init_sz, tk_data *ptk_data, tk_params &params)
{
	ptk_data->vec_feature_2D.clear();
	float patch_width = ptk_data->trans_model_sz.width * ptk_data->curr_scale*ptk_data->scale_fator;
	float patch_height = ptk_data->trans_model_sz.height * ptk_data->curr_scale*ptk_data->scale_fator;

	cv::Rect trans_roi;
	trans_roi.x = ptk_data->tk_ptf.x - patch_width / 2.0 + 0.5;
	trans_roi.y = ptk_data->tk_ptf.y - patch_height / 2.0 + 0.5;
	trans_roi.width = patch_width + 0.5;
	trans_roi.height = patch_height + 0.5;

#ifdef PROFLING	
	int timestampstart = 0;
	int timestampstop = 0;
	timestampstart = getcurtime();
#endif

	cv::Mat temp_mat = getsubwindow(img, trans_roi);
	cv::resize(temp_mat, ptk_data->trans_roi_mat, ptk_data->trans_model_sz);

#ifdef PROFLING
    timestampstop = getcurtime();
	printf("roi and resize = %d\n", timestampstop - timestampstart);
#endif
	st_fhog_feture *trans_map;
	IplImage im_ipl = ptk_data->trans_roi_mat;
	getFeatureMaps(&im_ipl, 4, &trans_map);
	normalizeAndTruncate(trans_map, 0.2f);
	PCAFeatureMaps(trans_map);
	int feature_num = trans_map->numFeatures;
	int sx = trans_map->sizeX;
	int sy = trans_map->sizeY;
	cv::Mat FeaturesMap = cv::Mat(cv::Size(feature_num, sx*sy), CV_32F, trans_map->map);
	FeaturesMap = FeaturesMap.t();

#ifdef PROFLING
    timestampstart = getcurtime();
	printf("get fhog feature = %d\n", timestampstart - timestampstop);
#endif

	cv::Mat small_mat;
	cv::resize(ptk_data->trans_roi_mat, small_mat, cv::Size(sx, sy));
#ifdef LAB
	cv::Mat labMat = cv::Mat(sy, sx, CV_32FC3, float(0));
	RGB2Lab(small_mat, labMat);
	cv::split(labMat, ptk_data->vec_feature_2D);
#else
	cv::split(small_mat, ptk_data->vec_feature_2D);
#endif
	for (size_t i = 0; i < ptk_data->vec_feature_2D.size(); ++i)
	{
#ifndef LAB
		ptk_data->vec_feature_2D[i].convertTo(ptk_data->vec_feature_2D[i], CV_32FC1);
#endif
		cv::normalize(ptk_data->vec_feature_2D[i], ptk_data->vec_feature_2D[i], -0.5, 0.5, CV_MINMAX);
	}

	for (size_t i = 0; i < feature_num; i++)
	{
		cv::Mat temp = FeaturesMap.row(i);
		temp = temp.reshape(1, sy);
		ptk_data->vec_feature_2D.push_back(temp);
	}

	for (size_t i = 0; i < ptk_data->vec_feature_2D.size(); ++i)
	{
		ptk_data->vec_feature_2D[i] = ptk_data->vec_feature_2D[i].mul(ptk_data->hann_trans_2D);
		ptk_data->vec_feature_2D[i] = fft2(ptk_data->vec_feature_2D[i], 0, 0);
	}

#ifdef PROFLING
    timestampstop = getcurtime();
	printf("prepare and fft = %d\n", timestampstop - timestampstart);
#endif

	freeFeatureMapObject(&trans_map);
}

void get_detect_sample(cv::Mat img, cv::Point2f tk_pt, cv::Size detect_sz, tk_data *ptk_data, tk_params &params)
{
	ptk_data->vec_feature_2D.clear();
	float patch_width = ptk_data->trans_model_sz.width * ptk_data->curr_scale*ptk_data->scale_fator;
	float patch_height = ptk_data->trans_model_sz.height * ptk_data->curr_scale*ptk_data->scale_fator;

	cv::Rect trans_roi;
	trans_roi.x = tk_pt.x - patch_width / 2.0 + 0.5;
	trans_roi.y = tk_pt.y - patch_height / 2.0 + 0.5;
	trans_roi.width = patch_width + 0.5;
	trans_roi.height = patch_height + 0.5;
	cv::Mat temp_mat = getsubwindow(img, trans_roi);
	cv::resize(temp_mat, ptk_data->trans_roi_mat, ptk_data->trans_model_sz);

	st_fhog_feture *trans_map;
	IplImage im_ipl = ptk_data->trans_roi_mat;
	getFeatureMaps(&im_ipl, 4, &trans_map);
	normalizeAndTruncate(trans_map, 0.2f);
	PCAFeatureMaps(trans_map);
	int feature_num = trans_map->numFeatures;
	int sx = trans_map->sizeX;
	int sy = trans_map->sizeY;
	cv::Mat FeaturesMap = cv::Mat(cv::Size(feature_num, sx*sy), CV_32F, trans_map->map);
	FeaturesMap = FeaturesMap.t();

	cv::Mat small_mat;
	cv::resize(ptk_data->trans_roi_mat, small_mat, cv::Size(sx, sy));
#ifdef LAB
	cv::Mat labMat = cv::Mat(sy, sx, CV_32FC3, float(0));
	RGB2Lab(small_mat, labMat);
	cv::split(labMat, ptk_data->vec_feature_2D);
#else
	cv::split(small_mat, ptk_data->vec_feature_2D);
#endif
	for (size_t i = 0; i < ptk_data->vec_feature_2D.size(); ++i)
	{
#ifndef LAB
		ptk_data->vec_feature_2D[i].convertTo(ptk_data->vec_feature_2D[i], CV_32FC1);
#endif
		cv::normalize(ptk_data->vec_feature_2D[i], ptk_data->vec_feature_2D[i], -0.5, 0.5, CV_MINMAX);
	}
	for (size_t i = 0; i < feature_num; i++)
	{
		cv::Mat temp = FeaturesMap.row(i);
		temp = temp.reshape(1, sy);
		ptk_data->vec_feature_2D.push_back(temp);
	}

	for (size_t i = 0; i < ptk_data->vec_feature_2D.size(); ++i)
	{
		ptk_data->vec_feature_2D[i] = ptk_data->vec_feature_2D[i].mul(ptk_data->hann_trans_2D);
		ptk_data->vec_feature_2D[i] = fft2(ptk_data->vec_feature_2D[i], 0, 0);
	}

	freeFeatureMapObject(&trans_map);
}

void get_scale_sample(cv::Mat img, cv::Point2f tk_pt, cv::Size tk_sz, tk_data *ptk_data, tk_params &params)
{
	st_fhog_feture *map[33];
	int cell_size = 4;
	int n_scales = params.number_of_scales;
	cv::Mat scale_feature = cv::Mat(ptk_data->scale_feature_length, n_scales, CV_32F, cv::Scalar(0));

#ifdef PROFLING
	int timestampstart = 0;
    int timestampstop = 0;
	timestampstart = getcurtime();
#endif

#ifdef PROFLING
	int timestamproi = 0;
    int timestampresize = 0;
    int timestampfhog = 0;
#endif


	for (int i = SCALE_NUM; i < n_scales-SCALE_NUM; i++)
	{
		float patch_width = tk_sz.width * ptk_data->scale_factors[i] * ptk_data->scale_fator;
		float patch_height = tk_sz.height * ptk_data->scale_factors[i] * ptk_data->scale_fator;

	#ifdef PROFLING 
		int timeteststart = getcurtime();
	#endif

		cv::Rect roi_rect;
	#if 1	
		roi_rect.x = cv::max((int)(ptk_data->tk_ptf.x - patch_width / 2.0 + 0.5), 0);
		roi_rect.y = cv::max((int)(ptk_data->tk_ptf.y - patch_height / 2.0 + 0.5), 0);
		roi_rect.width = cv::min((int)patch_width, img.cols-roi_rect.x);
		roi_rect.height = cv::min((int)patch_height, img.rows-roi_rect.y);

		roi_rect.x = cv::min(roi_rect.x, img.cols-2);
		roi_rect.y = cv::min(roi_rect.y, img.rows-2);

		roi_rect.width = roi_rect.width <= 0 ? 1 : roi_rect.width;
		roi_rect.height = roi_rect.height <= 0 ? 1 : roi_rect.height;
		cv::Mat im_patch = img(roi_rect);
	#else
		roi_rect.x = ptk_data->tk_ptf.x - patch_width / 2.0 + 0.5;
		roi_rect.y = ptk_data->tk_ptf.y - patch_height / 2.0 + 0.5;
		roi_rect.width = patch_width + 0.5;
		roi_rect.height = patch_height + 0.5;
		cv::Mat im_patch = getsubwindow(img, roi_rect);
	#endif

	#ifdef PROFLING 
		int timetestend = getcurtime();
		timestamproi += (timetestend - timeteststart);
	#endif

		cv::resize(im_patch, im_patch, ptk_data->scale_model_sz);
		float cos_ratio = ptk_data->hann_scale_1D.at<float>(0, i);

	#ifdef PROFLING 
		timeteststart = getcurtime();
		timestampresize += (timeteststart - timetestend);
	#endif

		IplImage im_ipl = im_patch;//????//
		getFeatureMaps_gray(&im_ipl, 4, &map[i]);
		normalizeAndTruncate(map[i], 0.2f);
		PCAFeatureMaps(map[i]);
		cv::Mat FeaturesMap = cv::Mat(ptk_data->scale_feature_length, 1, CV_32F, map[i]->map);
		FeaturesMap = cos_ratio * FeaturesMap;
		FeaturesMap.copyTo(scale_feature.col(i));

	#ifdef PROFLING 
		timetestend = getcurtime();
		timestampfhog += (timetestend - timeteststart);
	#endif

	}
#ifdef PROFLING
    timestampstop = getcurtime();
	printf("scaleresize time = %d\n", timestampstop - timestampstart);
	printf("getroi time = %d\n", timestamproi);
	printf("resize time = %d\n", timestampresize);
	printf("getfog time = %d\n", timestampfhog);

#endif
	ptk_data->scale_feature_2D = fftd(scale_feature, 0, 1);
	for (int i = SCALE_NUM; i < n_scales-SCALE_NUM; i++)
	{
		freeFeatureMapObject(&map[i]);
	}
}

void get_scale_response(tk_data *ptk_data, tk_params params, cv::Size &tk_sz)
{
	cv::Mat add_temp;
	//cv::reduce(complexMultiplication(ptk_data->sf_num, ptk_data->scale_feature_2D), add_temp, 0, CV_REDUCE_SUM);
	cv::mulSpectrums(ptk_data->sf_num, ptk_data->scale_feature_2D, add_temp, 0, false);
	cv::reduce(add_temp, add_temp, 0, CV_REDUCE_SUM);

	cv::Mat scale_response;
	cv::idft(complexDivisionReal(add_temp, (ptk_data->sf_den + params.lambda)), scale_response, cv::DFT_REAL_OUTPUT);

	cv::Point2i pi;
	double pv;
	cv::minMaxLoc(scale_response, NULL, &pv, NULL, &pi);
	ptk_data->scale_fator *= ptk_data->scale_factors[pi.x];

	if (ptk_data->scale_fator < 0.2)
		ptk_data->scale_fator = 0.2;
	else if (ptk_data->scale_fator > 5.0)
		ptk_data->scale_fator = 5.0;
}

float subPixelPeak(float left, float center, float right)
{
	float divisor = 2 * center - right - left;
	if (divisor == 0)
		return 0;
	return 0.5 * (right - left) / divisor;
}

float calc_target_score(cv::Mat response,cv::Point2f &tk_pt)
{
	double pv;
	cv::Point2i pi;
	cv::minMaxLoc(response, NULL, &pv, NULL, &pi);

	float x_coor = pi.x;
	float y_coor = pi.y;
	if (pi.x > 0 && pi.x < response.cols - 1)
	{
		x_coor += subPixelPeak(response.at<float>(pi.y, pi.x - 1), pv, response.at<float>(pi.y, pi.x + 1));
	}
	if (pi.y > 0 && pi.y < response.rows - 1)
	{
		y_coor += subPixelPeak(response.at<float>(pi.y - 1, pi.x), pv, response.at<float>(pi.y + 1, pi.x));
	}

	tk_pt.x = x_coor - response.cols / 2.0;
	tk_pt.y = y_coor - response.rows / 2.0;

	int width = response.cols;
	int height = response.rows;
	int peak_x = pi.x;
	int peak_y = pi.y;
	int peak_w = width / 10 > 1 ? width / 10 : 1;
	int peak_h = height / 10 > 1 ? height / 10 : 1;

	int th1 = peak_y - peak_h > 0 ? peak_y - peak_h : 0;
	int th2 = peak_y + peak_h < height - 1 ? peak_y + peak_h : height - 1;
	int th3 = peak_x - peak_w > 0 ? peak_x - peak_w : 0;
	int th4 = peak_x + peak_w < width - 1 ? peak_x + peak_w : width - 1;
	float sumofsidelobe = 0;
	int numofsidelobe = 0;

	for (int i = 0; i < height; ++i)
	{
		for (int j = 0; j < width; ++j)
		{
			if ((i >= th1) && (i <= th2) && (j >= th3) && (j <= th4))
			{
				continue;
			}
			sumofsidelobe += response.at<float>(i, j);
			numofsidelobe++;
		}
	}
	float mean_sidelobe = sumofsidelobe / numofsidelobe;
	float std_sidelobe = 0;
	for (int i = 0; i < height; ++i)
	{
		for (int j = 0; j < width; ++j)
		{
			if ((i >= th1) && (i <= th2) && (j >= th3) && (j <= th4))
			{
				continue;
			}
			std_sidelobe += (response.at<float>(i, j) - mean_sidelobe)*(response.at<float>(i, j) - mean_sidelobe);
		}
	}
	std_sidelobe = sqrt(std_sidelobe / numofsidelobe);
	float psr = (pv - mean_sidelobe) / (std_sidelobe+0.00002);
	return psr;
}

float calc_psr(cv::Mat response, cv::Point2f tk_pt)
{
	double pv;
	cv::Point2i pi;

	pi.x = tk_pt.x + response.cols / 2.0;
	pi.y = tk_pt.y + response.rows / 2.0;

	pv = response.at<float>(pi.y, pi.x);

	int width = response.cols;
	int height = response.rows;
	int peak_x = pi.x;
	int peak_y = pi.y;
	int peak_w = width / 10 > 1 ? width / 10 : 1;
	int peak_h = height / 10 > 1 ? height / 10 : 1;

	int th1 = peak_y - peak_h > 0 ? peak_y - peak_h : 0;
	int th2 = peak_y + peak_h < height - 1 ? peak_y + peak_h : height - 1;
	int th3 = peak_x - peak_w > 0 ? peak_x - peak_w : 0;
	int th4 = peak_x + peak_w < width - 1 ? peak_x + peak_w : width - 1;
	float sumofsidelobe = 0;
	int numofsidelobe = 0;

	for (int i = 0; i < height; ++i)
	{
		for (int j = 0; j < width; ++j)
		{
			if ((i >= th1) && (i <= th2) && (j >= th3) && (j <= th4))
			{
				continue;
			}
			sumofsidelobe += response.at<float>(i, j);
			numofsidelobe++;
		}
	}
	float mean_sidelobe = sumofsidelobe / numofsidelobe;
	float std_sidelobe = 0;
	for (int i = 0; i < height; ++i)
	{
		for (int j = 0; j < width; ++j)
		{
			if ((i >= th1) && (i <= th2) && (j >= th3) && (j <= th4))
			{
				continue;
			}
			std_sidelobe += (response.at<float>(i, j) - mean_sidelobe)*(response.at<float>(i, j) - mean_sidelobe);
		}
	}
	std_sidelobe = sqrt(std_sidelobe / numofsidelobe);
	float psr = (pv - mean_sidelobe) / (std_sidelobe+0.00002);
	return psr;
}

void sort_by_color_score(cv::Mat img, tk_data *ptk_data, std::vector<detect_candidate> &vec_candidate)
{
	for (size_t k = 0; k < vec_candidate.size(); k++)
	{
		cv::Rect tar_rect;
		tar_rect = vec_candidate[k].dect_rect;
		if (tar_rect.x<0 || tar_rect.y<0 || tar_rect.x + tar_rect.width>img.cols - 1 || tar_rect.y + tar_rect.height>img.rows - 1)
		{
			vec_candidate[k].score = 0;
			continue;
		}
		cv::Mat roi_mat = img(tar_rect);
		const int size[3] = { 32, 32, 32 };
		int roi_w = roi_mat.cols;
		int roi_h = roi_mat.rows;
		cv::MatND detect_hist = cv::Mat(3, size, CV_32F, float(1));
		for (int i = roi_h/4; i < roi_h*3/4; i++)
		{
			int start_x = roi_w/4;
			uchar* p_src = roi_mat.ptr<uchar>(i)+ start_x*3;
			for (int j = start_x; j < roi_w*3/4; j++)
			{
				int b = bin_map[*p_src++];
				int g = bin_map[*p_src++];
				int r = bin_map[*p_src++];
				detect_hist.at<float>(b, g, r)++;
			}
		}
		cv::normalize(detect_hist, detect_hist, 1, 0, CV_MINMAX);
		float *pdetect = detect_hist.ptr<float>(0);
		float *plearn = ptk_data->color_model.ptr<float>(0);
		float temp_score = 0;
		for (size_t i = 0; i < 32 * 32 * 32; i++)
		{
			temp_score += *pdetect++ * *plearn++;
		}
		vec_candidate[k].score *= temp_score;
	}
	std::sort(vec_candidate.begin(), vec_candidate.end(), sort_by_score);
}

int test_color_model(tk_data *ptk_data)
{
	const int size[3] = { 32, 32, 32 };
	cv::MatND new_color_model = cv::Mat(3, size, CV_32F, float(0));
	cv::MatND new_foreground = cv::Mat(3, size, CV_32F, float(1));
	cv::MatND new_background = cv::Mat(3, size, CV_32F, float(2));

	cv::Mat roi_mat = ptk_data->trans_roi_mat;
	int roi_w = roi_mat.cols;
	int roi_h = roi_mat.rows;

	for (int i = 0; i < roi_h; i++)
	{
		uchar* p_src = roi_mat.ptr<uchar>(i);
		for (int j = 0; j < roi_w; j++)
		{
			int b = bin_map[*p_src++];
			int g = bin_map[*p_src++];
			int r = bin_map[*p_src++];
			if (i>(roi_h / 4) && i<(roi_h * 3 / 4) && j>(roi_w / 4) && j<(roi_w * 3 / 4))
			{
				/*if (i>(roi_h *3.0 / 8) && i<(roi_h * 5.0 / 8) && j> (roi_w *3.0 / 8) && j<(roi_w * 5.0 / 8))
				{
				new_foreground.at<float>(b, g, r)++;
				}*/
				new_foreground.at<float>(b, g, r)++;
			}
			new_background.at<float>(b, g, r)++;
		}
	}
	new_color_model = new_foreground / new_background;

	//float *pdetect = new_foreground.ptr<float>(0);
	//float *plearn = ptk_data->color_foreground.ptr<float>(0);
	//float max_detect_val = 0;
	//float max_model_val = 0;
	//int index1 = 0;
	//int index2 = 0;
	//for (int i = 0; i < 32 * 32 * 32; ++i)
	//{

	//	if (max_detect_val<*pdetect)
	//	{
	//		max_detect_val = *pdetect;
	//		index1 = i;
	//	}
	//	if (max_model_val < *plearn)
	//	{
	//		max_model_val = *plearn;
	//		index2 = i;
	//	}
	//	pdetect++;
	//	plearn++;
	//}
	//printf("%d, %f, %d, %f\n", index1, max_detect_val, index2, max_model_val);

	int width = roi_mat.cols;
	int height = roi_mat.rows;
	cv::Mat reproject_img = cv::Mat(height, width, CV_32FC1, float(0));
	for (int i = 0; i < height; i++)
	{
		uchar* p_src = roi_mat.ptr<uchar>(i);
		for (int j = 0; j < width; j++)
		{
			int b = bin_map[*p_src++];
			int g = bin_map[*p_src++];
			int r = bin_map[*p_src++];
			reproject_img.at<float>(i, j) = new_foreground.at<float>(b, g, r);
		}
	}
	cv::normalize(reproject_img, reproject_img, 1, 0, CV_MINMAX);
	//cv::imshow("test_color_model", reproject_img);

	return 0;
}

void update_color_model(tk_data *ptk_data, float learning_rate)
{
	const int size[3] = { 32, 32, 32 };
	cv::MatND new_color_model = cv::Mat(3, size, CV_32F, float(0));
	cv::MatND new_foreground = cv::Mat(3, size, CV_32F, float(1));
	cv::MatND new_background = cv::Mat(3, size, CV_32F, float(2));

	cv::Mat roi_mat = ptk_data->trans_roi_mat;
	int roi_w = roi_mat.cols;
	int roi_h = roi_mat.rows;

	int th1 = roi_h / 4.0 + 0.5;
	int th2 = roi_h * 3.0 / 4.0 + 0.5;
	int th3 = roi_w / 4.0 + 0.5;
	int th4 = roi_w * 3.0 / 4.0 + 0.5;
	int th5 = roi_h * 3.0 / 8.0 + 0.5;
	int th6 = roi_h * 5.0 / 8.0 + 0.5;
	int th7 = roi_w * 3.0 / 8.0 + 0.5;
	int th8 = roi_w * 5.0 / 8.0 + 0.5;

	for (int i = 0; i < roi_h; i++)
	{
		uchar* p_src = roi_mat.ptr<uchar>(i);
		for (int j = 0; j < roi_w; j++)
		{
			int b = bin_map[*p_src++];
			int g = bin_map[*p_src++];
			int r = bin_map[*p_src++];
			if (i>th1 && i<th2 && j>th3 && j<th4)
			{
				if (i>th5 && i<th6 && j>th7 && j<th8)
				{
					new_foreground.at<float>(b, g, r)++;
				}
				new_foreground.at<float>(b, g, r) ++;
			}
			new_background.at<float>(b, g, r) ++;
		}
	}
	new_color_model = new_foreground / new_background;

	cv::normalize(new_color_model, new_color_model, 1, 0, CV_MINMAX);

	float local_learning_rate = 0;
	if (learning_rate < 0.05)
	{
		local_learning_rate = learning_rate/2.5;
	}
	else
	{
		local_learning_rate = learning_rate;
	}

	cv::addWeighted(ptk_data->color_foreground, (1 - local_learning_rate), new_foreground, local_learning_rate, 0, ptk_data->color_foreground);
	cv::addWeighted(ptk_data->color_background, (1 - local_learning_rate), new_background, local_learning_rate, 0, ptk_data->color_background);
	cv::addWeighted(ptk_data->color_model, (1 - local_learning_rate), new_color_model, local_learning_rate, 0, ptk_data->color_model);
}

cv::Mat calc_color_response(cv::Mat roi_mat, tk_data *ptk_data)
{
	int width = roi_mat.cols;
	int height = roi_mat.rows;
	cv::Mat reproject_img = cv::Mat(height, width, CV_32FC1, float(0)); 
	for (int i = 0; i < height; i++)
	{
		uchar* p_src = roi_mat.ptr<uchar>(i);
		for (int j = 0; j < width; j++)
		{
			int b = bin_map[*p_src++];
			int g = bin_map[*p_src++];
			int r = bin_map[*p_src++];
			reproject_img.at<float>(i, j) = ptk_data->color_model.at<float>(b, g, r);
		}
	}
	cv::normalize(reproject_img, reproject_img, 1, 0, CV_MINMAX);
	reproject_img = reproject_img.mul(ptk_data->haan_color_2D);//********//
	//cv::imshow("color mask", reproject_img);
	//cv::imshow("hanning windows", ptk_data->haan_color_2D);
	cv::Mat integral_mat;
	cv::integral(reproject_img, integral_mat);

	cv::Mat color_response = cv::Mat(height/2, width/2, CV_32FC1, float(0));

	int hw = width / 2.0 + 0.5;
	int hh = height / 2.0 + 0.5;
	int hhw = width / 8.0 + 0.5;
	int hhw3 = width * 3.0 / 8.0 + 0.5;
	int hhh = height / 8.0 + 0.5;
	int hhh3 = height * 3.0 / 8.0 + 0.5;
	for (int i = 0; i < height / 2.0; ++i)
	{
		for (int j = 0; j < width / 2.0; ++j)
		{
			float a = integral_mat.at<double>(i, j);
			float b = integral_mat.at<double>(i, j + hw);
			float c = integral_mat.at<double>(i + hh, j + hw);
			float d = integral_mat.at<double>(i + hh, j);

			float aa = integral_mat.at<double>(i + hhh, j + hhw);
			float bb = integral_mat.at<double>(i + hhh, j + hhw3);
			float cc = integral_mat.at<double>(i + hhh3, j + hhw3);
			float dd = integral_mat.at<double>(i + hhh3, j + hhw);
			color_response.at<float>(i, j) = (a+c-b-d)+(aa+cc-bb-dd)*1.2;
		}
	}
	cv::normalize(color_response, color_response, 1, 0, CV_MINMAX);
	//cv::imshow("color_reponse", color_response);
	return color_response;
}
#define COLOR
int icntt = 0;
float get_trans_response(tk_data *ptk_data, tk_params params, cv::Point2i &tk_pt)
{
	cv::Mat res_Mat = cv::Mat(ptk_data->sizeY, ptk_data->sizeX, CV_32FC2, cv::Scalar(0));
	for (size_t i = 0; i < ptk_data->vec_feature_2D.size(); ++i)
	{
		cv::Mat temp;
		cv::mulSpectrums(ptk_data->vec_hf_num[i], ptk_data->vec_feature_2D[i], temp, 0, false);
		cv::add(res_Mat, temp, res_Mat);
	}
	res_Mat = complexDivisionReal(res_Mat, real(ptk_data->hf_den) + params.lambda);
	cv::dft(res_Mat, res_Mat, cv::DFT_INVERSE | cv::DFT_SCALE);
	cv::Mat response = real(res_Mat);
	cv::Mat color_response = calc_color_response(ptk_data->trans_roi_mat, ptk_data);
	//cv::imshow("response", response);
	cv::resize(color_response, color_response, cv::Size(response.cols, response.rows));
	response = response.mul(color_response);
	response = response.mul(color_response);//********//

	cv::Point2f temp_pt;
	float score = calc_target_score(response, temp_pt);
	if (score>l_psr_th)
	{
#ifdef FHOG
		ptk_data->tk_ptf.x += temp_pt.x * ptk_data->curr_scale * ptk_data->scale_fator * 4;
		ptk_data->tk_ptf.y += temp_pt.y * ptk_data->curr_scale * ptk_data->scale_fator * 4;
#else
		ptk_data->tk_ptf.x += temp_pt.x * ptk_data->curr_scale * ptk_data->scale_fator;
		ptk_data->tk_ptf.y += temp_pt.y * ptk_data->curr_scale * ptk_data->scale_fator;
#endif
		tk_pt.x = ptk_data->tk_ptf.x + 0.5;
		tk_pt.y = ptk_data->tk_ptf.y + 0.5;
		if (tk_pt.x<0 || tk_pt.y<0 || tk_pt.x>ptk_data->img_width || tk_pt.y>ptk_data->img_height)
		{
			score = l_psr_th;
		}
		icntt = 0;
	}
#ifdef COLOR
	else
	{
		cv::Point2f color_pt;
		float color_score = calc_target_score(color_response, color_pt);
		icntt = 0;
		if ((color_score>2.0) &&(icntt<5))
		{
			icntt++;
			ptk_data->tk_color_ptf = ptk_data->tk_ptf;

			ptk_data->tk_color_ptf.x += color_pt.x * ptk_data->curr_scale * ptk_data->scale_fator * 4;
			ptk_data->tk_color_ptf.y += color_pt.y * ptk_data->curr_scale * ptk_data->scale_fator * 4;

			ptk_data->tk_ptf = ptk_data->tk_color_ptf;

			tk_pt.x = ptk_data->tk_ptf.x + 0.5;
			tk_pt.y = ptk_data->tk_ptf.y + 0.5;

			score = l_psr_th;
		}
	}
#endif
	return score;
}

float get_detect_response(tk_data *ptk_data, tk_params params, cv::Point2i &tk_pt)
{
	cv::Mat res_Mat = cv::Mat(ptk_data->sizeY, ptk_data->sizeX, CV_32FC2, cv::Scalar(0));
	for (size_t i = 0; i < ptk_data->vec_feature_2D.size(); ++i)
	{
		cv::Mat temp;
		cv::mulSpectrums(ptk_data->vec_hf_num[i], ptk_data->vec_feature_2D[i], temp, 0, false);
		cv::add(res_Mat, temp, res_Mat);
	}
	res_Mat = complexDivisionReal(res_Mat, real(ptk_data->hf_den) + params.lambda);
	cv::dft(res_Mat, res_Mat, cv::DFT_INVERSE | cv::DFT_SCALE);
	cv::Mat response = real(res_Mat);
	cv::Mat color_response = calc_color_response(ptk_data->trans_roi_mat, ptk_data);
	cv::resize(color_response, color_response, cv::Size(response.cols, response.rows));
	response = response.mul(color_response);
	response = response.mul(color_response);

	cv::Point2f temp_pt;
	float score = calc_target_score(response, temp_pt);

	tk_pt.x += temp_pt.x * ptk_data->curr_scale * ptk_data->scale_fator * 4 + 0.5;
	tk_pt.y += temp_pt.y * ptk_data->curr_scale * ptk_data->scale_fator * 4 + 0.5;
	return score;
}

void update_trans_model(tk_data *ptk_data, float learning_rate)
{
	for (size_t i = 0; i< ptk_data->vec_feature_2D.size(); ++i)
	{
		cv::Mat temp;
		cv::mulSpectrums(ptk_data->gauss_trans_2D, ptk_data->vec_feature_2D[i], temp, 0, true);
		cv::addWeighted(ptk_data->vec_hf_num[i], (1 - learning_rate), temp, learning_rate, 0, ptk_data->vec_hf_num[i]);
	}

	cv::Mat new_hf_den = cv::Mat(ptk_data->sizeY, ptk_data->sizeX, CV_32FC2, cv::Scalar(0));
	for (size_t i = 0; i< ptk_data->vec_feature_2D.size(); ++i)
	{
		cv::Mat temp; 
		cv::mulSpectrums(ptk_data->vec_feature_2D[i], ptk_data->vec_feature_2D[i], temp, 0, true);
		cv::add(temp, new_hf_den, new_hf_den);
	}
	cv::addWeighted(ptk_data->hf_den, (1 - learning_rate), new_hf_den, learning_rate, 0, ptk_data->hf_den);

	update_color_model(ptk_data, learning_rate);

	//calc_color_response(ptk_data->trans_roi_mat, ptk_data);
}

void update_scale_model(tk_data *ptk_data, float learning_rate)
{
	cv::Mat new_sf_num;
	cv::mulSpectrums(ptk_data->gauss_scale_2D, ptk_data->scale_feature_2D, new_sf_num, 0, true);
	cv::addWeighted(ptk_data->sf_num, (1 - learning_rate), new_sf_num, learning_rate, 0, ptk_data->sf_num);

	cv::Mat new_sf_den;
	cv::mulSpectrums(ptk_data->scale_feature_2D, ptk_data->scale_feature_2D, new_sf_den, 0, true);
	cv::reduce(real(new_sf_den), new_sf_den, 0, CV_REDUCE_SUM);
	cv::addWeighted(ptk_data->sf_den, (1 - learning_rate), new_sf_den, learning_rate, 0, ptk_data->sf_den);
}


void get_local_peak_candidate(cv::Mat response, std::vector<detect_candidate> &vec_candidate, int winsize)
{
	int height = response.rows;
	int width = response.cols;
	vec_candidate.clear();
	cv::Mat tem_mask = cv::Mat(height, width, CV_8UC1, uchar(1));

	int half_w = cv::min(7, winsize);
	int half_h = cv::min(7, winsize);

	for (int i = half_h; i < height-half_h; ++i)
	{
		for (int j = half_w; j < width-half_w; ++j)
		{
			float val = response.at<float>(i, j);
			if (tem_mask.at<uchar>(i, j)&&(val>0.5))
			{	
				bool b_is_peak = true;
				for (int m = -half_h; m <= half_h; ++m)
				{
					for (int n = -half_w; n <= half_w; ++n)
					{
						int loc_y = i + m;
						int loc_x = j + n;
						if (val < response.at<float>(loc_y, loc_x))
						{
							b_is_peak = false;
						}
						else if (val > response.at<float>(loc_y, loc_x))
						{
							tem_mask.at<uchar>(loc_y, loc_x) = 0;
						}
					}
				}
				if (b_is_peak)
				{
					detect_candidate temp;
					temp.score = val;
					temp.coor = cv::Point2f(j, i);
					vec_candidate.push_back(temp);
				}
			}
		}
	}
	int size = vec_candidate.size();
	std::sort(vec_candidate.begin(), vec_candidate.end(), sort_by_score);

	int isize = vec_candidate.size();
	isize = cv::min(isize, 7);
	vec_candidate.resize(isize);
}