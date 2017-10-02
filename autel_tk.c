#include "autel_tk.h"
#include "rectangle.h"

#ifdef HAVE_TBB
#include <tbb/tbb.h>
#include "tbb/parallel_for.h"
#include "tbb/blocked_range.h"
#endif

void tk_init(AutelMat au_img, AutelPoint2i au_tk_pt, AutelSize au_init_sz, tk_data *ptk_data, tk_params &params)
{
	cv::Mat img = cv::Mat(au_img.height, au_img.width, CV_8UC3,au_img.buffer);
	cv::Point2i tk_pt;
	tk_pt.x = au_tk_pt.x;
	tk_pt.y = au_tk_pt.y;//zhongxinweizhi
	cv::Size init_sz;
	init_sz.width = au_init_sz.width;
	init_sz.height = au_init_sz.height;//chushihuakuangdaxiao 
#ifdef PROFLING
   int timestart = getcurtime();
 #endif
	getoptimizerect(img,tk_pt,init_sz);
#ifdef PROFLING
   int timestop = getcurtime();
   printf("salient detect time cost = %d\n", timestop - timestart);
 #endif

	tk_init(img, tk_pt, init_sz, ptk_data, params);
}

void tk_init(cv::Mat img, cv::Point2i tk_pt, cv::Size init_sz, tk_data *ptk_data, tk_params &params)
{
	///////////////////////////////////////////////////////////
	//////prepare init data struct////////////////////////////
	init_api(img, ptk_data, init_sz, params);

	ptk_data->ilost_cnt = 0;
	ptk_data->tk_ptf = tk_pt;
	ptk_data->tk_color_ptf = tk_pt;
	////////////////////////////////////////////////////////////
	//////////for trans module//////////////////////////////////
	get_trans_sample(img, tk_pt, init_sz, ptk_data, params);

	for (size_t i = 0; i < ptk_data->vec_feature_2D.size(); ++i)
	{
		ptk_data->vec_hf_num.push_back(cv::Mat(ptk_data->sizeY, ptk_data->sizeX, CV_32FC2, cv::Scalar(0)));
	}
	ptk_data->hf_den = cv::Mat(ptk_data->sizeY, ptk_data->sizeX, CV_32FC2, cv::Scalar(0));
	update_trans_model(ptk_data, 1.0);
	//get_trans_response(ptk_data, params, tk_pt);
	
	////////////////////////////////////////////////////////////
	//////////for scale module//////////////////////////////////
	cv::cvtColor(img, ptk_data->gray, CV_BGR2GRAY);
	get_scale_sample(ptk_data->gray, tk_pt, init_sz, ptk_data, params);
	update_scale_model(ptk_data, 1.0);
	//get_scale_response(ptk_data, params);
	//get_space_model(ptk_data, params, cv::Mat());
}

void tk_track(AutelMat au_img, AutelPoint2i &au_tk_pt, AutelSize &au_tk_sz, tk_data *ptk_data, tk_params &params)
{
#ifdef PROFLING
	int timestampstart = getcurtime();
#endif
	uchar * pdst = ptk_data->input_img.data;
	uchar * psrc = au_img.buffer;
	for (int i = 0; i <au_img.height*au_img.width*3; ++i)
	{
		*pdst++ = *psrc++;
	}
#ifdef PROFLING
	int timestampstop = getcurtime();
	printf("data copy = %d\n", timestampstop - timestampstart);
#endif
	cv::Point2i tk_pt;
	tk_pt.x = au_tk_pt.x;
	tk_pt.y = au_tk_pt.y;
	cv::Size tk_sz;
	tk_sz.width = au_tk_sz.width;
	tk_sz.height = au_tk_sz.height;
	tk_track(ptk_data->input_img, tk_pt, tk_sz, ptk_data, params);

	au_tk_pt.x = tk_pt.x;
	au_tk_pt.y = tk_pt.y;
	au_tk_sz.width = tk_sz.width;
	au_tk_sz.height = tk_sz.height;

	if (ptk_data->tk_psr>h_psr_th)
	{
		ptk_data->ilost_cnt = 0;
	}
	else
	{
		if (ptk_data->ilost_cnt > 60)
		{
			cv::Point2i detect_pt;
			cv::Size detect_sz = tk_sz;
			tk_detect(ptk_data->input_img, detect_pt, detect_sz, ptk_data, params);
		#ifdef PROFLING
			printf("detect psr = %f\n", ptk_data->dt_psr);
			au_tk_pt.x = detect_pt.x;
			au_tk_pt.y = detect_pt.y;
			au_tk_sz.width = detect_sz.width;
			au_tk_sz.height = detect_sz.height;
		#endif
			if (ptk_data->dt_psr > h_psr_th-1)
			{
				ptk_data->tk_ptf.x = detect_pt.x;
				ptk_data->tk_ptf.y = detect_pt.y;
			}
		}
		ptk_data->ilost_cnt++;
	}
}

void tk_track(cv::Mat img, cv::Point2i &tk_pt, cv::Size &tk_sz, tk_data *ptk_data, tk_params &params)
{
    int timestampstart = 0;
    int timestampstop = 0;
	/////////////////
	////track////////
#ifdef PROFLING
	timestampstart = getcurtime();
#endif
	
	get_trans_sample(img, tk_pt, tk_sz, ptk_data, params);
#ifdef PROFLING
    timestampstop = getcurtime();
	printf("translation sample = %d\n", timestampstop - timestampstart);
#endif

#ifdef PROFLING
	timestampstart = getcurtime();
#endif
	ptk_data->tk_psr = get_trans_response(ptk_data, params, tk_pt);

#ifdef PROFLING
    timestampstop = getcurtime();
	printf("translation response = %d\n", timestampstop - timestampstart);
#endif
	if (ptk_data->tk_psr < h_psr_th)
	{
		return;
	}

#ifdef PROFLING
	timestampstart = getcurtime();
#endif
	cv::cvtColor(img, ptk_data->gray, CV_BGR2GRAY);
	get_scale_sample(ptk_data->gray, tk_pt, tk_sz, ptk_data, params);

#ifdef PROFLING
	timestampstop = getcurtime();
    printf("scale sample = %d\n", timestampstop - timestampstart);
#endif

	get_scale_response(ptk_data, params, tk_sz);

#ifdef PROFLING
	timestampstart = getcurtime();
    printf("scale response = %d\n", timestampstart - timestampstop);
#endif
	/////////////////////////////
	///update trans model////////
	get_trans_sample(img, tk_pt, tk_sz, ptk_data, params);
	update_trans_model(ptk_data, params.learning_rate);
#ifdef PROFLING
    timestampstop = getcurtime();
	printf("translation update timevalue = %d\n", timestampstop - timestampstart);
#endif
	/////////////////////////////
	///update scale model////////
	get_scale_sample(ptk_data->gray, tk_pt, tk_sz, ptk_data, params);
	update_scale_model(ptk_data, params.learning_rate/2);
#ifdef PROFLING
	timestampstart = getcurtime();
    printf("scale update timevalue = %d\n", timestampstart - timestampstop);
#endif
	//cv::Mat space_model;
	//get_space_model(ptk_data, params, space_model);
}

void tk_release(tk_data *ptk_data, tk_params &params)
{
	if (ptk_data->scale_factors)
		delete ptk_data->scale_factors;
}

void tk_detect(cv::Mat img, cv::Point2i &tk_pt, cv::Size &tk_sz, tk_data *ptk_data, tk_params &params)
{
	int curr_width = tk_sz.width * ptk_data->scale_fator*2;
	int curr_height = tk_sz.height * ptk_data->scale_fator*2;

	cv::Mat space_model;
	space_model = get_space_model(ptk_data);
	cv::resize(space_model, space_model, cv::Size(curr_width, curr_height));

	cv::Rect roi_rect;
	roi_rect.x  = img.cols / 8;
	roi_rect.y = img.rows / 8;
	roi_rect.width = img.cols * 3/4;
	roi_rect.height = img.rows * 3/4;

	cv::Mat roi_mat = img(roi_rect);
	cv::cvtColor(roi_mat, roi_mat, CV_BGR2GRAY);
	roi_mat.convertTo(roi_mat, CV_32F, 1 / 255.f);
	ptk_data->dt_psr = 0;

	if ((space_model.rows < roi_mat.rows) && (space_model.cols < roi_mat.cols))
	{
		cv::Mat result;
		cv::matchTemplate(roi_mat, space_model, result, cv::TM_CCOEFF_NORMED);

		cv::normalize(result, result, 0, 1, CV_MINMAX);
		//cv::imshow("result", result);
		std::vector<detect_candidate> vec_candidate;
		int winsize = (curr_width + curr_height) / 16;
		get_local_peak_candidate(result, vec_candidate, winsize);
		ptk_data->dt_psr = 1.1;
		if (vec_candidate.size() > 0)
		{
			ptk_data->dt_psr = 1.2;
			for (size_t i = 0; i < vec_candidate.size(); i++)
			{
				cv::Rect tar_rect;
				float height = tk_sz.height;
				float width = tk_sz.width;
				tar_rect.x = vec_candidate[i].coor.x + roi_rect.x + curr_width/4.0;
				tar_rect.y = vec_candidate[i].coor.y + roi_rect.y + curr_height/4.0;
				tar_rect.height = curr_height / 2.0;
				tar_rect.width = curr_width / 2.0;
				vec_candidate[i].dect_rect = tar_rect;
			}
			sort_by_color_score(img, ptk_data, vec_candidate);
			tk_pt.x = vec_candidate[0].dect_rect.x + vec_candidate[0].dect_rect.width / 2;
			tk_pt.y = vec_candidate[0].dect_rect.y + vec_candidate[0].dect_rect.height / 2;
			std::vector<cv::Mat> vec_detect_feature;
			get_detect_sample(img, tk_pt, tk_sz, ptk_data, params);
			ptk_data->dt_psr = get_detect_response(ptk_data, params, tk_pt);
		}
	}
	else
	{
		ptk_data->dt_psr = 1.3;
	}
}