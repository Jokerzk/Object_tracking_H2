#include "tk_interface.h"
#include "autel_tk.h"
#include "tk_api.h"

typedef struct
{
	const char* module;
	int (*SetTrackerArea)(float x, float y, float width, float height);
	void (*StopTracking)();
	int (*RegsiterTrackerResultInfo)(TRACKER_RESULT_APP p_function, TRACKER_RESULT_MOVIDIUS p_movidus);
	//int (*RegsiterTrackerResultInfo)(TRACKER_RESULT_APP p_function);
	int (*UnRegsiterTrackerResultInfo)(TRACKER_RESULT_APP p_function, TRACKER_RESULT_MOVIDIUS p_movidus);
	//int (*UnRegsiterTrackerResultInfo)(TRACKER_RESULT_APP p_function);
	int (*TrackerInterface)(unsigned char* buffer, int width, int height, int data_format, unsigned long long pts_camera);

	int (*SetMovidiusDataToH2)(void *data);
}tracker_API;

int g_isInitialized = 0;         // Flag for Initialization and Tracking
AutelRect g_initialBox;
TRACKER_RESULT_APP g_pFunction;  // Function to interact with the APP.
TRACKER_RESULT_MOVIDIUS g_pFunctionMovidus; // Function to interact with the MOVIDIUS.
char g_trackingStatus = 0;       // Track Termination Flag
tk_data *track_obj = NULL;

int SetTrackerArea(float x, float y, float width, float height)
{
	g_initialBox.x = x;
	g_initialBox.y = y;
	g_initialBox.width = width;
	g_initialBox.height = height;

	g_isInitialized = 2;  // Flag to Do Tracking

	return 1;
}
// Movidius --> H2
int SetMovidiusDataToH2(void* _data)
{
  movidius_pose* data = (movidius_pose*)_data;
	printf("timeStamp1: %lld, timeStamp2: %lld, pos: %f, %f, %f, vel: %f, %f, %f, rpy: %f, %f, %f\n",
		data->timeStamp1, data->timeStamp2, data->pos[0], data->pos[1], data->pos[2],
		data->vel[0], data->vel[1], data->vel[2], data->rpy[0], data->rpy[1], data->rpy[2]);
	return 1;
}

void StopTracking()
{
	g_isInitialized = 0;
	tracking_output_app tracking_output = { 0 };
	g_pFunction((void *)&tracking_output);
}

int RegsiterTrackerResultInfo(TRACKER_RESULT_APP p_function, TRACKER_RESULT_MOVIDIUS p_movidus)
{
	fprintf(stderr, "RegsiterTrackerResultInfo\n");
	g_pFunction = p_function;
	g_pFunctionMovidus = p_movidus;
	return 1;
}

int UnRegsiterTrackerResultInfo(TRACKER_RESULT_APP p_function, TRACKER_RESULT_MOVIDIUS p_movidus)
{
	if (g_pFunction == p_function)
		g_pFunction = NULL;
	if (g_pFunctionMovidus == p_movidus)
		g_pFunctionMovidus = NULL;
	return 1;
}

tk_params params;
AutelPoint2i tk_pt;
AutelSize tk_sz;
bool bisfirst = true;
int TrackerInterface(unsigned char* buffer, int width, int height, int data_format, unsigned long long pts_camera)
{
	// this define the factor of scaling back and forth from normal
	//screen to tablet screen
	int data_type = 0;
    int timestampstart = 0;
    int timestampstop = 0;
	if (g_isInitialized == 0 || width == 0 || height == 0 )
		return 0;

	AutelMat image;
	image.width = width;
	image.height = height;
	image.buffer = buffer;

	AutelRect return_box = g_initialBox;

	if (g_isInitialized == 2)
	{
        if (track_obj != NULL) {
            delete track_obj;
            track_obj = NULL;
        }
        params.padding = 1.0;
		params.output_sigma_factor = 1 / 16.0;
		params.scale_sigma_factor = 1 / 5.0;
		params.lambda = 1e-2;
		params.learning_rate = 0.030;
		params.number_of_scales = 33;
		params.scale_step = 1.04;
		params.scale_model_max_area = 512;
		bisfirst = true;
		///
		/// INITIAL FRAME        
		/// Conversion -> box: proportion --> pixel coordinate
		///
		if (g_initialBox.width < 1 && g_initialBox.height < 1)
		{
			g_initialBox.width = g_initialBox.width * image.width;
			g_initialBox.height = g_initialBox.height * image.height;
			g_initialBox.x = g_initialBox.x * image.width;
			g_initialBox.y = g_initialBox.y * image.height;
		}
		fprintf(stderr, "TrackerInterface...\n");
		g_isInitialized = 1;

		tk_pt.x = g_initialBox.x + g_initialBox.width / 2;
		tk_pt.y = g_initialBox.y + g_initialBox.height / 2;

		tk_sz.width = g_initialBox.width;
		tk_sz.height = g_initialBox.height;
		fprintf(stderr, "TrackerInterface width = %d, tk_sz.height = %d\n", tk_sz.width, tk_sz.height);
        track_obj = new tk_data;
		tk_init(image, tk_pt, tk_sz, track_obj, params);
	}
	else
	{
		///
		/// TARGET DETECTION ON THE GIVEN FRAME USING THE KCF
		///
		timestampstart = getcurtime();
		tk_track(image, tk_pt, tk_sz, track_obj, params);
        int roi_width = tk_sz.width*track_obj->scale_fator;
        int roi_height = tk_sz.height*track_obj->scale_fator;
        timestampstop = getcurtime();
		/// Return data to APP
		///
		g_trackingStatus = 0;
		if (track_obj->tk_psr > l_psr_th)
		{
			g_trackingStatus = 1;
		}
	#ifdef PROFLING
		printf("total time cost = %d, status: %d, sscore: %f\n", timestampstop - timestampstart, g_trackingStatus, track_obj->tk_psr);
    #endif
		float startx = (tk_pt.x - roi_width / 2.0) / (float)image.width;
		float starty = (tk_pt.y - roi_height / 2.0) / (float)image.height;
		float frame_width = roi_width / (float)image.width;
		float frame_height = roi_height / (float)image.height;
        //fprintf(stderr, "psr = %f, tk_pt.x = %d, y = %d, tk_sz.width = %d, height = %d,  g_trackingStatus = %d, tk_track timevale = %d, %f, %f, %f, %f\n",
        //    track_obj->tk_psr, tk_pt.x, tk_pt.y, roi_width, roi_height, g_trackingStatus, timestampstop - timestampstart,
        //    startx, starty, frame_width, frame_height);

		if(bisfirst)
		{
			bisfirst = false;
			g_trackingStatus = 2;
		}
		tracking_output_app tracking_output = { pts_camera, startx, starty, frame_width, frame_height, g_trackingStatus, 0 };
		g_pFunction((void *)&tracking_output);

		int stx = startx*image.width;
		int sty = starty*image.height;
		int stw = roi_width;
		int sth = roi_height;

		//tracking_output_movidius tracking_output_mv = {pts_camera, stx, sty, stw, sth, 0, 0, 0, 0, g_trackingStatus};
		//g_pFunctionMovidus((void*)&tracking_output_mv);
	}

	return 1;
}

extern "C" const tracker_API tracker = { "tracker", SetTrackerArea, StopTracking, RegsiterTrackerResultInfo, UnRegsiterTrackerResultInfo, TrackerInterface, SetMovidiusDataToH2};
