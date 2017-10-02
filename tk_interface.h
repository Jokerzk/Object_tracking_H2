#ifndef _TK_INTERFACE_H_
#define _TK_INTERFACE_H_
#ifdef __cplusplus
  extern "C" {
#endif

typedef struct {
  unsigned long long timeStamp;
  int u; // pixel coordinate x;
  int v; // pixel coordinate y;
  int w;           // width of bounding box in pixel
  int h;           // height of bounding box in pixel
  float dist1;     // yaw in deg
  float dist2;     // ground coordinate x
  float dist3;     // ground coordinate y
  float dist4;     // ground coordinate z
  int   status;
} tracking_output_movidius;
typedef int(*TRACKER_RESULT_MOVIDIUS)(void *data);
//// end

//// transmitting Movidius to H2
typedef struct {
    unsigned long long timeStamp1;
    unsigned long long timeStamp2;
    float pos[3];
    float vel[3];
    float rpy[3];
} movidius_pose;
// Movidius ---> H2
int SetMovidiusDataToH2(void *data);

//// H2-->App: define box of object, tracking status, distance between camera to object
typedef struct {
	unsigned long long timeStamp; //time stamp
	float x;  // ratio of x coordinate in image width
	float y;  // ratio of y coordinate in image height
	float width;  //ratio of box width in image width;
	float height; // ratio of box height in image height;
	char status;  // tracking status: 0: failed; 1: OK;
	float distance; // distance between camera with target
} tracking_output_app;

typedef int(*TRACKER_RESULT_APP)(void *data);

/*tracker info transfer movidius, datalen must less than 255*/
typedef int(*TRACKER_INFO_TRANSFER_MOVIDIUS)(void *data, int datalen);
// set bounding box of object for initializing tracking algorithm
// API
// int SetTrackerArea(float x, float y, float width, float height);
//
// x                - x coordinate of object bounding box in first frame(proportion to image)
// y                - y coordinate of object bounding box in first frame(proportion to image)
// width            - width of object bounding box in first frame(proportion to image)
// height           - height of object bounding box in first frame(proportion to image)
int SetTrackerArea(float x, float y, float width, float height);

// stop tracking algorithm
// API
// void StopTracking();
void StopTracking();

// recall function for getting result of tracking
// API
// int RegsiterTrackerResultInfo(TRACKER_RESULT_APP p_function, START_RECORD_VIDEO p_start_record, STOP_RECORD_VIDEO p_stop_record, TRACKER_RESULT_MOVIDIUS p_movidus);
//
// p_function         - function pointer
int RegsiterTrackerResultInfo(TRACKER_RESULT_APP p_function, TRACKER_RESULT_MOVIDIUS p_movidus);

// release function pointer
// API
// int UnRegsiterTrackerResultInfo(TRACKER_RESULT_APP p_function, START_RECORD_VIDEO p_start_record, STOP_RECORD_VIDEO p_stop_record, TRACKER_RESULT_MOVIDIUS p_movidus);
//
// p_function         - function pointer
int UnRegsiterTrackerResultInfo(TRACKER_RESULT_APP p_function, TRACKER_RESULT_MOVIDIUS p_movidus);

// interface of tracking algorithm, input: initialized bounding box of object and frames of video;
// output: rectangle of tracking for object in frame
//
// API
// AuRectf TrackerInterface(AuRectf initial_box, AuMat image);// INPUT

// buffer             - data of every frame
// width              - image width
// height             - image height
// data_fomat         - data format: 0: ( YUV 4:2:0 ); 1: ( YUV 4:2:2 ); 2: ( RGB or BGR )
int TrackerInterface(unsigned char* buffer, int width, int height, int data_format, unsigned long long pts_camera);

#ifdef __cplusplus
  }
#endif

#endif