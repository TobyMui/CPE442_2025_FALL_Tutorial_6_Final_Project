#include <opencv2/opencv.hpp>
#include <pthread.h>
#include <atomic>
#include <iostream>
#include <cstdio>
#include <cctype>
#include <algorithm>
#include <math.h>
#include <arm_neon.h>
#include <chrono>
#include <papi.h>

using namespace std::chrono;

const int NUM_THREADS = 4; //Num Threads

// Sobel Filter Kernel
static int16_t X_KERNEL[9] = { -1,0,1, -2,0,2, -1,0,1 };
static int16_t Y_KERNEL[9] = { -1,-2,-1, 0,0,0, 1,2,1 };

// All great things start with an FSM 
enum class ComputeStages: int {GRAY_SCALE = 0, SOBEL_MAG = 1, STOP = 2}; 

// Shared data between main thread and worker threads
struct FrameJob {
    const cv::Mat* bgrFrame = nullptr; 
    cv::Mat* grayFrame = nullptr; 

    // Goal is to get rid of the need for sobleXOut and sobelYOut. We can calculate both at the same time then compute magnitude there. 
    cv::Mat* sobelXOut = nullptr; 
    cv::Mat* sobelYOut = nullptr; 
    cv::Mat* mag32fOut = nullptr; 
    int imgRows = 0, imgCols= 0; 

    std::atomic<ComputeStages> currentStage{ComputeStages::STOP}; //Shared Resource, so atomic required. 

    pthread_barrier_t startBarrier; 
    pthread_barrier_t endBarrier;
};

struct WorkerContext { 
    FrameJob* job = nullptr; 
    int tid = 0; 
};

//Calculates the magnitude of GX and GY
static inline void magnitude(const cv::Mat& gx32f, const cv::Mat& gy32f, cv::Mat& mag32f, int yStart, int yEnd){
    const int rows = gx32f.rows,  cols = gx32f.cols;
    const int ys = std::max(0, yStart); // Starting row, max to prevent clipping
    const int ye = std::min(rows, yEnd); // Ending row, min to prevent clipping

    for(int y = ys; y < ye; ++y){
        const float* gX_row = gx32f.ptr<float>(y);
        const float* gY_row = gy32f.ptr<float>(y); 
        float* out = mag32f.ptr<float>(y); 
        for (int x = 0; x < cols; ++x) {
            out[x] = sqrt(gX_row[x] * gX_row[x] + gY_row[x] * gY_row[x]);
        }
    }
}

// New function that calculates X, Y , Mag in 1 go. 
void sobel_calculation(const cv::Mat& gray, cv::Mat& mag32f, int yStart, int yEnd){
    const int border = 1; 
    const int rows = gray.rows, cols = gray.cols; 

    // Bounds checking
    const int ys = std::max(border, yStart); 
    const int ye = std::min(rows - border, yEnd); 

    // Iterating the columns
    for (int y = ys; y < ye; ++y) {
        const uchar* prev = gray.ptr<uchar>(y - 1);
        const uchar* curr = gray.ptr<uchar>(y);
        const uchar* next = gray.ptr<uchar>(y + 1);
        float* out = mag32f.ptr<float>(y);
        
        // Iterating the rows 
        for (int x = border; x < cols - border; ++x) {
            // Sobel X
            float gx =
                prev[x-1]*X_KERNEL[0] + prev[x]*X_KERNEL[1] + prev[x+1]*X_KERNEL[2] +
                curr[x-1]*X_KERNEL[3] + curr[x]*X_KERNEL[4] + curr[x+1]*X_KERNEL[5] +
                next[x-1]*X_KERNEL[6] + next[x]*X_KERNEL[7] + next[x+1]*X_KERNEL[8];

            // Sobel Y
            float gy =
                prev[x-1]*Y_KERNEL[0] + prev[x]*Y_KERNEL[1] + prev[x+1]*Y_KERNEL[2] +
                curr[x-1]*Y_KERNEL[3] + curr[x]*Y_KERNEL[4] + curr[x+1]*Y_KERNEL[5] +
                next[x-1]*Y_KERNEL[6] + next[x]*Y_KERNEL[7] + next[x+1]*Y_KERNEL[8];

            out[x] = std::sqrt(gx*gx + gy*gy);
        }
    }
};

static inline __attribute__((noinline)) void grayscale_calculation(const cv::Mat& bgr, cv::Mat& gray,int yStart, int yEnd){
    const int rows = bgr.rows, cols = bgr.cols;
    const int ys = std::max(0, yStart);
    const int ye = std::min(rows, yEnd);

    // auto start = high_resolution_clock::now();

    for (int y = ys; y < ye; ++y) {
        const uchar* in = bgr.ptr<uchar>(y);
        uchar* out = gray.ptr<uchar>(y);

        int x = 0;
        for (; x <= cols - 8; x += 8) {
            // const uchar B = in[x][0], G = in[x][1], R = in[x][2];
            // leads in 8 pixels, each pixel has three color chanels
            uint8x8x3_t bgrPix = vld3_u8(in + 3 * x);

            uint16x8_t gray16 = vmull_u8(bgrPix.val[2], vdup_n_u8(77));   // R * 77
            gray16 = vmlal_u8(gray16, bgrPix.val[1], vdup_n_u8(150));     // + G * 150
            gray16 = vmlal_u8(gray16, bgrPix.val[0], vdup_n_u8(29));      // + B * 29
            gray16 = vaddq_u16(gray16, vdupq_n_u16(128));                 // + 128 bias
            // out[x] = static_cast<uchar>((R*77 + G*150 + B*29 + 128) >> 8);

            // Shift right by 8 bits and narrow back to 8-bit
            uint8x8_t gray8 = vshrn_n_u16(gray16, 8);

            // Store the result
            vst1_u8(out + x, gray8);
        
        }

        // Process leftover pixels
        for (; x < cols; x++) {
            // const uchar B = in[x][0], G = in[x][1], R = in[x][2];
            const uchar B = in[3*x + 0], G = in[3*x + 1], R = in[3*x + 2];
            out[x] = static_cast<uchar>((R*77 + G*150 + B*29 + 128) >> 8);
        }
    }
}

void* workerThread(void* arg){
    WorkerContext* ctx = static_cast<WorkerContext*>(arg);
    FrameJob* job = ctx->job;

    for (;;) {
        // Wait for main thread to assign a new stage
        pthread_barrier_wait(&job->startBarrier);

        ComputeStages stage = job->currentStage.load(std::memory_order_acquire);
        if (stage == ComputeStages::STOP) break;

        // Compute my portion of rows for this stage
        int totalRows = job->imgRows;
        int rowsPerThread = (totalRows + NUM_THREADS - 1) / NUM_THREADS;
        int yStart = ctx->tid * rowsPerThread;
        int yEnd = std::min(totalRows, yStart + rowsPerThread);

        if (stage == ComputeStages::SOBEL_MAG) {
            sobel_calculation(*job->grayFrame, *job->mag32fOut, yStart, yEnd);
        } else if (stage == ComputeStages::GRAY_SCALE){
            grayscale_calculation(*job->bgrFrame,*job->grayFrame,yStart, yEnd);
        }

        // Signal that this thread has finished its slice
        pthread_barrier_wait(&job->endBarrier);
    }

    // Final sync so main can safely join threads
    pthread_barrier_wait(&job->endBarrier);
    return nullptr;
}

int main(int argc, char** argv){
    #ifdef __ARM_NEON
        std::cout << "NEON supported and enabled!\n";
    #endif

    if(argc < 2){
        std::cerr << "Usage: " << argv[0] << " video_path or camera index>\n";
        return 1; 
    }

    // SETS UP PAPI
    int EventSet_Sobel = PAPI_NULL;
    int EventSet_Grey = PAPI_NULL;
    long long values[2][3]; // Array to store event values (e.g., PAPI_REAL_USEC)
    long long avg_sobel = 0;
    long long avg_sobel_l1dcm = 0;
    long long avg_sobel_l2dcm = 0;
    long long avg_grey = 0;
    long long avg_grey_l1dcm = 0;
    long long avg_grey_l2dcm = 0;
    long long num_events = 0;
    int retval;

    retval = PAPI_library_init(PAPI_VER_CURRENT);
    if (retval != PAPI_VER_CURRENT && retval < 0) {
        fprintf(stderr, "PAPI library init error: %s\n", PAPI_strerror(retval));
        return 1;
    }

    retval = PAPI_create_eventset(&EventSet_Sobel);
    if (retval != PAPI_OK) {
        fprintf(stderr, "PAPI create eventset error: %s\n", PAPI_strerror(retval));
        return 1;
    }
    retval = PAPI_create_eventset(&EventSet_Grey);
    if (retval != PAPI_OK) {
        fprintf(stderr, "PAPI create eventset error: %s\n", PAPI_strerror(retval));
        return 1;
    }

    retval = PAPI_add_event(EventSet_Sobel, PAPI_TOT_CYC);
    retval = PAPI_add_event(EventSet_Sobel, PAPI_L1_DCM);
    retval = PAPI_add_event(EventSet_Sobel, PAPI_L2_DCM);
    if (retval != PAPI_OK) {
        fprintf(stderr, "PAPI add event error sobel: %s\n", PAPI_strerror(retval));
        return 1;
    }
    retval = PAPI_add_event(EventSet_Grey, PAPI_TOT_CYC);
    retval = PAPI_add_event(EventSet_Grey, PAPI_L1_DCM);
    retval = PAPI_add_event(EventSet_Grey, PAPI_L2_DCM);
    if (retval != PAPI_OK) {
        fprintf(stderr, "PAPI add event error grey: %s\n", PAPI_strerror(retval));
        return 1;
    }

    cv::VideoCapture cap; 
    if(std::isdigit(argv[1][0])) cap.open(std::stoi(argv[1]));
    else cap.open(argv[1]);
    if(!cap.isOpened()){
        std::cerr << "Error: could not open source.\n";
        return 1; 
    }

    FrameJob job; 
    pthread_barrier_init(&job.startBarrier,nullptr,NUM_THREADS + 1);
    pthread_barrier_init(&job.endBarrier, nullptr, NUM_THREADS + 1); 

    pthread_t threads[NUM_THREADS]; 
    WorkerContext workerCtx[NUM_THREADS]; 
    for (int i = 0; i < NUM_THREADS; ++i){
        workerCtx[i] = WorkerContext{&job, i }; 
        pthread_create(&threads[i], nullptr, workerThread, &workerCtx[i]);
    }

    cv::namedWindow("Sobel", cv::WINDOW_AUTOSIZE);

    //Init for grayscale and sobel
    cv::Mat frame, gray, mag32f, mag8u; 

    while(1){
        if (!cap.read(frame) || frame.empty()) break;

        // (Re)allocate outputs on size change
        if (gray.size() != frame.size()) {
            gray.create(frame.size(), CV_8UC1);
            mag32f.create(frame.size(), CV_32FC1);
        }
    
        // Share current frame info
        job.bgrFrame  = &frame;
        job.grayFrame = &gray;
        job.mag32fOut = &mag32f; 
        job.imgRows   = gray.rows;
        job.imgCols   = gray.cols;
    
      // ---- Stage 1: Parallel GRAYSCALE ----
        job.currentStage.store(ComputeStages::GRAY_SCALE, std::memory_order_release);
        PAPI_start(EventSet_Grey);
        pthread_barrier_wait(&job.startBarrier);
        pthread_barrier_wait(&job.endBarrier);
        PAPI_stop(EventSet_Grey, values[0]);

        // ---- Stage 2: Parallel SOBEL (Both X and Y) ----
        job.currentStage.store(ComputeStages::SOBEL_MAG, std::memory_order_release);
        PAPI_start(EventSet_Sobel);
        pthread_barrier_wait(&job.startBarrier);
        pthread_barrier_wait(&job.endBarrier);
        PAPI_stop(EventSet_Sobel, values[1]);

        // add to running totals
        avg_sobel += values[1][0];
        avg_sobel_l1dcm += values[1][1];
        avg_sobel_l2dcm += values[1][2];
        avg_grey += values[0][0];
        avg_grey_l1dcm += values[0][1];
        avg_grey_l2dcm += values[0][2];
        num_events += 1;

        // reset the Event Sets
        PAPI_reset(EventSet_Grey);
        PAPI_reset(EventSet_Sobel);

        // Combine results & display
        mag32f.convertTo(mag8u, CV_8U, 0.5);
        cv::imshow("Sobel", mag8u);

        // Close Window
        int key = cv::waitKey(1);             
        if (key == 27) break;                    // ESC to quit
        if (cv::getWindowProperty("Sobel", cv::WND_PROP_VISIBLE) < 1) break;
    }

    // Tell workers to stop
    job.currentStage.store(ComputeStages::STOP, std::memory_order_release);
    pthread_barrier_wait(&job.startBarrier);
    pthread_barrier_wait(&job.endBarrier);

    for (int i = 0; i < NUM_THREADS; ++i) pthread_join(threads[i], nullptr);
    pthread_barrier_destroy(&job.startBarrier);
    pthread_barrier_destroy(&job.endBarrier);
    

    avg_sobel /= num_events;
    avg_grey /= num_events;
    avg_grey_l1dcm /= num_events;
    avg_grey_l2dcm /= num_events;
    avg_sobel_l1dcm /= num_events;
    avg_sobel_l2dcm /= num_events;

    std::cout << "\nGreyscale | Time (Cycles): " << avg_grey
          << " | L1 Cache Miss: " << avg_grey_l1dcm
          << " | L2 Cache Miss: " << avg_grey_l2dcm << "\n";

    std::cout << "Sobel | Time (Cycles): " << avg_sobel
          << " | L1 Cache Miss: " << avg_sobel_l1dcm
          << " | L2 Cache Miss: " << avg_sobel_l2dcm << "\n";
    
    PAPI_shutdown();

    return 0;
}

