#ifndef YOLO_DETECTOR_H
#define YOLO_DETECTOR_H

#define OPENCV
#define GPU 1
#define FPS
//#define SHOW_DETECTED
//#define CAMERA
//#define SAVE

#include <torch/script.h>
#include <torch/utils.h>
#include <opencv2/opencv.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "Yolo3Detection.h"
#include "yolo_v2_class.hpp"
#include <thread>
#include "kalman.h"

#define center_detect_tolerant_pixels 5
#define canny_threshold_min 30
#define canny_threshold_max 70
#define houghline_threshold 20
#define dilate_kernel_size 2
#define open_kernel_size 3
#define detect_number_area_width_height_ratio 0.2
#define detect_number_area_center_ratio 0.1

template<typename T>
class send_one_replaceable_object_t {
    const bool sync;
    std::atomic<T *> a_ptr;
public:
    void send(T const &_obj) {
        T *new_ptr = new T;
        *new_ptr = _obj;
        if (sync) {
            while (a_ptr.load()) std::this_thread::sleep_for(std::chrono::milliseconds(3));
        }
        std::unique_ptr<T> old_ptr(a_ptr.exchange(new_ptr));
    }

    T receive() {
        std::unique_ptr<T> ptr;
        do {
            while (!a_ptr.load()) std::this_thread::sleep_for(std::chrono::milliseconds(3));
            ptr.reset(a_ptr.exchange(NULL));
        } while (!ptr);
        T obj = *ptr;
        return obj;
    }

    send_one_replaceable_object_t(bool _sync) : sync(_sync), a_ptr(NULL) {}
};

struct infer_data_t {
    position p;
    position p_top;
    cv::Mat target_frame;
    cv::Mat cut_frame;
    int prediction;
    bool has_top_box;

    infer_data_t() : has_top_box(false), prediction(0) {}
};

struct detection_data_t {
    cv::Mat cap_frame;
    std::vector<position> p_vec;
    std::vector<infer_data_t> infer_data;
    bool exit_flag;

    detection_data_t() : exit_flag(false) {}
};

struct infer_out_t {
    int digit1_prediction;
    int digit2_prediction;
    float digit1_weight;
    float digit2_weight;
};

class yolo_detector {
private:
    cv::VideoCapture cap;
    cv::VideoWriter vw;
    cv::Size frame_size;

    std::shared_ptr<torch::jit::script::Module> module;
    torch::NoGradGuard noGradGuard;
    tk::dnn::Yolo3Detection yolo;
    tk::dnn::DetectionNN *detNN;
    tk::dnn::Yolo3Detection yolo_top;
    tk::dnn::DetectionNN *detNN_top;
    Detector *detector_middle;

    track_kalman_t track_kalman;
    std::vector<float *> track_id_perdictions_digit1_weight;
    std::vector<float *> track_id_perdictions_digit2_weight;
    std::vector<std::vector<int>> track_id_perdictions_all_digits;
    int max_tack_id = 0;

    std::thread t_cap, t_detect, t_top_prepare, t_top_detect, t_cut, t_infer, t_show;
    send_one_replaceable_object_t<detection_data_t> cap2prepare, detect2top_prepare, top_prepare2top_detect, top_detect2cut, cut2infer, infer2show;
    std::atomic<bool> exit_flag;

#ifdef FPS
    std::atomic<int> fps_cap_counter, fps_det_counter;
    std::atomic<int> current_fps_cap, current_fps_det;
    std::chrono::steady_clock::time_point steady_start, steady_end;
#endif

    /**
     * infer the number on an image
     *
     * @param image an image of cv::Mat form
     * @return infer_out_t structure contains each digit and its weight
     */
    infer_out_t infer(cv::Mat &image);
    /**
     * rotate an image
     *
     * @param src an image of cv::Mat form to be rotated
     * @param angle angle of rotation
     * @return image rotated
     */
    static cv::Mat RotateImage(cv::Mat &src, float angle);
    /**
     * rotate an image
     *
     * @param src an image of cv::Mat form to be rotated
     * @param angle angle of rotation
     * @param center_x x-coord of the center of rotation
     * @param center_y y-coord of the center of rotation
     * @return image rotated
     */
    static cv::Mat RotateImage(cv::Mat &src, float angle, int center_x, int center_y);
    /**
     * make a slightly inclinded image to be erect
     *
     * @param src an image of cv::Mat form to be make upright
     * @return an erect image
     */
    static cv::Mat make_upright(cv::Mat &src);
    /**
     * detect the area which contains the digits
     *
     * @param src an image of cv::Mat form where detection take place
     * @return image of the area which contains the digits
     */
    static cv::Mat detect_number_area(cv::Mat &src);
    /**
     * sharpen image
     *
     * @param src image to be shapened
     * @return sharpened image
     */
    static cv::Mat sharpen(cv::Mat &src);
    /**
     * correct the current number(number1-left) using detection history of numbers on the same place of the same target
     *
     * @param current_prediction
     * @param weight confidence of the current number
     * @param track_id track_id of the target which contains this current number
     * @return number corrected
     */
    int maxnum1(int current_prediction, float weight, int track_id);
    /**
     * correct the current number(number2-right) using detection history of numbers on the same place of the same target
     *
     * @param current_prediction
     * @param weight confidence of the current number
     * @param track_id track_id of the target which contains this current number
     * @return number corrected
     */
    int maxnum2(int current_prediction, float weight, int track_id);

public:
    yolo_detector();

    int start();
};

#endif