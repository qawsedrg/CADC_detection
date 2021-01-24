#include "yolo_detector.h"

infer_out_t yolo_detector::infer(cv::Mat &image) {
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    cv::resize(image, image, cv::Size(54, 54));
    image = image(cv::Rect(0, 0, 54, 54)).clone();

    torch::Tensor images_tensor = torch::from_blob(image.data, {image.rows, image.cols, image.channels()},
                                                   torch::kByte).unsqueeze(0);
    images_tensor = images_tensor.permute({0, 3, 1, 2});
    images_tensor = images_tensor.toType(torch::kFloat);
    images_tensor = images_tensor.div(255);
    images_tensor = images_tensor.sub(0.5).div(0.5);
    images_tensor = images_tensor.to(torch::kCUDA);

    auto outputs = module->forward({images_tensor}).toTuple()->elements();

    torch::Tensor digit1_logits = outputs[0].toTensor();
    torch::Tensor digit2_logits = outputs[1].toTensor();
    auto digit1_prediction = std::get<1>(digit1_logits.max(1)).item<int>();
    auto digit2_prediction = std::get<1>(digit2_logits.max(1)).item<int>();

    infer_out_t result;
    result.digit1_prediction = digit1_prediction;
    result.digit2_prediction = digit2_prediction;
    result.digit1_weight = digit1_logits[0][digit1_prediction].item<float>();
    result.digit2_weight = digit2_logits[0][digit2_prediction].item<float>();
    return result;
}

cv::Mat yolo_detector::RotateImage(cv::Mat &src, float angle) {
    cv::Mat dst;
    cv::Size dst_sz(src.cols, src.rows);
    cv::Point2f center(src.cols / 2., src.rows / 2.);
    cv::Mat rot_mat = cv::getRotationMatrix2D(center, angle, 1);
    cv::warpAffine(src, dst, rot_mat, dst_sz, cv::INTER_LINEAR, cv::BORDER_CONSTANT);
    return dst;
}

cv::Mat yolo_detector::RotateImage(cv::Mat &src, float angle, int center_x, int center_y) {
    cv::Mat dst;
    cv::Size dst_sz(src.cols, src.rows);
    cv::Point2f center(center_x, center_y);
    cv::Mat rot_mat = cv::getRotationMatrix2D(center, angle, 1);
    cv::warpAffine(src, dst, rot_mat, dst_sz, cv::INTER_CUBIC, cv::BORDER_REPLICATE);
    return dst;
}

cv::Mat yolo_detector::make_upright(cv::Mat &src) {
    cv::Mat dst;
    cv::Canny(src, dst, canny_threshold_min, canny_threshold_max);
    std::vector<cv::Vec2f> lines;
    cv::HoughLines(dst, lines, 1, CV_PI / 180, houghline_threshold);
    if (lines.empty()) {
        src.copyTo(dst);
        return dst;
    }
    float theta = lines[0][1];
    if (theta <= CV_PI / 4) dst = RotateImage(src, theta / CV_PI * 180);
    if (theta >= 3 * CV_PI / 4) dst = RotateImage(src, theta / CV_PI * 180 - 180);
    if (theta > CV_PI / 4 && theta <= CV_PI / 2) dst = RotateImage(src, theta / CV_PI * 180 - 90);
    if (theta < 3 * CV_PI / 4 && theta > CV_PI / 2) dst = RotateImage(src, theta / CV_PI * 180 - 90);
    return dst;
}

cv::Mat yolo_detector::detect_number_area(cv::Mat &src) {
    cv::Mat hsv;
    cv::Mat mask;
    cv::Mat dilate;
    cv::Mat open;
    cv::Mat empty;
    cv::cvtColor(src, hsv, CV_BGR2HSV);
    cv::inRange(hsv, cv::Scalar(0, 0, 200), cv::Scalar(180, 38, 255), mask);
    cv::Mat kernel_dilate = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(dilate_kernel_size, dilate_kernel_size));
    cv::dilate(mask, dilate, kernel_dilate);
    cv::Mat kernel_open = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(open_kernel_size, open_kernel_size));
    morphologyEx(dilate, open, cv::MORPH_OPEN, kernel_open);
    cv::Rect rect = cv::boundingRect(open);
    int x_center = rect.x + rect.width / 2;
    int y_center = rect.y + rect.height / 2;
    if (((rect.width < detect_number_area_width_height_ratio * src.cols) ||
         (rect.height < detect_number_area_width_height_ratio * src.rows)) &&
        ((x_center < detect_number_area_center_ratio * src.cols) ||
         (x_center > (1 - detect_number_area_center_ratio) * src.cols) ||
         (y_center > (1 - detect_number_area_center_ratio) * src.rows) ||
         (y_center < detect_number_area_center_ratio * src.rows)))
        return empty;
    return src(rect);
}

cv::Mat yolo_detector::sharpen(cv::Mat &src) {
    cv::Mat kernel(3, 3, CV_32F, cv::Scalar(0));
    kernel.at<float>(1, 1) = 5.0;
    kernel.at<float>(0, 1) = -1.0;
    kernel.at<float>(2, 1) = -1.0;
    kernel.at<float>(1, 0) = -1.0;
    kernel.at<float>(1, 2) = -1.0;
    cv::Mat result;
    cv::filter2D(src, result, src.depth(), kernel);
    return result;
}

int yolo_detector::maxnum1(int current_prediction, float weight, int track_id) {
    track_id_perdictions_digit1_weight[track_id - 1][current_prediction] += weight;
    return std::max_element(track_id_perdictions_digit1_weight[track_id - 1],
                            track_id_perdictions_digit1_weight[track_id - 1] + 10) -
           track_id_perdictions_digit1_weight[track_id - 1];
}

int yolo_detector::maxnum2(int current_prediction, float weight, int track_id) {
    track_id_perdictions_digit2_weight[track_id - 1][current_prediction] += weight;
    return std::max_element(track_id_perdictions_digit2_weight[track_id - 1],
                            track_id_perdictions_digit2_weight[track_id - 1] + 10) -
           track_id_perdictions_digit2_weight[track_id - 1];
}


yolo_detector::yolo_detector() : cap2prepare(true), detect2top_prepare(true), top_prepare2top_detect(true),
                                 top_detect2cut(true), cut2infer(true), infer2show(true), fps_cap_counter(0),
                                 fps_det_counter(0), current_fps_cap(0), current_fps_det(0), exit_flag(false) {
#ifndef CAMERA
    std::string filename = "/home/chz/CLionProjects/darknet_yolo/video/test_real51.mp4";
    cap.open(filename);
#endif
#ifdef CAMERA
    do{
        cap.open(0);
    }while(!cap.isOpened());
    cap.set(CV_CAP_PROP_FRAME_WIDTH,1920);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT,1080);
    cap.set(CV_CAP_PROP_FPS, 30);
    cap.set(CV_CAP_PROP_FOURCC, CV_FOURCC('M','J','P','G'));
#endif
    cv::Mat cur_frame;
    cap >> cur_frame;
    frame_size = cur_frame.size();
#ifdef SAVE
    std::string name = ".avi";
    int index = 1;
    std::string tmp = std::to_string(index);
    tmp.append(name);
    std::ifstream f(tmp);
    while (f.good()) {
        index += 1;
        tmp = std::to_string(index);
        tmp.append(name);
        f = std::ifstream(tmp);
    }
    vw.open(tmp, cv::VideoWriter::fourcc('X', '2', '6', '4'), 30, cv::Size(1920, 1080));
#endif
    //libtorch model
    module = torch::jit::load("/home/chz/CLionProjects/darknet_yolo/weights/mobilenet.pt");
    module->to(torch::kCUDA);
    //tkDNN model
    detNN = &yolo;
    detNN->init("/home/chz/CLionProjects/darknet_yolo/weights/tkdarknet_fp16.rt", 2, 1);
    detNN->setthreshold(0.6);

    detNN_top = &yolo_top;
    detNN_top->init("/home/chz/CLionProjects/darknet_yolo/weights/tkdarknet_fp32_top.rt", 2, 3);
    detNN->setthreshold(0.01);
    //darknet yolo model
    detector_middle = new Detector("/home/chz/CLionProjects/darknet_yolo/cfg/yolov3-tiny-obj_96_2classes.cfg",
                                   "/home/chz/CLionProjects/darknet_yolo/weights/yolov3-tiny-obj_96_2classes_best.weights",
                                   0);
}


int yolo_detector::start() {
    while (true) {
        try {
            //read image
            t_cap = std::thread([&]() {
                detection_data_t detection_data;
                do {
                    detection_data = detection_data_t();
                    cap >> detection_data.cap_frame;
#ifdef FPS
                    fps_cap_counter++;
#endif
                    if (detection_data.cap_frame.empty() || exit_flag) {
                        std::cout << " exit_flag: detection_data.cap_frame.size = " << detection_data.cap_frame.size()
                                  << std::endl;
                        detection_data.exit_flag = true;
                        detection_data.cap_frame = cv::Mat(frame_size, CV_8UC3);
                    }
                    cap2prepare.send(detection_data);
                } while (!detection_data.exit_flag);
                std::cout << " t_cap exit \n";
            });
            //detect target: id,x,y,w,h
            t_detect = std::thread([&]() {
                detection_data_t detection_data;
                do {
#ifdef FPS
                    fps_det_counter++;
#endif
                    detection_data = cap2prepare.receive();
                    std::vector<cv::Mat> batch_dnn_input;
#ifdef SHOW_DETECTED
                    std::vector<cv::Mat> batch_frames;
                    batch_frames.push_back(detection_data.cap_frame.clone());
#endif
                    batch_dnn_input.push_back(detection_data.cap_frame.clone());
                    detNN->update(batch_dnn_input, 1);
                    std::vector<tk::dnn::box> detected_bbox;
                    detected_bbox = detNN->detected;
#ifdef SHOW_DETECTED
                    detNN->draw(batch_frames);
                    cv::imshow("test", batch_frames[0]);
                    cv::waitKey(1);
#endif
#ifdef SAVE
                    vw.write(batch_frames[0]);
#endif
                    std::vector<position> p_vec;
                    for (auto d:detected_bbox) {
                        position p;

                        p.x = d.x;
                        p.y = d.y;
                        p.w = d.w;
                        p.h = d.h;;
                        p.obj_id = d.cl;
                        p_vec.push_back(p);
                    }
                    detection_data.p_vec = p_vec;
                    detect2top_prepare.send(detection_data);
                } while (!detection_data.exit_flag);
                std::cout << " t_detect exit \n";
            });
            //track using kalman filter, initialize the number detection list if newly detected, cut target and surrounding area out
            t_top_prepare = std::thread([&]() {
                detection_data_t detection_data;
                cv::Mat cap_frame;
                std::vector<position> p_vec;
                do {
                    detection_data = detect2top_prepare.receive();
                    cap_frame = detection_data.cap_frame;
                    p_vec = detection_data.p_vec;
                    p_vec = track_kalman.correct(p_vec);
                    detection_data.p_vec = p_vec;
                    for (auto &p : p_vec) {
                        infer_data_t infer_data;
                        int x = p.x;
                        int y = p.y;
                        int w = p.w;
                        int h = p.h;
                        position infer_data_p;
                        infer_data_p.track_id = p.track_id;
                        if (p.track_id > max_tack_id) {
                            float *track_id_perdiction_digit1_weight = new float[10]{0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
                            track_id_perdictions_digit1_weight.push_back(track_id_perdiction_digit1_weight);
                            float *track_id_perdiction_digit2_weight = new float[10]{0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
                            track_id_perdictions_digit2_weight.push_back(track_id_perdiction_digit2_weight);
                            std::vector<int> track_id_perdiction_all_digit;
                            track_id_perdictions_all_digits.push_back(track_id_perdiction_all_digit);
                            max_tack_id = p.track_id;
                        }
                        infer_data.target_frame = cap_frame(cv::Rect(
                                std::min(std::max(x + w / 2 - std::max(96, w / 2), 0), cap_frame.cols - 1),
                                std::min(std::max(y + h / 2 - std::max(96, h / 2), 0), cap_frame.rows - 1),
                                std::min(std::min(std::max(x + w / 2 - std::max(96, w / 2), 0),
                                                  cap_frame.cols - 1) + std::max(192, w),
                                         cap_frame.cols - 1) -
                                std::min(std::max(x + w / 2 - std::max(96, w / 2), 0), cap_frame.cols - 1),
                                std::min(std::min(std::max(y + h / 2 - std::max(96, h / 2), 0),
                                                  cap_frame.rows - 1) + std::max(192, h),
                                         cap_frame.rows - 1) -
                                std::min(std::max(y + h / 2 - std::max(96, h / 2), 0),
                                         cap_frame.rows - 1)));
                        if (infer_data.target_frame.empty()) continue;
                        infer_data_p.x = x - std::min(std::max(x + w / 2 - std::max(96, w / 2), 0), cap_frame.cols - 1);
                        infer_data_p.y = y - std::min(std::max(y + h / 2 - std::max(96, h / 2), 0), cap_frame.rows - 1);
                        infer_data_p.w = std::min(x + w, cap_frame.cols - 1) - x;
                        infer_data_p.h = std::min(y + h, cap_frame.rows - 1) - y;
                        if (infer_data_p.h <= 10 || infer_data_p.w <= 10)continue;
                        infer_data.p = infer_data_p;
                        detection_data.infer_data.push_back(infer_data);
                    }
                    top_prepare2top_detect.send(detection_data);
                } while (!detection_data.exit_flag);
                std::cout << " t_prepare exit \n";
            });
            //detect top area of a target, support multi-target batch(max_size=3) detection
            t_top_detect = std::thread([&]() {
                detection_data_t detection_data;
                do {
                    detection_data = top_prepare2top_detect.receive();
                    std::vector<cv::Mat> batch_dnn_input;
                    for (auto &infer_data:detection_data.infer_data)
                        batch_dnn_input.push_back(infer_data.target_frame.clone());
                    detNN_top->update(batch_dnn_input, batch_dnn_input.size());
                    std::vector<std::vector<tk::dnn::box>> detected_bboxes;
                    detected_bboxes = detNN_top->batchDetected;
                    int i = 0;
                    for (auto &detected_bbox:detected_bboxes) {
                        infer_data_t *infer_data = &detection_data.infer_data[i];
                        if (!detected_bbox.empty()) {
                            infer_data->has_top_box = true;
                            position p_top;
                            p_top.x = detected_bbox[0].x;
                            p_top.y = detected_bbox[0].y;
                            p_top.w = detected_bbox[0].w;
                            p_top.h = detected_bbox[0].h;
                            infer_data->p_top = p_top;
                        }
                        i += 1;
                    }
                    top_detect2cut.send(detection_data);
                } while (!detection_data.exit_flag);
                std::cout << " t_detect exit \n";
            });
            //rotate the target into erect position, detect and cut out area which contains number
            t_cut = std::thread([&]() {
                detection_data_t detection_data;
                int x, y, w, h;
                int top_x, top_y, top_w, top_h;
                int x_center, y_center, top_x_center, top_y_center;
                int vector_x, vector_y;
                int angle;
                cv::Mat temp_frame;
                do {
                    detection_data = top_detect2cut.receive();
                    for (auto &infer_data : detection_data.infer_data) {
                        if (!infer_data.has_top_box) continue;
                        x = infer_data.p.x;
                        y = infer_data.p.y;
                        w = infer_data.p.w;
                        h = infer_data.p.h;
                        x_center = x + w / 2;
                        y_center = y + h / 2;
                        top_x = infer_data.p_top.x;
                        top_y = infer_data.p_top.y;
                        top_w = infer_data.p_top.w;
                        top_h = infer_data.p_top.h;
                        top_x_center = top_x + top_w / 2;
                        top_y_center = top_y + top_h / 2;
                        temp_frame = infer_data.target_frame;
                        vector_x = top_x_center - x_center;
                        vector_y = y_center - top_y_center;
                        if (vector_x == 0)angle = 90;
                        else angle = atan(abs(vector_y / vector_x)) / CV_PI * 180;
                        if (vector_x >= 0) {
                            if (vector_y >= 0)angle = 90 - angle;
                            else angle = 90 + angle;
                        } else {
                            if (vector_y >= 0)angle = angle - 90;
                            else angle = -angle - 90;
                        }
                        cv::Mat dst = RotateImage(temp_frame, angle, x_center, y_center);
                        dst = dst(cv::Rect(std::max(x_center - 48, 0), std::max(y_center - 48, 0),
                                           std::min(std::max(x_center - 48, 0) + 96, dst.cols) -
                                           std::max(x_center - 48, 0),
                                           std::min(std::max(y_center - 48, 0) + 96, dst.rows) -
                                           std::max(y_center - 48, 0)));
                        std::vector<bbox_t> result_vec_middle = detector_middle->detect(dst);
                        if (!result_vec_middle.empty()) {
                            int middle_x = result_vec_middle[0].x;
                            int middle_y = result_vec_middle[0].y;
                            int middle_w = result_vec_middle[0].w;
                            int middle_h = result_vec_middle[0].h;
                            dst = dst(cv::Rect(std::max(middle_x - center_detect_tolerant_pixels, 1),
                                               std::max(middle_y - center_detect_tolerant_pixels, 1),
                                               std::min(middle_x + middle_w + center_detect_tolerant_pixels,
                                                        dst.cols - 1) - middle_x,
                                               std::min(middle_y + middle_h + center_detect_tolerant_pixels,
                                                        dst.rows - 1) - middle_y));
                        } else continue;
                        if (dst.empty())continue;
                        dst = make_upright(dst);
                        dst = detect_number_area(dst);
                        if (dst.empty())continue;
                        dst = sharpen(dst);
                        infer_data.cut_frame = dst;
                    }
                    cut2infer.send(detection_data);
                } while (!detection_data.exit_flag);
                std::cout << " t_cut exit \n";
            });
            //infer the numbers and correct them using detection history
            //if any of the previous procedures fail, 0 will be presented as result
            t_infer = std::thread([&]() {
                detection_data_t detection_data;
                cv::Mat cut_frame;
                int track_id;
                do {
                    detection_data = cut2infer.receive();
                    for (auto &infer_data : detection_data.infer_data) {
                        int digit1, digit2;
                        cut_frame = infer_data.cut_frame;
                        track_id = infer_data.p.track_id;
                        if (!cut_frame.empty()) {
                            infer_out_t infer_result = infer(cut_frame);
                            digit1 = infer_result.digit1_prediction;
                            digit2 = infer_result.digit2_prediction;
                            if (digit1 * 10 + digit2 >= 10) {
                                digit1 = maxnum1(digit1, infer_result.digit1_weight, track_id);
                                digit2 = maxnum2(digit2, infer_result.digit2_weight, track_id);
                                if (digit1 * 10 + digit2 >= 10)infer_data.prediction = digit1 * 10 + digit2;
                            }
                        }
                        printf("track_id: %d", track_id);
                        printf("digits_pred: %d\n", infer_data.prediction);
                    }
#ifdef FPS
                    infer2show.send(detection_data);
#endif
                } while (!detection_data.exit_flag);
                std::cout << " t_infer exit \n";
            });
#ifdef FPS
            t_show = std::thread([&]() {
                detection_data_t detection_data;
                do {
                    detection_data = infer2show.receive();
                    steady_end = std::chrono::steady_clock::now();
                    float time_sec = std::chrono::duration<double>(steady_end - steady_start).count();
                    if (time_sec >= 1) {
                        current_fps_det = fps_det_counter.load() / time_sec;
                        current_fps_cap = fps_cap_counter.load() / time_sec;
                        steady_start = steady_end;
                        fps_det_counter = 0;
                        fps_cap_counter = 0;
                    }
                    std::cout << " current_fps_det = " << current_fps_det << ", current_fps_cap = " << current_fps_cap
                              << std::endl;
                } while (!detection_data.exit_flag);
                std::cout << " t_show exit \n";
            });
#endif
            if (t_cap.joinable()) t_cap.join();
            if (t_detect.joinable()) t_detect.join();
            if (t_top_prepare.joinable()) t_top_prepare.join();
            if (t_top_detect.joinable()) t_top_detect.join();
            if (t_cut.joinable()) t_cut.join();
            if (t_infer.joinable()) t_infer.join();
#ifdef FPS
            if (t_show.joinable()) t_show.join();
#endif
            break;
        }
        catch (...) {
            continue;
        }
    }
    return 0;
}

int main() {
    yolo_detector y;
    y.start();
    return 0;
}
