#include <iostream>
#include <stdio.h>
#include <opencv2/highgui.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/core/core.hpp>
#include "FacePreprocess.h"
#include "MTCNN/mtcnn_opencv.hpp"
#include <numeric>
#include <math.h>
#include "facetracking.hpp"
#include <time.h>


#define PI 3.14159265
using namespace std;
using namespace cv;

double sum_score, sum_fps;

/**
 * This is a normalize function before calculating the cosine distance. Experiment has proven it can destory the
 * original distribution in order to make two feature more distinguishable.
 */
Mat Zscore(const Mat &fc) {
    Mat mean, std;
    meanStdDev(fc, mean, std);
//    cout << mean << std << endl;
    Mat fc_norm = (fc - mean) / std;
    return fc_norm;

}

/**
 * This module is using to computing the cosine distance between input feature and ground truth feature
 */
inline float CosineDistance(const cv::Mat &v1, const cv::Mat &v2) {
    double dot = v1.dot(v2);
    double denom_v1 = norm(v1);
    double denom_v2 = norm(v2);

    return dot / (denom_v1 * denom_v2);

}

inline double count_angle(float landmark[5][2]) {
    double a = landmark[2][1] - (landmark[0][1] + landmark[1][1]) / 2;
    double b = landmark[2][0] - (landmark[0][0] + landmark[1][0]) / 2;
    double angle = atan(abs(b) / a) * 180 / PI;
    return angle;
}

inline float count_padding(float xmin, float ymin, float xmax, float ymax, cv::Mat frame) {
    cv::Size frame_s = frame.size();
    float w_border = frame_s.width;
    float h_border = frame_s.height;

    float xm2border = w_border - xmax;
    float ym2border = h_border - ymax;

    return min({xmin, ymin, xm2border, ym2border});
}

/**
 * Formatting output structure
 */
inline cv::Mat draw_conclucion(String intro, double input, cv::Mat result_cnn, int position) {
    char string[10];
    sprintf(string, "%.2f", input);
    std::string introString(intro);
    introString += string;
    putText(result_cnn, introString, cv::Point(5, position), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255));
    return result_cnn;
}

/**
 * Face Recognition pipeline using camera. Firstly, it will use MTCNN face detector to detect the faces [x,y,w,h] and [eyes, nose, cheeks] landmarks
 * Then, face alignment will be implemented for wraping the face into decided center point as possible as we can. Finally, the aligned
 * face will be sent into TVM-mobilefacenet-arcface model and output the feature of aligned face which will be compared with the ground
 * truth face we have set in advanced. The similarity score will be output at the imshow windows.
 * -------
 * Args:
 *      &detector: Address of loaded MTCNN model
 *      &deploy: Address of loaed TVM model
 */
int MTCNNTracking(MTCNN &detector, FR_MFN_Deploy &deploy) {

    //OpenCV Version
    cout << "OpenCV Version: " << CV_MAJOR_VERSION << "."
         << CV_MINOR_VERSION << "."
         << CV_SUBMINOR_VERSION << endl;

    clock_t start, end;

    //TVM
    Mat faces, face_avg;
    vector<Mat> face_list;
    for (int i = 1; i <= avg_face; i++) {
        faces = imread("/Users/marksonzhang/Project/Face-Recognition-Cpp/" + format("img/zzw_%d.jpg", i));
//        GaussianBlur(faces,faces,Size( 3, 3 ), 0, 0);
//        sharpen(faces,faces);
        resize(faces, faces, Size(112, 112), 0, 0, INTER_LINEAR);
        face_list.push_back(faces);
    }
    for (int i = 1; i < face_list.size(); i++) {
        face_list[0] += face_list[i];
        face_list[0] /= 2;
    }
    face_avg = face_list[0];

    if (0) {
        imshow("face average", face_avg);
    }
    Mat fc1 = deploy.forward(face_avg);

    fc1 = Zscore(fc1);
    int count = 0;
    // MTCNN Parameters
    float factor = 0.709f;
    float threshold[3] = {0.7f, 0.6f, 0.6f};

    VideoCapture cap(0); //using camera capturing
    if (!cap.isOpened()) {
        cerr << "nothing" << endl;
        return -1;
    }

    double fps, current;
    char string[10];
    char buff[10];
    Mat frame;

    // gt face landmark
    float v1[5][2] = {
            {30.2946f, 51.6963f},
            {65.5318f, 51.5014f},
            {48.0252f, 71.7366f},
            {33.5493f, 92.3655f},
            {62.7299f, 92.2041f}};

    cv::Mat src(5, 2, CV_32FC1, v1);

    memcpy(src.data, v1, 2 * 5 * sizeof(float));

    double score, angle, padding;
    vector<double> angle_list;
    while (count < 1000) {
        count++;
        double t = (double) cv::getTickCount();
        cap >> frame;

        cerr << "height  " << frame.rows << "  width  " << frame.cols << endl;
        cerr << "height  " << frame.size().height << "  width  " << frame.size().width << endl;
        cv::imshow("origin", frame);
        //TODO: input image down here!!!!!!
//        frame = cv::imread(format("/Users/marksonzhang/Downloads/fail_imgs/k%d.png", count - 1));
        resize(frame, frame, frame_size, 0.5, 0.5, INTER_LINEAR);
        Mat result_cnn = frame.clone();
        cerr << "result_cnn height  " << result_cnn.rows << " result_cnn width  " << result_cnn.cols << endl;
        cerr << "result_cnn height  " << result_cnn.size().height << " result_cnn width  " << result_cnn.size().width
             << endl;
        vector<FaceInfo> faceInfo = detector.Detect_mtcnn(frame, minSize, threshold, factor, stage);
        for (int i = 0; i < faceInfo.size(); i++) {
            cout << faceInfo[i].bbox.score << endl;
        }
        for (int i = 0; i < faceInfo.size(); i++) {
            int x = (int) faceInfo[i].bbox.xmin;
            int y = (int) faceInfo[i].bbox.ymin;
            int w = (int) (faceInfo[i].bbox.xmax - faceInfo[i].bbox.xmin + 1);
            int h = (int) (faceInfo[i].bbox.ymax - faceInfo[i].bbox.ymin + 1);
            cv::rectangle(result_cnn, cv::Rect(x, y, w, h), cv::Scalar(0, 0, 255), 2);

            // compute padding
            padding = count_padding(faceInfo[i].bbox.xmin, faceInfo[i].bbox.ymin, faceInfo[i].bbox.xmax,
                                    faceInfo[i].bbox.ymax, frame);

            // Perspective Transformation
            float v2[5][2] =
                    {{faceInfo[i].landmark[0], faceInfo[i].landmark[1]},
                     {faceInfo[i].landmark[2], faceInfo[i].landmark[3]},
                     {faceInfo[i].landmark[4], faceInfo[i].landmark[5]},
                     {faceInfo[i].landmark[6], faceInfo[i].landmark[7]},
                     {faceInfo[i].landmark[8], faceInfo[i].landmark[9]},
                    };

            // compute angle
            angle = count_angle(v2);
            angle_list.push_back(angle);
            cout << "INFO：Angle  " << angle << endl;

            cv::Mat dst(5, 2, CV_32FC1, v2);
            memcpy(dst.data, v2, 2 * 5 * sizeof(float));

            cv::Mat m = FacePreprocess::similarTransform(dst, src);
            cv::Mat aligned = frame.clone();
            cv::warpPerspective(frame, aligned, m, cv::Size(96, 112), INTER_LINEAR);
            resize(aligned, aligned, Size(112, 112), 0, 0, INTER_LINEAR);
            if (0) {
                imshow("aligned face", aligned);
            }

            // TODO: remember to set it to 1 before using this pipeline. This will get the ground truth image of your face for further calculating.
            if (0) {
                imwrite("/Users/marksonzhang/Project/Face-Recognition-Cpp/" + format("img/zzw_%d.jpg", count), aligned);
                imshow("crop face", aligned);
                waitKey(0);
            }

            start = clock();
            Mat fc2 = deploy.forward(aligned);
            end = clock();
            cerr << "inference cost: " << (double) (end - start) / CLOCKS_PER_SEC << endl;

            // normalize
            fc2 = Zscore(fc2);
            current = CosineDistance(fc1, fc2);

//            cerr << "Inference score: " << current << endl;
            sum_score += current;

            sprintf(string, "%.4f", current);
            cv::putText(result_cnn, string, Point(x, y), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(255, 255, 0));

            for (int j = 0; j < 10; j += 2) {
                if (j == 0 or j == 6) {
                    cv::circle(result_cnn, Point(faceInfo[i].landmark[j], faceInfo[i].landmark[j + 1]), 3,
                               Scalar(0, 255, 0),
                               FILLED, LINE_AA);
                } else {
                    cv::circle(result_cnn, Point(faceInfo[i].landmark[j], faceInfo[i].landmark[j + 1]), 3,
                               Scalar(0, 0, 255),
                               FILLED, LINE_AA);
                }
            }
            score = faceInfo[i].bbox.score;
        }
//        cerr << score << endl;
        t = ((double) cv::getTickCount() - t) / cv::getTickFrequency();
        fps = 1.0 / t;
        sum_fps += fps;

        result_cnn = draw_conclucion("FPS: ", fps, result_cnn, 20);
        result_cnn = draw_conclucion("Frame: ", count, result_cnn, 35);
        result_cnn = draw_conclucion("Avg Face: ", avg_face, result_cnn, 65);
        result_cnn = draw_conclucion("Angle: ", angle, result_cnn, 80);

        cv::imshow("image", result_cnn);
        cv::waitKey(1);
    }
    cout << "average fps: " << sum_fps / (float) count << endl;
    cout << "average score: " << sum_score / (float) count << endl;

}

/**
 * Face Recognition pipeline using camera. Instead, the model is using RetinaFace-TVM, others remain the same
 * as MTCNNTracking
 * -------
 * Args:
 *      &deploy_track: Address of loaded RetinaFace-TVM model
 *      &deploy_rec: Address of loaded MobileFaceNet-TVM model
 */
int RetinaFaceTracking(RetinaFaceDeploy &deploy_track, FR_MFN_Deploy &deploy_rec) {
    //OpenCV Version
    cout << "OpenCV Version: " << CV_MAJOR_VERSION << "."
         << CV_MINOR_VERSION << "."
         << CV_SUBMINOR_VERSION << endl;

    clock_t start, end;
    //TVM
    Mat faces, face_avg;
    vector<Mat> face_list;
    if (1) {
        for (int i = 1; i <= avg_face; i++) {
            faces = imread("/Users/marksonzhang/Project/Face-Recognition-Cpp/" + format("img/zzw_%d_retina.jpg", i));
            resize(faces, faces, Size(112, 112), 0, 0, INTER_LINEAR);
            face_list.push_back(faces);
        }
    } else {
        faces = imread("/Users/marksonzhang/Project/Face-Recognition-Cpp/img/fr_retina.jpg");
        resize(faces, faces, Size(112, 112), 0, 0, INTER_LINEAR);
        face_list.push_back(faces);
    }
    for (int i = 1; i < face_list.size(); i++) {
        face_list[0] += face_list[i];
        face_list[0] /= 2;
    }
    face_avg = face_list[0];

    if (0) {
        imshow("face average", face_avg);
    }
    Mat fc1 = deploy_rec.forward(face_avg);

    fc1 = Zscore(fc1);
    int count = 0;
    VideoCapture cap(0); //using camera capturing
    if (!cap.isOpened()) {
        cerr << "nothing" << endl;
        return -1;
    }
    int frame_width = cap.get(CAP_PROP_FRAME_WIDTH);
    int frame_height = cap.get(CAP_PROP_FRAME_HEIGHT);

//    VideoWriter video;
//    video.open("/Users/marksonzhang/Movies/outcpp.avi", cv::VideoWriter::fourcc('M','J','P','G'), cap.get(cv::CAP_PROP_FPS), Size(frame_width, frame_height), true);

    double fps, current;
    char string[10];
    char buff[10];
    Mat frame;


    // gt face landmark
    float v1[5][2] = {
            {30.2946f, 51.6963f},
            {65.5318f, 51.5014f},
            {48.0252f, 71.7366f},
            {33.5493f, 92.3655f},
            {62.7299f, 92.2041f}};

    cv::Mat src(5, 2, CV_32FC1, v1);

    memcpy(src.data, v1, 2 * 5 * sizeof(float));

    double score, angle;
    while (count <= 50000) {
        count++;
        double t = (double) cv::getTickCount();
        cap >> frame;
        resize(frame, frame, frame_size, 0.5, 0.5, INTER_LINEAR);
        Mat result_cnn = frame.clone();
        RetinaOutput output_ = deploy_track.forward(frame);
        vector<Anchor> faceInfo = output_.result;
        float ratio_x = output_.ratio.x;
        float ratio_y = output_.ratio.y;

        for (int i = 0; i < faceInfo.size(); i++) {
            int x = (int) faceInfo[i].finalbox.x * ratio_x;
            int y = (int) faceInfo[i].finalbox.y * ratio_y;
            int w = (int) faceInfo[i].finalbox.width * ratio_x;
            int h = (int) faceInfo[i].finalbox.height * ratio_y;
            cv::rectangle(result_cnn, Point(x, y), Point(w, h), cv::Scalar(0, 0, 255), 2);
            cv::circle(result_cnn, Point(x, y), 3, Scalar(255, 255, 0), FILLED, LINE_AA);
            cv::circle(result_cnn, Point(w, h), 3, Scalar(255, 255, 0), FILLED, LINE_AA);
            // Perspective Transformation
            float v2[5][2] =
                    {{faceInfo[i].pts[0].x * ratio_x, faceInfo[i].pts[0].y * ratio_y},
                     {faceInfo[i].pts[1].x * ratio_x, faceInfo[i].pts[1].y * ratio_y},
                     {faceInfo[i].pts[2].x * ratio_x, faceInfo[i].pts[2].y * ratio_y},
                     {faceInfo[i].pts[3].x * ratio_x, faceInfo[i].pts[3].y * ratio_y},
                     {faceInfo[i].pts[4].x * ratio_x, faceInfo[i].pts[4].y * ratio_y},
                    };

            // compute angle
            angle = count_angle(v2);

            cv::Mat dst(5, 2, CV_32FC1, v2);
            memcpy(dst.data, v2, 2 * 5 * sizeof(float));

            cv::Mat m = FacePreprocess::similarTransform(dst, src);
            cv::Mat aligned = frame.clone();
            cv::warpPerspective(frame, aligned, m, cv::Size(96, 112), INTER_LINEAR);
            resize(aligned, aligned, Size(112, 112), 0, 0, INTER_LINEAR);
            if (0) {
                imshow("aligned face", aligned);
            }

            // TODO: remember to set it to 1 before using this pipeline. This will get the ground truth image of your face for further calculating.
            if (0) {
                imwrite("/Users/marksonzhang/Project/Face-Recognition-Cpp/" + format("img/fr_%d_retina.jpg", count),
                        aligned);
                imshow("crop face", aligned);
                waitKey(0);
            }

            start = clock();
            Mat fc2 = deploy_rec.forward(aligned);
            end = clock();
//            cerr << "inference cost: " << (double) (end - start) / CLOCKS_PER_SEC << endl;

            // normalize
            fc2 = Zscore(fc2);
            current = CosineDistance(fc1, fc2);

//            cerr << "Inference score: " << current << endl;
            sum_score += current;

            sprintf(string, "%.4f", current);
            cv::putText(result_cnn, string, Point(x, y), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(255, 255, 0));

            for (int j = 0; j < faceInfo[i].pts.size(); ++j) {
                if (j == 0 or j == 3) {
                    cv::circle(result_cnn, Point(faceInfo[i].pts[j].x * ratio_x, faceInfo[i].pts[j].y * ratio_y), 3,
                               Scalar(0, 255, 0),
                               FILLED, LINE_AA);
                } else {
                    cv::circle(result_cnn, Point(faceInfo[i].pts[j].x * ratio_x, faceInfo[i].pts[j].y * ratio_y), 3,
                               Scalar(0, 0, 255),
                               FILLED, LINE_AA);
                }
            }
        }
        t = ((double) cv::getTickCount() - t) / cv::getTickFrequency();
        fps = 1.0 / t;
        sum_fps += fps;

        result_cnn = draw_conclucion("FPS: ", fps, result_cnn, 20);
        result_cnn = draw_conclucion("Frame: ", count, result_cnn, 35);
        result_cnn = draw_conclucion("Avg Face: ", avg_face, result_cnn, 50);
        result_cnn = draw_conclucion("Angle: ", angle, result_cnn, 65);

//        video << result_cnn;
        cv::imshow("image", result_cnn);
        cv::waitKey(1);

        cout << sum_score / (float) count << endl;
    }


    cout << "average fps: " << sum_fps / (float) count << endl;
    cout << "average score: " << sum_score / (float) count << endl;

    cap.release();
//    video.release();
    destroyAllWindows();
    return 0;
}

/**
 * Face Detection pipeline using camera. Instead, the model is using RetinaFace-TVM.
 * -------
 * Args:
 *      &deploy_track: Address of loaded RetinaFace-TVM model
 */
int RetinaFace(RetinaFaceDeploy &deploy_track) {
    //OpenCV Version
    cout << "OpenCV Version: " << CV_MAJOR_VERSION << "."
         << CV_MINOR_VERSION << "."
         << CV_SUBMINOR_VERSION << endl;

    clock_t start, end;
    //TVM
    Mat faces, face_avg;
    vector<Mat> face_list;


    int count = 0;
    VideoCapture cap(0); //using camera capturing
    if (!cap.isOpened()) {
        cerr << "nothing" << endl;
        return -1;
    }
    double fps, current;
    char string[10];
    char buff[10];
    Mat frame;

    double score, angle;
    while (count <= 50000) {
        count++;
        double t = (double) cv::getTickCount();
        cap >> frame;
        resize(frame, frame, frame_size, 0.5, 0.5, INTER_LINEAR);
        Mat result_cnn = frame.clone();
        RetinaOutput output_ = deploy_track.forward(frame);
        vector<Anchor> faceInfo = output_.result;
        float ratio_x = output_.ratio.x;
        float ratio_y = output_.ratio.y;
        cout << ratio_x << "  " << ratio_y << endl;
        for (int i = 0; i < faceInfo.size(); i++) {
            int x = (int) faceInfo[i].finalbox.x * ratio_x;
            int y = (int) faceInfo[i].finalbox.y * ratio_y;
            int w = (int) faceInfo[i].finalbox.width * ratio_x;
            int h = (int) faceInfo[i].finalbox.height * ratio_y;
            cv::rectangle(result_cnn, Point(x, y), Point(w, h), cv::Scalar(0, 0, 255), 2);
            cv::circle(result_cnn, Point(x, y), 3, Scalar(255, 255, 0), FILLED, LINE_AA);
            cv::circle(result_cnn, Point(w, h), 3, Scalar(255, 255, 0), FILLED, LINE_AA);
            // Perspective Transformation
            float v2[5][2] =
                    {{faceInfo[i].pts[0].x * ratio_x, faceInfo[i].pts[0].y * ratio_y},
                     {faceInfo[i].pts[1].x * ratio_x, faceInfo[i].pts[1].y * ratio_y},
                     {faceInfo[i].pts[2].x * ratio_x, faceInfo[i].pts[2].y * ratio_y},
                     {faceInfo[i].pts[3].x * ratio_x, faceInfo[i].pts[3].y * ratio_y},
                     {faceInfo[i].pts[4].x * ratio_x, faceInfo[i].pts[4].y * ratio_y},
                    };

            // compute angle
            angle = count_angle(v2);
//            cerr << "inference cost: " << (double) (end - start) / CLOCKS_PER_SEC << endl;
//            cerr << "Inference score: " << current << endl;
            sum_score += current;

            for (int j = 0; j < faceInfo[i].pts.size(); ++j) {
                if (j == 0 or j == 3) {
                    cv::circle(result_cnn, Point(faceInfo[i].pts[j].x * ratio_x, faceInfo[i].pts[j].y * ratio_y), 3,
                               Scalar(0, 255, 0),
                               FILLED, LINE_AA);
                } else {
                    cv::circle(result_cnn, Point(faceInfo[i].pts[j].x * ratio_x, faceInfo[i].pts[j].y * ratio_y), 3,
                               Scalar(0, 0, 255),
                               FILLED, LINE_AA);
                }
            }
        }
        t = ((double) cv::getTickCount() - t) / cv::getTickFrequency();
        fps = 1.0 / t;
        sum_fps += fps;

        result_cnn = draw_conclucion("FPS: ", fps, result_cnn, 20);
        result_cnn = draw_conclucion("Frame: ", count, result_cnn, 35);
        result_cnn = draw_conclucion("Angle: ", angle, result_cnn, 50);

        cv::imshow("image", result_cnn);
        cv::waitKey(1);
    }

    return 0;
}

/**
 * Face Recognition pipeline using camera. Firstly, it will use MTCNN face detector to detect the faces [x,y,w,h] and [eyes, nose, cheeks] landmarks
 * Then, face alignment will be implemented for wraping the face into decided center point as possible as we can. Finally, the aligned
 * face will be sent into TVM-mobilefacenet-arcface model and output the feature of aligned face which will be compared with the ground
 * truth face we have set in advanced. The similarity score will be output at the imshow windows.
 * -------
 * Args:
 *      &detector: Address of loaded MTCNN model
 *      &deploy: Address of loaed TVM model
 */
int MTCNNDetection(MTCNN &detector) {

    //OpenCV Version
    cout << "OpenCV Version: " << CV_MAJOR_VERSION << "."
         << CV_MINOR_VERSION << "."
         << CV_SUBMINOR_VERSION << endl;

    clock_t start, end;

    int count = 0;
    // MTCNN Parameters
    float factor = 0.709f;
    float threshold[3] = {0.7f, 0.6f, 0.6f};

    VideoCapture cap(0); //using camera capturing
    if (!cap.isOpened()) {
        cerr << "nothing" << endl;
        return -1;
    }

    double fps, current;
    char string[10];
    char buff[10];
    Mat frame;

    double score, angle, padding;
    vector<double> angle_list;
    while (count < 1000) {
        count++;
        double t = (double) cv::getTickCount();
        cap >> frame;

        cerr << "height  " << frame.rows << "  width  " << frame.cols << endl;
        cerr << "height  " << frame.size().height << "  width  " << frame.size().width << endl;
        cv::imshow("origin", frame);
        //TODO: input image down here!!!!!!
//        frame = cv::imread(format("/Users/marksonzhang/Downloads/fail_imgs/k%d.png", count - 1));
        resize(frame, frame, frame_size, 0.5, 0.5, INTER_LINEAR);
        Mat result_cnn = frame.clone();
        cerr << "result_cnn height  " << result_cnn.rows << " result_cnn width  " << result_cnn.cols << endl;
        cerr << "result_cnn height  " << result_cnn.size().height << " result_cnn width  " << result_cnn.size().width
             << endl;
        vector<FaceInfo> faceInfo = detector.Detect_mtcnn(frame, minSize, threshold, factor, stage);
        for (int i = 0; i < faceInfo.size(); i++) {
            cout << faceInfo[i].bbox.score << endl;
        }
        for (int i = 0; i < faceInfo.size(); i++) {
            int x = (int) faceInfo[i].bbox.xmin;
            int y = (int) faceInfo[i].bbox.ymin;
            int w = (int) (faceInfo[i].bbox.xmax - faceInfo[i].bbox.xmin + 1);
            int h = (int) (faceInfo[i].bbox.ymax - faceInfo[i].bbox.ymin + 1);
            cv::rectangle(result_cnn, cv::Rect(x, y, w, h), cv::Scalar(0, 0, 255), 2);

            // compute padding
            padding = count_padding(faceInfo[i].bbox.xmin, faceInfo[i].bbox.ymin, faceInfo[i].bbox.xmax,
                                    faceInfo[i].bbox.ymax, frame);

            // Perspective Transformation
            float v2[5][2] =
                    {{faceInfo[i].landmark[0], faceInfo[i].landmark[1]},
                     {faceInfo[i].landmark[2], faceInfo[i].landmark[3]},
                     {faceInfo[i].landmark[4], faceInfo[i].landmark[5]},
                     {faceInfo[i].landmark[6], faceInfo[i].landmark[7]},
                     {faceInfo[i].landmark[8], faceInfo[i].landmark[9]},
                    };
            current = faceInfo[i].bbox.score;
            // compute angle
            angle = count_angle(v2);
            angle_list.push_back(angle);
            cout << "INFO：Angle  " << angle << endl;

            cv::Mat dst(5, 2, CV_32FC1, v2);
            memcpy(dst.data, v2, 2 * 5 * sizeof(float));

//            cerr << "Inference score: " << current << endl;
            sum_score += current;

            for (int j = 0; j < 10; j += 2) {
                if (j == 0 or j == 6) {
                    cv::circle(result_cnn, Point(faceInfo[i].landmark[j], faceInfo[i].landmark[j + 1]), 3,
                               Scalar(0, 255, 0),
                               FILLED, LINE_AA);
                } else {
                    cv::circle(result_cnn, Point(faceInfo[i].landmark[j], faceInfo[i].landmark[j + 1]), 3,
                               Scalar(0, 0, 255),
                               FILLED, LINE_AA);
                }
            }
            score = faceInfo[i].bbox.score;
        }
//        cerr << score << endl;
        t = ((double) cv::getTickCount() - t) / cv::getTickFrequency();
        fps = 1.0 / t;
        sum_fps += fps;

        result_cnn = draw_conclucion("FPS: ", fps, result_cnn, 20);
        result_cnn = draw_conclucion("Frame: ", count, result_cnn, 35);
        result_cnn = draw_conclucion("Confidence: ", current, result_cnn, 50);
        result_cnn = draw_conclucion("Angle: ", angle, result_cnn, 65);

        cv::imshow("image", result_cnn);
        cv::waitKey(1);
    }
    cout << "average fps: " << sum_fps / (float) count << endl;
    cout << "average score: " << sum_score / (float) count << endl;

}

/**
 * Face Recognition pipeline using camera. Instead, the model is using RetinaFace-TVM, others remain the same
 * as MTCNNTracking
 * -------
 * Args:
 *      &deploy_track: Address of loaded RetinaFace-TVM model
 *      &deploy_rec: Address of loaded MobileFaceNet-TVM model
 */
int InferenceOnce(RetinaFaceDeploy &deploy_track, FR_MFN_Deploy &deploy_rec) {

    String image_t, image_p;
    cin >> image_t;
    cin >> image_p;

    clock_t start, end;
    //TVM
    Mat faces, face_avg;
    vector<Mat> face_list;

    faces = cv::imread(image_t);
    resize(faces, faces, Size(112, 112), 0, 0, INTER_LINEAR);
    face_avg = faces;

    if (0) {
        imshow("face average", face_avg);
    }
    Mat fc1 = deploy_rec.forward(face_avg);

    fc1 = Zscore(fc1);

    double current;
    char string[10];
    char buff[10];
    Mat frame;

    // gt face landmark
    float v1[5][2] = {
            {30.2946f, 51.6963f},
            {65.5318f, 51.5014f},
            {48.0252f, 71.7366f},
            {33.5493f, 92.3655f},
            {62.7299f, 92.2041f}};

    cv::Mat src(5, 2, CV_32FC1, v1);

    memcpy(src.data, v1, 2 * 5 * sizeof(float));

    double score, angle;

    frame = imread(image_p);
    resize(frame, frame, frame_size, 0.5, 0.5, INTER_LINEAR);
    Mat result_cnn = frame.clone();
    RetinaOutput output_ = deploy_track.forward(frame);
    vector<Anchor> faceInfo = output_.result;
    float ratio_x = output_.ratio.x;
    float ratio_y = output_.ratio.y;

    for (int i = 0; i < faceInfo.size(); i++) {
        int x = (int) faceInfo[i].finalbox.x * ratio_x;
        int y = (int) faceInfo[i].finalbox.y * ratio_y;
        int w = (int) faceInfo[i].finalbox.width * ratio_x;
        int h = (int) faceInfo[i].finalbox.height * ratio_y;
        cv::rectangle(result_cnn, Point(x, y), Point(w, h), cv::Scalar(0, 0, 255), 2);
        cv::circle(result_cnn, Point(x, y), 3, Scalar(255, 255, 0), FILLED, LINE_AA);
        cv::circle(result_cnn, Point(w, h), 3, Scalar(255, 255, 0), FILLED, LINE_AA);
        // Perspective Transformation
        float v2[5][2] =
                {{faceInfo[i].pts[0].x * ratio_x, faceInfo[i].pts[0].y * ratio_y},
                 {faceInfo[i].pts[1].x * ratio_x, faceInfo[i].pts[1].y * ratio_y},
                 {faceInfo[i].pts[2].x * ratio_x, faceInfo[i].pts[2].y * ratio_y},
                 {faceInfo[i].pts[3].x * ratio_x, faceInfo[i].pts[3].y * ratio_y},
                 {faceInfo[i].pts[4].x * ratio_x, faceInfo[i].pts[4].y * ratio_y},
                };

        // compute angle
        angle = count_angle(v2);

        cv::Mat dst(5, 2, CV_32FC1, v2);
        memcpy(dst.data, v2, 2 * 5 * sizeof(float));

        cv::Mat m = FacePreprocess::similarTransform(dst, src);
        cv::Mat aligned = frame.clone();
        cv::warpPerspective(frame, aligned, m, cv::Size(96, 112), INTER_LINEAR);
        resize(aligned, aligned, Size(112, 112), 0, 0, INTER_LINEAR);

        Mat fc2 = deploy_rec.forward(aligned);

        // normalize
        fc2 = Zscore(fc2);
        current = CosineDistance(fc1, fc2);
        cout << current << endl;
    }
}