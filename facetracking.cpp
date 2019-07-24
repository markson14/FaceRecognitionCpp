#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "FacePreprocess.h"
#include "mtcnn_opencv.hpp"
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/packed_func.h>
#include <numeric>
#include "facetracking.hpp"

using namespace std;
using namespace cv;

#define USEMTCNN 1
#define gStreamer 1

Mat Zscore(const Mat &fc) {
    Mat mean, std;
    meanStdDev(fc, mean, std);
//    cout << mean << std << endl;
    Mat fc_norm = (fc - mean) / std;
    return fc_norm;

}


inline float CosineDistance(const cv::Mat &v1, const cv::Mat &v2) {

//    return static_cast<float>(v1.dot(v2));
    double dot = v1.dot(v2);
    double denom_v1 = norm(v1);
    double denom_v2 = norm(v2);

    return dot / (denom_v1 * denom_v2);

}

struct _FaceInfo writestruct(vector<FaceInfo> &faceInfo){
    struct _FaceInfo faces;
    faces.face_count = faceInfo.size();

    for (int i = 0; i < faceInfo.size(); i++) {
        int x = (int) faceInfo[i].bbox.xmin;
        int y = (int) faceInfo[i].bbox.ymin;
        int w = (int) (faceInfo[i].bbox.xmax - faceInfo[i].bbox.xmin + 1);
        int h = (int) (faceInfo[i].bbox.ymax - faceInfo[i].bbox.ymin + 1);

        std::array<double, 15> face_details;
        face_details[0] = faceInfo[i].bbox.score;
        face_details[1] = x;
        face_details[2] = y;
        face_details[3] = w;
        face_details[4] = h;

        int k = 0;
        for (int j = 5; j < 15; j++) {
            face_details[j] = faceInfo[i].landmark[k];
            k++;
        }
        faces.face_details.push_back(face_details);

    }
    return faces;
}

#if USEMTCNN

int MTCNNTracking(MTCNN &detector, FR_MFN_Deploy &deploy) {

    //OpenCV Version
    cout << "OpenCV Version: " << CV_MAJOR_VERSION << "."
         << CV_MINOR_VERSION << "."
         << CV_SUBMINOR_VERSION << endl;

    //TVM
    Mat faces, face_avg;
    vector<Mat> face_list;
    for (int i = 1; i <= avg_face; i++) {
        faces = imread(prefix + format("img/zzw_%d.jpg", i));
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

    imshow("face average", face_avg);
    Mat fc1 = deploy.forward(face_avg);

    fc1 = Zscore(fc1);
    int count = 0;
    float sum_score = 0;

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

    double score;
    while (cap.isOpened()) {
        count++;
        double t = (double) cv::getTickCount();
        cap >> frame;
//        medianBlur(frame,frame,3);
//        GaussianBlur(frame,frame,Size( 3, 3 ), 0, 0);
//        sharpen(frame,frame);
        resize(frame, frame, frame_size, 0, 0, INTER_LINEAR);
        Mat result_cnn = frame.clone();
        vector<FaceInfo> faceInfo = detector.Detect_mtcnn(frame, minSize, threshold, factor, stage);
        for (int i = 0; i < faceInfo.size(); i++) {
            int x = (int) faceInfo[i].bbox.xmin;
            int y = (int) faceInfo[i].bbox.ymin;
            int w = (int) (faceInfo[i].bbox.xmax - faceInfo[i].bbox.xmin + 1);
            int h = (int) (faceInfo[i].bbox.ymax - faceInfo[i].bbox.ymin + 1);
            cv::rectangle(result_cnn, cv::Rect(x, y, w, h), cv::Scalar(0, 0, 255), 2);

            // Perspective Transformation
            float v2[5][2] =
                    {{faceInfo[i].landmark[0], faceInfo[i].landmark[1]},
                     {faceInfo[i].landmark[2], faceInfo[i].landmark[3]},
                     {faceInfo[i].landmark[4], faceInfo[i].landmark[5]},
                     {faceInfo[i].landmark[6], faceInfo[i].landmark[7]},
                     {faceInfo[i].landmark[8], faceInfo[i].landmark[9]},
                    };
            cv::Mat dst(5, 2, CV_32FC1, v2);
            memcpy(dst.data, v2, 2 * 5 * sizeof(float));

            cv::Mat m = FacePreprocess::similarTransform(dst, src);
            cv::Mat aligned = frame.clone();
            cv::warpPerspective(frame, aligned, m, cv::Size(96, 112), INTER_LINEAR);
            resize(aligned, aligned, Size(112, 112), 0, 0, INTER_LINEAR);
            imshow("aligned face", aligned);
//            cvtColor(aligned,aligned,COLOR_BGR2RGB);

            //Set 1 to record faces
            if (0) {
                imwrite(prefix + format("img/zzw_%d.jpg", count), aligned);
                waitKey(0);
            }
            Mat fc2 = deploy.forward(aligned);

            // normalize
            fc2 = Zscore(fc2);
            current = CosineDistance(fc1, fc2);

            sum_score += current;

//
            cerr << "Inference score: " << current << endl;

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
        sprintf(string, "%.2f", fps);
        std::string fpsString("FPS: ");
        fpsString += string;
        putText(result_cnn, fpsString, cv::Point(5, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255));
        std::string framecount("Frame: ");
        framecount += std::to_string(count);
        putText(result_cnn, framecount, cv::Point(5, 35), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255));
        std::string confidence("Confidence: ");
        sprintf(buff, "%.2f", current);
        confidence += buff;
        putText(result_cnn, confidence, cv::Point(5, 50), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255));
        std::string avgface("Avg Face: ");
        avgface += to_string(avg_face);
        putText(result_cnn, avgface, cv::Point(5, 65), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255));
        cv::imshow("image", result_cnn);
        cv::waitKey(1);
    }
    return 1;
}

#endif

#if gStreamer

struct _FaceInfo face_detecting(MTCNN *detector) {
    VideoCapture cap(0); //using camera capturing
    if (!cap.isOpened()) {
        cerr << "nothing" << endl;
    }
    struct _FaceInfo faces;
    Mat frame;
    cv::Size frame_size = Size(640, 480);
//    cv::Size frame_size = Size(320,240);
    float threshold[3] = {0.7f, 0.6f, 0.6f};
    while (cap.isOpened()) {
        cap >> frame;
        resize(frame, frame, frame_size, 0, 0, INTER_LINEAR);
        cout << frame.type() << "\t" << frame.elemSize() << "\t" << frame.channels() <<"\t" << frame.depth() << endl;
        vector<FaceInfo> faceInfo = detector->Detect_mtcnn(frame, minSize, threshold, factor, 4);

        faces = writestruct(faceInfo);
//        for (int i = 0; i < 15; i++) {
//            cout << faces.face_details[0][i] << endl;
//        }
        return faces;
    }

    //release the buffer
}

#else
int Tester(int argc, char **argv) {
    //TVM
    FR_MFN_Deploy deploy(prefix + "models");
    Mat faces, face_avg;
    vector<Mat> face_list;
    for (int i = 1; i <= avg_face; i++) {
        faces = imread(prefix + format("img/zzw_%d.jpg", i));
        resize(faces, faces, Size(112, 112), 0, 0, INTER_LINEAR);
        face_list.push_back(faces);
    }
    for (int i = 1; i < face_list.size(); i++) {
        face_list[0] += face_list[i];
        face_list[0] /= 2;
    }
    face_avg = face_list[0];

    imshow("face average", face_avg);
    Mat fc1 = deploy.forward(face_avg);
    vector<float> result;
    fc1 = Zscore(fc1);

    vector<string> Testlist;
    Testlist = GetTestImg(prefix);
    cout << avg_face << "  " << arcface_model << endl;
    if (0) {
        for (int i = 0; i < Testlist.size(); i++) {
            Mat img = imread(Testlist[i]);
            Testing(img, fc1, result);
        }
    } else {
        VideoCapture cap(0); //using camera capturing
        if (!cap.isOpened()) {
            cerr << "nothing" << endl;
            return -1;
        }
        Mat frame;
        int count = 0;
        while (count < 50) {
            count++;
            cap >> frame;
            Testing(frame, fc1, result);
        }

        vector<double> re;
        re = mean_std(result);

        double mean = re[0];
        double stdev = re[1];

        cout << setprecision(3) << "mean: " << mean << "  std: " << stdev << endl;
    }
}
#endif