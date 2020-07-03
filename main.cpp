//
// Created by markson zhang on 2019-03-20.
//
#include "facetracking.hpp"
#include <stdio.h>
#include <iostream>
#include <MTCNN/mtcnn_opencv.hpp>

using namespace std;
using namespace cv;

int main() {
    //load model
    FR_MFN_Deploy deploy_rec(prefix);
    RetinaFaceDeploy deploy_track(prefix);
    MTCNN detector(prefix);

//    MTCNNTracking(detector, deploy_rec);
    RetinaFaceTracking(deploy_track, deploy_rec);
//    RetinaFace(deploy_track);
//    MTCNNDetection(detector);
//    InferenceOnce(deploy_track, deploy_rec);
}