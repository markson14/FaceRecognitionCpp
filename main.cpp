//
// Created by markson zhang on 2019-07-17.
//
#include "facetracking.hpp"
#include <stdio.h>
#include <iostream>
#include <mtcnn_opencv.hpp>

using namespace std;
using namespace cv;

int main(){
    //load model
    FR_MFN_Deploy deploy(prefix+"models");
    MTCNN detector("/Users/marksonzhang/WorkSpace/Face_Tracking/face-tracking/models");

    //execute function
    if(1){
        struct _FaceInfo faces;
//        cout << "xxx" << sizeof(faces) << endl;
        faces = face_detecting(&detector);
        cout << "count: " << faces.face_count << endl;
        cout << faces.face_details[0][0] << endl;
        for(int i=0;i<faces.face_count;i++){
            printf("score: %.3f  face_rect=[%.1f, %.1f, %.1f, %.1f] \n", faces.face_details[i][0],
                   faces.face_details[i][1], faces.face_details[i][2], faces.face_details[i][3],
                   faces.face_details[i][4]);
            cout << "left eye landmark: " << faces.face_details[i][5] << ", " << faces.face_details[i][6] << endl;
            cout << "right eye landmark: " << faces.face_details[i][7] << ", " << faces.face_details[i][8] << endl;
            cout << "nose landmark: " << faces.face_details[i][9] << ", " << faces.face_details[i][10] << endl;
            cout << "left cheek landmark: " << faces.face_details[i][11] << ", " << faces.face_details[i][12] << endl;
            cout << "right cheek landmark: " << faces.face_details[i][13] << ", " << faces.face_details[i][14]
                 << endl;
            cout << "==============================" << endl;
        }
    }
    else{
        MTCNNTracking(detector, deploy);
    }
}