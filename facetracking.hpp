//
// Created by markson zhang on 2019-07-17.
//
#include <iostream>
#include <stdio.h>
#include <array>
#include <vector>
#include <opencv2/opencv.hpp>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/packed_func.h>
#include "stdlib.h"


using namespace std;
using namespace cv;

// Tunable Parameters
const int avg_face = 1;
const int minSize = 60;
const int stage = 4;
const float factor = 0.709f;
//const cv::Size frame_size = Size(1280,760);
const cv::Size frame_size = Size(320,240);
const string prefix = "/Users/marksonzhang/WorkSpace/Face_Tracking/face-tracking/";
const char arcface_model[30] = "y1-arcface-emore_109";

const extern class MTCNN;

struct _FaceInfo{
    int face_count;
    std::vector<std::array<double, 15>> face_details;
//    double face_details[][15];
};

class FR_MFN_Deploy{
private:
    void * handle;

public:
    FR_MFN_Deploy(std::string modelFolder)
    {
        tvm::runtime::Module mod_syslib = tvm::runtime::Module::LoadFromFile(modelFolder + format("/deploy_lib_%s.so", arcface_model));
        //load graph
        std::ifstream json_in(modelFolder + format("/deploy_graph_%s.json", arcface_model));
        std::string json_data((std::istreambuf_iterator<char>(json_in)), std::istreambuf_iterator<char>());
        json_in.close();
        int device_type = kDLCPU;
        int device_id = 0;
        // get global function module for graph runtime
        tvm::runtime::Module mod = (*tvm::runtime::Registry::Get("tvm.graph_runtime.create"))(json_data, mod_syslib, device_type, device_id);
        this->handle = new tvm::runtime::Module(mod);
        //load param
        std::ifstream params_in(modelFolder + format("/deploy_param_%s.params",arcface_model), std::ios::binary);
        std::string params_data((std::istreambuf_iterator<char>(params_in)), std::istreambuf_iterator<char>());
        params_in.close();
        TVMByteArray params_arr;
        params_arr.data = params_data.c_str();
        params_arr.size = params_data.length();
        tvm::runtime::PackedFunc load_params = mod.GetFunction("load_params");
        load_params(params_arr);
    }


    cv::Mat forward(cv::Mat inputImageAligned)
    {
        //mobilefacnet preprocess has been written in graph.
        cv::Mat tensor = cv::dnn::blobFromImage(inputImageAligned,1.0,cv::Size(112,112),cv::Scalar(0,0,0),true);
        //convert uint8 to float32 and convert to RGB via opencv dnn function
        DLTensor* input;
        constexpr int dtype_code = kDLFloat;
        constexpr int dtype_bits = 32;
        constexpr int dtype_lanes = 1;
        constexpr int device_type = kDLCPU;
        constexpr int device_id = 0;
        constexpr int in_ndim = 4;
        const int64_t in_shape[in_ndim] = {1, 3, 112, 112};
        TVMArrayAlloc(in_shape, in_ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &input);//
        TVMArrayCopyFromBytes(input,tensor.data,112*3*112*4);
        tvm::runtime::Module* mod = (tvm::runtime::Module*)handle;
        tvm::runtime::PackedFunc set_input = mod->GetFunction("set_input");
        set_input("data", input);
        tvm::runtime::PackedFunc run = mod->GetFunction("run");
        run();
        tvm::runtime::PackedFunc get_output = mod->GetFunction("get_output");
        tvm::runtime::NDArray res = get_output(0);
        cv::Mat vector(128,1,CV_32F);
        memcpy(vector.data,res->data,128*4);
        cv::Mat _l2;
        cv::multiply(vector,vector,_l2);
        float l2 =  cv::sqrt(cv::sum(_l2).val[0]);
        vector = vector / l2;
        TVMArrayFree(input);
        return vector;
    }

};

int MTCNNTracking(MTCNN &detector, FR_MFN_Deploy &deploy);
_FaceInfo face_detecting(MTCNN *detector);
int Tester(int argc, char **argv);