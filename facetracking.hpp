//
// Created by markson zhang on 2019-03-20.
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
#include <iostream>
#include <array>
#include <vector>
#include <opencv2/opencv.hpp>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/packed_func.h>
#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/packed_func.h>
#include <fstream>
#include <iterator>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "RetinaFace/anchor_generator.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "RetinaFace/config.h"
#include "RetinaFace/tools.h"
#include "RetinaFace/ulsMatF.h"

using namespace std;
using namespace cv;

// Tunable Parameters
const int avg_face = 1;
const int minSize = 60;
const int stage = 4;
const int input_width = 640;
const int input_height = 480;
const cv::Size frame_size = Size(input_width, input_height);
const float ratio_x = input_width / 640.;
const float ratio_y = input_height / 480.;
const string prefix = "/Users/marksonzhang/Project/Face-Recognition-Cpp/models/macos";
const char arcface_model[30] = "y1-arcface-emore_115";

struct _FaceInfo {
    /**
     * Structure _FaceInfo
     * face_count: the count of total face
     * face_details: the [confidence, x, y, w, h, eyes, nose, cheek] coordinators
     */
    int face_count;
    std::vector<std::array<double, 15>> face_details;
//    double face_details[][15];
};

struct RetinaOutput {
    std::vector<Anchor> result;
    cv::Point2f ratio;
};


class MTCNN;

/**
 * Class of TVM model implementation, it contains the model definition module and the inference function.
 * the inference function is the forward
 */
class FR_MFN_Deploy {
private:
    std::unique_ptr<tvm::runtime::Module> handle;

public:
    FR_MFN_Deploy(std::string modelFolder) {
        tvm::runtime::Module mod_syslib = tvm::runtime::Module::LoadFromFile(
                modelFolder + format("/deploy_lib_%s.so", arcface_model));
        //load graph
        std::ifstream json_in(modelFolder + format("/deploy_graph_%s.json", arcface_model));
        std::string json_data((std::istreambuf_iterator<char>(json_in)), std::istreambuf_iterator<char>());
        json_in.close();
        int device_type = kDLCPU;
        int device_id = 0;
        // get global function module for graph runtime
        tvm::runtime::Module mod = (*tvm::runtime::Registry::Get("tvm.graph_runtime.create"))(json_data, mod_syslib,
                                                                                              device_type, device_id);
        this->handle.reset(new tvm::runtime::Module(mod));
        //load param
        std::ifstream params_in(modelFolder + format("/deploy_param_%s.params", arcface_model), std::ios::binary);
        std::string params_data((std::istreambuf_iterator<char>(params_in)), std::istreambuf_iterator<char>());
        params_in.close();
        TVMByteArray params_arr;
        params_arr.data = params_data.c_str();
        params_arr.size = params_data.length();
        tvm::runtime::PackedFunc load_params = mod.GetFunction("load_params");
        load_params(params_arr);
    }


    cv::Mat forward(cv::Mat inputImageAligned) {
        //mobilefacnet preprocess has been written in graph.
        cv::Mat tensor = cv::dnn::blobFromImage(inputImageAligned, 1.0, cv::Size(112, 112), cv::Scalar(0, 0, 0), true);
        //convert uint8 to float32 and convert to RGB via opencv dnn function
        DLTensor *input;
        constexpr int dtype_code = kDLFloat;
        constexpr int dtype_bits = 32;
        constexpr int dtype_lanes = 1;
        constexpr int device_type = kDLCPU;
        constexpr int device_id = 0;
        constexpr int in_ndim = 4;
        const int64_t in_shape[in_ndim] = {1, 3, 112, 112};
        TVMArrayAlloc(in_shape, in_ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &input);//
        TVMArrayCopyFromBytes(input, tensor.data, 112 * 3 * 112 * 4);
        tvm::runtime::Module *mod = (tvm::runtime::Module *) handle.get();
        tvm::runtime::PackedFunc set_input = mod->GetFunction("set_input");
        set_input("data", input);
        tvm::runtime::PackedFunc run = mod->GetFunction("run");
        run();
        tvm::runtime::PackedFunc get_output = mod->GetFunction("get_output");
        tvm::runtime::NDArray res = get_output(0);
        cv::Mat vector(128, 1, CV_32F);
        memcpy(vector.data, res->data, 128 * 4);
        cv::Mat _l2;
        cv::multiply(vector, vector, _l2);
        float l2 = cv::sqrt(cv::sum(_l2).val[0]);
        vector = vector / l2;
        TVMArrayFree(input);
        return vector;
    }

};

class RetinaFaceDeploy {
private:
    std::unique_ptr<tvm::runtime::Module> handle;

public:
    RetinaFaceDeploy(std::string modelFolder) {
        // tvm module for compiled functions
        tvm::runtime::Module mod_syslib = tvm::runtime::Module::LoadFromFile(modelFolder + "/mnet.25.x86.cpu.so");
        // json graph
        std::ifstream json_in(modelFolder + "/mnet.25.x86.cpu.json", std::ios::in);
        std::string json_data((std::istreambuf_iterator<char>(json_in)), std::istreambuf_iterator<char>());
        json_in.close();

        int dtype_code = kDLFloat;
        int dtype_bits = 32;

        int dtype_lanes = 1;
        int device_type = kDLCPU;//kDLGPU
        int device_id = 0;
        // get global function module for graph runtime
        tvm::runtime::Module mod = (*tvm::runtime::Registry::Get("tvm.graph_runtime.create"))(json_data, mod_syslib,
                                                                                              device_type, device_id);
        this->handle.reset(new tvm::runtime::Module(mod));
        // parameters in binary
        std::ifstream params_in(modelFolder + "/mnet.25.x86.cpu.params", std::ios::binary);
        std::string params_data((std::istreambuf_iterator<char>(params_in)), std::istreambuf_iterator<char>());
        params_in.close();
        // parameters need to be TVMByteArray type to indicate the binary data
        TVMByteArray params_arr;
        params_arr.data = params_data.c_str();
        params_arr.size = params_data.length();
        tvm::runtime::PackedFunc load_params = mod.GetFunction("load_params");
        load_params(params_arr);
    }

    RetinaOutput forward(cv::Mat image) {
        constexpr int dtype_code = kDLFloat;
        constexpr int dtype_bits = 32;

        constexpr int dtype_lanes = 1;
        constexpr int device_type = kDLCPU;//kDLGPU
        constexpr int device_id = 0;
        DLTensor *x;
        int in_ndim = 4;
        int in_c = 3, in_h = 480, in_w = 640;
        int64_t in_shape[4] = {1, in_c, in_h, in_w};
        TVMArrayAlloc(in_shape, in_ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &x);

        int64_t w1 = ceil(in_w / 32.0), w2 = ceil(in_w / 16.0), w3 = ceil(in_w / 8.0), h1 = ceil(
                in_h / 32.0), h2 = ceil(in_h / 16.0), h3 = ceil(in_h / 8.0);
        int out_num = (w1 * h1 + w2 * h2 + w3 * h3) * (4 + 8 + 20);

        tvm::runtime::Module *mod = (tvm::runtime::Module *) handle.get();

        int total_input = 3 * in_w * in_h;
        float *data_x = (float *) malloc(total_input * sizeof(float));

        //float* y_iter = (float*)malloc(out_num*4);

        if (!image.data)
            printf("load error");

        //input data
        cv::Mat resizeImage;
        cv::resize(image, resizeImage, cv::Size(in_w, in_h), cv::INTER_AREA);
        cv::Mat input_mat;

        resizeImage.convertTo(input_mat, CV_32FC3);
        //cv::cvtColor(input_mat, input_mat, cv::COLOR_BGR2RGB);
        cv::Mat split_mat[3];
        cv::split(input_mat, split_mat);
        memcpy(data_x, split_mat[2].ptr<float>(), input_mat.cols * input_mat.rows * sizeof(float));
        memcpy(data_x + input_mat.cols * input_mat.rows, split_mat[1].ptr<float>(),
               input_mat.cols * input_mat.rows * sizeof(float));
        memcpy(data_x + input_mat.cols * input_mat.rows * 2, split_mat[0].ptr<float>(),
               input_mat.cols * input_mat.rows * sizeof(float));
        TVMArrayCopyFromBytes(x, data_x, total_input * sizeof(float));

        // get the function from the module(set input data)
        tvm::runtime::PackedFunc set_input = mod->GetFunction("set_input");
        set_input("data", x);
        // get the function from the module(run it)
        tvm::runtime::PackedFunc run = mod->GetFunction("run");
        run();
        tvm::runtime::PackedFunc get_output = mod->GetFunction("get_output");
        std::vector<AnchorGenerator> ac(_feat_stride_fpn.size());
        for (int i = 0; i < _feat_stride_fpn.size(); ++i) {
            int stride = _feat_stride_fpn[i];
            ac[i].Init(stride, anchor_cfg[stride], false);
        }
        std::vector<Anchor> proposals;
        proposals.clear();

        int64_t w[3] = {w1, w2, w3};
        int64_t h[3] = {h1, h2, h3};
        int64_t out_size[9] = {w1 * h1 * 4, w1 * h1 * 8, w1 * h1 * 20, w2 * h2 * 4, w2 * h2 * 8, w2 * h2 * 20,
                               w3 * h3 * 4, w3 * h3 * 8, w3 * h3 * 20};

        int out_ndim = 4;
        int64_t out_shape[9][4] = {{1, 4,  h1, w1},
                                   {1, 8,  h1, w1},
                                   {1, 20, h1, w1},
                                   {1, 4,  h2, w2},
                                   {1, 8,  h2, w2},
                                   {1, 20, h2, w2},
                                   {1, 4,  h3, w3},
                                   {1, 8,  h3, w3},
                                   {1, 20, h3, w3}};
        DLTensor *y[9];
        for (int i = 0; i < 9; i++)
            TVMArrayAlloc(out_shape[i], out_ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &y[i]);
        for (int i = 0; i < 9; i += 3) {
            get_output(i, y[i]);
            get_output(i + 1, y[i + 1]);
            get_output(i + 2, y[i + 2]);

            ulsMatF clsMat(w[i / 3], h[i / 3], 4);
            ulsMatF regMat(w[i / 3], h[i / 3], 8);
            ulsMatF ptsMat(w[i / 3], h[i / 3], 20);


            TVMArrayCopyToBytes(y[i], clsMat.m_data, out_size[i] * sizeof(float));
            TVMArrayCopyToBytes(y[i + 1], regMat.m_data, out_size[i + 1] * sizeof(float));
            TVMArrayCopyToBytes(y[i + 2], ptsMat.m_data, out_size[i + 2] * sizeof(float));


            ac[i / 3].FilterAnchor(clsMat, regMat, ptsMat, proposals);
//            std::cout << "proposals:" << proposals.size() << std::endl;

        }

        // nms
        std::vector<Anchor> result;
        nms_cpu(proposals, nms_threshold, result);
//        printf("final proposals: %ld\n", result.size());

        // free buffer
        free(data_x);
        data_x = nullptr;
        TVMArrayFree(x);
        for (int i = 0; i < 9; i++)
            TVMArrayFree(y[i]);

        RetinaOutput output_;
        output_.result = result;
        output_.ratio.x = ratio_x;
        output_.ratio.y = ratio_y;
        return output_;
    }
};

int MTCNNTracking(MTCNN &detector, FR_MFN_Deploy &deploy);
int RetinaFaceTracking(RetinaFaceDeploy &deploy_track, FR_MFN_Deploy &deploy_rec);
int RetinaFace(RetinaFaceDeploy &deploy_track);
int MTCNNDetection(MTCNN &detector);
int InferenceOnce(RetinaFaceDeploy &deploy_track, FR_MFN_Deploy &deploy_rec);
