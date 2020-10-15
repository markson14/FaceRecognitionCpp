# Face-Recognition-Cpp

### Introduction

- Detect:
  - [Optional] [Fast-MTCNN](https://github.com/imistyrain/MTCNN/tree/master/Fast-MTCNN)
  - [Default] [RetinaFace-TVM](https://github.com/Howave/RetinaFace-TVM)
- Verification: MobileFaceNet + Arcface

This project is using **Fast-MTCNN** for face detection and **TVM inference model** for face recognition. At the face detection stage, the the module will output the `x,y,w,h` coordinations as well as `5` facial landmarks for further alignment. At the face recognition stage, the `112x112` image crop by the first stage output will be the second stage input. The output will be an `1x128` feature vector for cosine similarity measuring. The recognition pipeline can run 50FPS on CPU **(2.8 GHz Quad-Core Intel Core i7)**.

![output](assets/demo.gif)

### PerformanceÏ

|      Backbone      |  Size   |     FPS     | Average Cosine Simi |
| :----------------: | :-----: | :---------: | :-----------------: |
|     **MTCNN**      | 640x480 | **31.7331** |      80.8787%       |
| **RetinaFace-TVM** | 640x480 |   20.9007   |    **87.8968%**     |

### Dependency:

- OpenCV >= 3.4.1
- TVM

### Set up:

- **OpenCV**

```shell
# macos
brew install opencv
brew link opencv
brew install pkg-config
pkg-config --cflags --libs /[path-to-your-opencv]/lib/pkgconfig/opencv.pc

# ========================================================================
#linux
# Install minimal prerequisites (Ubuntu 18.04 as reference)
sudo apt update && sudo apt install -y cmake g++ wget unzip
# Download and unpack sources
wget -O opencv.zip https://github.com/opencv/opencv/archive/master.zip
wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/master.zip
unzip opencv.zip
unzip opencv_contrib.zip
# Create build directory and switch into it
mkdir -p build && cd build
# Configure
cmake -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib-master/modules ../opencv-master
# Build
cmake --build .
```

- **[TVM](https://docs.tvm.ai/install/from_source.html#python-package-installation)**

```shell
git clone --recursive https://github.com/dmlc/tvm

mkdir build
cp cmake/config.cmake build
cd build
cmake ..
make -j4
```

- **tvm_complier**

  Now you are able create your own .so file by using pretrained MXNet models on your own environment. Here I am using mobilefacenet-arcface model as face recognition backbone.

- **CMakeList.txt**

  `modify the TVM path into your own `

- **Prefix:** set the prefix model path to your own.

- **Recording ground truth:**`mkdir img` and set record to `1` to record ground truth image for face recognition.

---

### Run:

Run the project may activate your camera to capture images.

```shell
mkdir build
cd build
cmake ..
make -j4
./FaceRecognitionCpp
```

---

### **Tun-able Parameters：**

1. **minSize:** set the minimum size of faces for MTCNN detector. Larger size can ensure quick inference time.
2. **factor:** set the step factor for pyramid of image.  Larger factor will get fewer images after doing pyramid.
3. **Frame size:** set the camera or streaming capturing frame size.
4. **Stage:** set how many stage for MTCNN to implement.
5. **Average Faces:** default 1 

---

### Reference

```markdown
@inproceedings{imistyrain2018MTCNN,
title={Fast-MTCNN https://github.com/imistyrain/MTCNN/tree/master/Fast-MTCNN},
author={Jack Yu},
}

@inproceedings{RetinaFace-TVM,
title={RetinaFace-TVM https://github.com/Howave/RetinaFace-TVM},
author={Howave},
}

@inproceedings{deng2019retinaface,
title={RetinaFace: Single-stage Dense Face Localisation in the Wild},
author={Deng, Jiankang and Guo, Jia and Yuxiang, Zhou and Jinke Yu and Irene Kotsia and Zafeiriou, Stefanos},
booktitle={arxiv},
year={2019}
}

@inproceedings{guo2018stacked,
  title={Stacked Dense U-Nets with Dual Transformers for Robust Face Alignment},
  author={Guo, Jia and Deng, Jiankang and Xue, Niannan and Zafeiriou, Stefanos},
  booktitle={BMVC},
  year={2018}
}

@article{deng2018menpo,
  title={The Menpo benchmark for multi-pose 2D and 3D facial landmark localisation and tracking},
  author={Deng, Jiankang and Roussos, Anastasios and Chrysos, Grigorios and Ververas, Evangelos and Kotsia, Irene and Shen, Jie and Zafeiriou, Stefanos},
  journal={IJCV},
  year={2018}
}

@inproceedings{deng2018arcface,
title={ArcFace: Additive Angular Margin Loss for Deep Face Recognition},
author={Deng, Jiankang and Guo, Jia and Niannan, Xue and Zafeiriou, Stefanos},
booktitle={CVPR},
year={2019}
}
```

