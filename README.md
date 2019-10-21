# Face-Recognition-Cpp

### Introduction

- Detect: Fast-MTCNN
- Verification: MobileFaceNet + Arcface

This project is using Fast-MTCNN for face detection and TVM inference model for face recognition. At the face detection stage, the the module will output the `x,y,w,h` coordinations as well as `5` facial landmarks for further alignment. At the face recognition stage, the `112x112` image crop by the first stage output will be the second stage input. The output will be an `1x128` feature vector for cosine similarity measuring. 

### Set up:

- **Require OpenCV**

```shell
brew install opencv
brew link opencv
brew install pkg-config
pkg-config --cflags --libs /usr/local/Cellar/opencv/<version_number>/lib/pkgconfig/opencv.pc
```

- **Require [TVM](https://docs.tvm.ai/install/from_source.html#python-package-installation)**

```shell
git clone --recursive https://github.com/dmlc/tvm

mkdir build
cp cmake/config.cmake build
cd build
cmake ..
make -j4
```

- **About the .os file**

Now Linux users can replace the model files with those inside folder `model/linux`

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

### Output Structure:

```c++
struct _FaceInfo {
    int face_count;
    double face_details[][15];
} faceinfo;
```

Score port: 0

Bbox port: 1~5

Landmark port: 6~14

---

### **Tun-able Parameters：**

1. **minSize:** set the minimum size of faces for MTCNN detector. Larger size can ensure quick inference time.
2. **factor:** set the step factor for pyramid of image.  Larger factor will get fewer images after doing pyramid.
3. **Frame size:** set the camera or streaming capturing frame size.
4. **Stage:** set how many stage for MTCNN to implement.
5. **Average Faces:** default 1 



### #TODO

- Quantize arcface model

---

**Citation:**

```markdown
@inproceedings{imistyrain2018MTCNN,
title={Fast-MTCNN https://github.com/imistyrain/MTCNN/tree/master/Fast-MTCNN},
author={Jack Yu},
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

