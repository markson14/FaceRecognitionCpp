# Face Tracking

#### ~~Version 1~~

- ~~Detect: dlibfacedetect~~
- ~~Verification: MobileNet0.5+Arcface~~

#### Version 2

- Detect: Fast-MTCNN
- Verification: MobileFaceNet + Arcface

**Set up:**

- **Require OpenCV**

```
brew install opencv
brew link opencv
brew install pkg-config
pkg-config --cflags --libs /usr/local/Cellar/opencv/<version_number>/lib/pkgconfig/opencv.pc
```

- **Require TVM**

```shell
git clone --recursive https://github.com/dmlc/tvm

mkdir build
cp cmake/config.cmake build
cd build
cmake ..
make -j4
```

- About the .os file

**In this project, the os file is complied in MacOS system. If you want to use in your own PC which is not MacOS, you have to recomplie the .json and .params files in your own computer by using TVM complier. The code will be submitted in wiki later.**

- **CMakeList.txt**

  `modify the TVM path into your own `

- **Prefix:** set the prefix path to your own

- **Recording ground truth:**`mkdir img` and set record to `1`

---

**Output Structure:**

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

**Tun-able Parameters：**

1. **minSize:** set the minimum size of faces for MTCNN detector. Larger size can ensure quick inference time.
2. **factor:** set the step factor for pyramid of image.  Larger factor will get fewer images after doing pyramid.
3. **Frame size:** set the camera or streaming capturing frame size.
4. **Stage:** set how many stage for MTCNN to implement.
5. **Average Faces:** default 1 

---

Demo:

![demo](./assets/Screen Shot 2019-07-11 at 4.46.44 pm.png)

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

