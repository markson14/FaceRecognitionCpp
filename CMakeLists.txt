CMAKE_MINIMUM_REQUIRED(VERSION 3.6)
PROJECT(FaceRecognitionCpp)
FIND_PACKAGE(OpenCV 4 REQUIRED)
SET(CMAKE_CXX_STANDARD 11)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -ldl -lpthread")
SET(HOME_TVM /Users/marksonzhang/my_lib/tvm)
SET(BUILD_SHARED_LIBS OFF) # need to be turn off for demo

LINK_DIRECTORIES(${HOME_TVM}/build)

INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS}
        ${PROJECT_SOURCE_DIR}/include)
INCLUDE_DIRECTORIES(${HOME_TVM}/include)
INCLUDE_DIRECTORIES(${HOME_TVM}/3rdparty/dmlc-core/include)
INCLUDE_DIRECTORIES(${HOME_TVM}/3rdparty/dlpack/include)

ADD_LIBRARY(facedetect include/MTCNN/mtcnn_opencv.cpp
        include/FacePreprocess.h
        include/MTCNN/mtcnn_opencv.hpp
        include/RetinaFace/anchor_generator.cpp
        include/RetinaFace/anchor_generator.h
        include/RetinaFace/config.cpp
        include/RetinaFace/config.h
        include/RetinaFace/tools.cpp
        include/RetinaFace/tools.h
        include/RetinaFace/ulsMatF.h
        facetracking.cpp
        facetracking.hpp)

ADD_EXECUTABLE(FaceRecognitionCpp main.cpp)

TARGET_LINK_LIBRARIES(FaceRecognitionCpp ${OpenCV_LIBS} facedetect -ltvm_runtime)
