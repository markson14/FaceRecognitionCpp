CMAKE_MINIMUM_REQUIRED(VERSION 3.6)
project(face-tracking)

FIND_PACKAGE(OpenCV 3.4.6 REQUIRED)

SET(CMAKE_CXX_STANDARD 11)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -ldl -lpthread")
SET(HOME_TVM /Users/marksonzhang/tvm)
#SET(BUILD_SHARED_LIBS ON)

INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS}
        ${PROJECT_SOURCE_DIR}/include)
INCLUDE_DIRECTORIES(${HOME_TVM}/include)
INCLUDE_DIRECTORIES(${HOME_TVM}/3rdparty/dmlc-core/include)
INCLUDE_DIRECTORIES(${HOME_TVM}/3rdparty/dlpack/include)

ADD_LIBRARY(facedetect  include/mtcnn_opencv.cpp
                        include/FacePreprocess.h
        		include/mtcnn_opencv.hpp
                        facetracking.cpp
                        facetracking.hpp)

ADD_EXECUTABLE(face-tracking tvm_runtime_pack.cc main.cpp)

TARGET_LINK_LIBRARIES(face-tracking ${OpenCV_LIBS} facedetect)
