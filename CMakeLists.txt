cmake_minimum_required(VERSION 3.16)
project(ex4_stavimn CXX)

set(CMAKE_CXX_STANDARD 14)
SET(CMAKE_C_FLAGS_DEBUG "-D_DEBUG")

include_directories(.)

add_executable(Main
        Activation.h
        Dense.h
        main.cpp
        Matrix.h
        MlpNetwork.h Matrix.cpp Activation.cpp Dense.cpp MlpNetwork.cpp)

add_executable(Test Activation.h
        Dense.h
        Matrix.h
        MlpNetwork.h Matrix.cpp Activation.cpp Dense.cpp MlpNetwork.cpp)


add_executable(Presubmit Activation.h
        Dense.h
        Matrix.h
        MlpNetwork.h Matrix.cpp Activation.cpp Dense.cpp MlpNetwork.cpp)