cmake_minimum_required(VERSION 3.5.0)

project(project VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(EIGEN3_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/external/eigen)
set(EIGENRAND_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/external/EigenRand)
include_directories(${EIGEN3_INCLUDE_DIR} ${EIGENRAND_INCLUDE_DIR})

set(SOURCES
    ActivationFunction.cpp
    Adam.cpp
    DataLoader.cpp
    Except.cpp
    LinearLayer.cpp
    LossFunction.cpp
    main.cpp
    NeuralNetwork.cpp
    NonLinearLayer.cpp
    Random.cpp
    SGD.cpp
    Test.cpp
    Train.cpp
    )

add_executable(CourseProject ${SOURCES})
