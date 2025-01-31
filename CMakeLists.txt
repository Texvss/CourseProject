# cmake_minimum_required(VERSION 3.5.0)
# project(project VERSION 0.1.0 LANGUAGES C CXX)


# set(CMAKE_CXX_STANDARD 11)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)

# find_package(Eigen3 3.3 REQUIRED NO_MODULE)
# include_directories(${EIGEN3_INCLUDE_DIR})
# add_executable(project main.cpp NeuralNetwork.cpp NeuralNetwork.h LinearLayer.h LinearLayer.cpp ActivationFunction.cpp ActivationFunction.h NonLinearLayer.h NonLinearLayer.cpp LossFunction.h LossFunction.cpp)

cmake_minimum_required(VERSION 3.5.0)

project(project VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(external/eigen)

include_directories(external/eigen)

set(SOURCES
    main.cpp
    NeuralNetwork.cpp
    LinearLayer.cpp
    ActivationFunction.cpp
    NonLinearLayer.cpp
    LossFunction.cpp
)

set(HEADERS
    NeuralNetwork.h
    LinearLayer.h
    ActivationFunction.h
    NonLinearLayer.h
    LossFunction.h
)

add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

target_link_libraries(${PROJECT_NAME} Eigen3::Eigen)
