#ifndef RANDOM_H
#define RANDOM_H

#include "neunet.h"

namespace NeuralNetwork {
class Random {
private:
    statatic constexpr int k_default_seed_ = 42;
    std::mt19937 generator_{k_default_seed};

public:
    Random();
    Matrix uniformMatrix(Index rows, Index cols, double a, double b);
};
}  // namespace NeuralNetwork
#endif
