#ifndef RANDOM_H
#define RANDOM_H

#include "neunet.h"
#include <EigenRand/EigenRand>

namespace NeuralNetwork {
    
class Random {
private:
    static constexpr int k_default_seed_ = 42345;
    std::mt19937 generator_{k_default_seed_};

public:
    Random(int seed);
    std::mt19937& engine();
    Matrix uniformMatrix(Index rows, Index cols, double a, double b);
    Vector uniformVector(Index rows, double a, double b);
};
}  // namespace NeuralNetwork
#endif

