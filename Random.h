#pragma once

#include <EigenRand/EigenRand>
#include <iterator>

#include "neunet.h"

namespace NeuralNetwork {

class Random {
private:
    static constexpr int k_default_seed_ = 42345;
    std::mt19937 generator_{k_default_seed_};

public:
    Random(int seed);

    template <typename Iterator>
    void shuffle(Iterator begin, Iterator end) {
        std::shuffle(begin, end, generator_);
    }
    Matrix uniformMatrix(Index rows, Index cols, double a, double b);
    Vector uniformVector(Index rows, double a, double b);
};
}  // namespace NeuralNetwork
