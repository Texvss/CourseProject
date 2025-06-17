#pragma once

#include "Cache.h"
#include "neunet.h"
#include "Optimizer.h"
#include "Random.h"

namespace NeuralNetwork {

enum In : Index;
enum Out : Index;

class LinearLayer {
private:
    Matrix weights_;
    Vector biases_;
    std::unique_ptr<Cache> cache_;

    static Random& globalRandom();
    static Matrix initializeMatrix(Index rows, Index cols, Random& rnd);
    static Vector initializeVector(Index rows, Random& rnd);

public:
    LinearLayer(In x, Out y, Random& rnd = globalRandom());

    void turn_on_learning_mod();
    void turn_off_learning_mod();
    Matrix forward(const Matrix&& input);
    Matrix backward(const Matrix& gradOutput, Optimizer& opt,
                    double learningRate);
};
}  // namespace NeuralNetwork
