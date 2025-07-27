#pragma once

#include "neunet.h"

namespace NeuralNetwork {
class SGD {
private:
    double learningRate_;

public:
    SGD(double learningRate);

    void update(Vector& params, const Vector& grad);
    void update(Matrix& params, const Matrix& grad);
};
}  // namespace NeuralNetwork
