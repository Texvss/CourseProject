#include "SGD.h"

#include <cassert>

namespace NeuralNetwork {
SGD::SGD(double learningRate) : learningRate_(learningRate) {
    assert(learningRate > 0 && "Learning rate must be positive");
}

void SGD::update(Vector& params, const Vector& grad) {
    assert(params.size() == grad.size() &&
           "Parameter and gradient size mismatch");

    params -= learningRate_ * grad;
}

void SGD::update(Matrix& params, const Matrix& grad) {
    assert(params.rows() == grad.rows() && params.cols() == grad.cols() &&
           "Parameter and gradient size mismatch");

    params -= learningRate_ * grad;
}
}  // namespace  NeuralNetwork
