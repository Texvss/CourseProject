#include "NonLinearLayer.h"
#include "neunet.h"
#include <iostream>

namespace NeuralNetwork {

NonLinearLayer::NonLinearLayer(ActivationFunction&& activateF)
    : activateF_(std::move(activateF)) {}

Matrix NonLinearLayer::forward(const Matrix& input) {
    if (!cache_) {
        cache_ = std::make_unique<Cache>();
    }
    cache_->input = input;
    return activateF_.forward(input);
}

Matrix NonLinearLayer::computeGradients(const Matrix& gradOutput) {
    if (!cache_) {
        throw std::runtime_error("NonLinearLayer::computeGradients: кеш пуст, forward не вызван");
    }
    return activateF_.backward(cache_->input, gradOutput);
}

}  // namespace NeuralNetwork
