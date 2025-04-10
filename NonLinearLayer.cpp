#include "NonLinearLayer.h"
#include "neunet.h"
#include <iostream>

namespace NeuralNetwork {

NonLinearLayer::NonLinearLayer(ActivationFunction&& activateF)
    : activateF_(std::move(activateF)) {
}

Matrix NonLinearLayer::forward(const Matrix& input) {
    if (!cache_) {
        cache_ = std::make_unique<Cache>();
    }
    cache_->input = input;
    return activateF_.evaluate0(input);
}

Matrix NonLinearLayer::backward(const Matrix& gradOutput) {
    if (!cache_) {
        throw std::runtime_error(
            "Ошибка: cache_ путстой, но backward() вызван!");
    }
    Matrix& input = cache_->input;
    return activateF_.evaluate1(cache_->input).cwiseProduct(gradOutput);
}
}  // namespace NeuralNetwork
