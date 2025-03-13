#include "NonLinearLayer.h"

#include "neunet.h"

namespace NeuralNetwork {
NonLinearLayer::NonLinearLayer(ActivationFunction&& activateF)
    : activateF_(std::move(activateF)) {
}

Matrix NonLinearLayer::forward(const Matrix& input) {
    if (cache_) {
        cache_->input = input;
    }
    this->inputStore_ = input;
    return activateF_.evaluate0(input);
}

Matrix NonLinearLayer::backward(const Matrix& gradOutput) {
    Matrix input;
    if (!cache_) {
        throw std::runtime_error(
            "Ошибка: cache_ путстой, но backward() вызван!");
    }
    cache_->input = input;
    return activateF_.evaluate1(input).cwiseProduct(gradOutput);
}
}  // namespace NeuralNetwork
