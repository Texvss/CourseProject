#include "NonLinearLayer.h"

#include "neunet.h"

namespace NeuralNetwork {

NonLinearLayer::NonLinearLayer(ActivationFunction&& activateF)
    : activateF_(std::move(activateF)) {
}

void NonLinearLayer::turn_on_learning_mod() {
    cache_ = std::make_unique<Cache>();
}

void NonLinearLayer::turn_off_learning_mod() {
    cache_.reset();
}

Matrix NonLinearLayer::forward(const Matrix& input) {
    if (!cache_) {
        cache_ = std::make_unique<Cache>();
    }
    cache_->input = input;
    return activateF_.forward(input);
}

Matrix NonLinearLayer::backward(const Matrix& gradOutput) {
    if (!cache_) {
        throw std::runtime_error(
            "NonLinearLayer::backward called without forward pass");
    }

    Matrix localGrad = activateF_.backward(cache_->input, gradOutput);
    return localGrad;
}
}  // namespace NeuralNetwork
