#include "LinearLayer.h"
#include <iostream>

namespace NeuralNetwork {

LinearLayer::LinearLayer(X x, Y y, Random& rnd)
    : weights_(initializeMatrix(x, y, rnd)),
      biases_(initializeVector(x, rnd)) {}

Matrix LinearLayer::forward(const Matrix& input) {
    if (!cache_) {
        cache_ = std::make_unique<Cache>();
    }
    cache_->input = input;
    Matrix output = (weights_ * input).colwise() + biases_;
    return output;
}

Matrix LinearLayer::backward(const Matrix& gradOutput, double learningSpeed) {
    if (!cache_) {
        throw std::runtime_error(
            "Ошибка: cache_ путстой, но backward() вызван!");
    }
    const Matrix input = cache_->input;
    Matrix gradInput = weights_.transpose() * gradOutput;
    Matrix gradWeights_ = gradOutput * input.transpose();
    Vector gradBiases_ = gradOutput.rowwise().sum();
    ;
    weights_ -= learningSpeed * gradWeights_;
    biases_ -= learningSpeed * gradBiases_;
    return gradInput;
}
}  // namespace NeuralNetwork
