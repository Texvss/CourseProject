#include "LinearLayer.h"

namespace NeuralNetwork {
LinearLayer::LinearLayer(Input x, Output y)
    : weights_(Matrix::Random(y, x) * sqrt(2.0 / x)),
      biases_(Vector::Random(y) * sqrt(2.0 / y)) {
}

Matrix LinearLayer::forward(const Matrix& input) {
    if (cache_) {
        cache_->input = input;
    }
    Matrix output = (weights_ * input).colwise() + biases_;
    return output;
}

Matrix LinearLayer::backward(const Vector& gradOutput, double learningSpeed) {
    if (!cache_) {
        throw std::runtime_error(
            "Ошибка: cache_ путстой, но backward() вызван!");
    }
    const Matrix input = cache_->input;
    Vector gradInput = weights_.transpose() * gradOutput;
    Matrix gradWeights_ = gradOutput * input.transpose();
    Vector gradBiases_ = gradOutput.rowwise().sum();
    ;
    weights_ -= learningSpeed * gradWeights_;
    biases_ -= learningSpeed * gradBiases_;
    return gradInput;
}
}  // namespace NeuralNetwork
