#include "LinearLayer.h"

namespace NeuralNetwork {

LinearLayer::LinearLayer(X x, Y y, Random& rnd)
    : weights_(initializeMatrix(static_cast<Index>(y), static_cast<Index>(x), rnd)),
      biases_(initializeVector(static_cast<Index>(y), rnd)) {}

Matrix LinearLayer::forward(const Matrix& input) {
    if (!cache_) {
        cache_ = std::make_unique<Cache>();
    }
    cache_->input = input;
    Matrix output = weights_ * input;
    for (Index i = 0; i < output.cols(); ++i) {
        output.col(i) += biases_;
    }
    return output;
}

Matrix LinearLayer::computeGradients(const Matrix& gradOutput) {
    if (!cache_) {
        throw std::runtime_error("LinearLayer::computeGradients: cache пуст, forward не вызван");
    }
    return weights_.transpose() * gradOutput;
}

std::tuple<Matrix, Matrix, Vector> LinearLayer::computeGradientsWithParams(const Matrix& gradOutput) {
    if (!cache_) {
        throw std::runtime_error("LinearLayer::computeGradientsWithParams: cache пуст, forward не вызван");
    }
    const Matrix& input = cache_->input;
    Matrix gradInput = weights_.transpose() * gradOutput;
    Matrix gradW = gradOutput * input.transpose();
    Vector gradB = Vector::Zero(gradOutput.rows());
    for (Index i = 0; i < gradOutput.rows(); ++i) {
        gradB(i) = gradOutput.row(i).sum();
    }
    return {gradInput, gradW, gradB};
}

Matrix& LinearLayer::getWeight() {
    return weights_;
}

Vector& LinearLayer::getBias() {
    return biases_;
}

const Matrix& LinearLayer::getWeight() const {
    return weights_;
}

const Vector& LinearLayer::getBias() const {
    return biases_;
}

} // namespace NeuralNetwork
