#include "LinearLayer.h"

#include <cassert>

namespace NeuralNetwork {

LinearLayer::LinearLayer(In x, Out y, Random& rnd)
    : weights_(initializeMatrix(y, x, rnd)), biases_(initializeVector(y, rnd)) {
}

Random& LinearLayer::globalRandom() {
    static Random rnd(42);
    return rnd;
}

Matrix LinearLayer::initializeMatrix(Index rows, Index cols, Random& rnd) {
    assert(rows > 0 && cols > 0);
    return rnd.uniformMatrix(rows, cols, -1, 1);
}

Vector LinearLayer::initializeVector(Index rows, Random& rnd) {
    assert(rows > 0);
    Vector v = rnd.uniformVector(rows, -1, 1);
    assert(v.rows() == rows && v.cols() == 1);
    return v;
}

void LinearLayer::turn_on_learning_mod() {
    cache_ = std::make_unique<Cache>();
}

void LinearLayer::turn_off_learning_mod() {
    cache_.reset();
}

Matrix LinearLayer::forward(const Matrix&& input) {
    if (!cache_) {
        cache_ = std::make_unique<Cache>();
    }
    cache_->input = std::move(input);
    assert(biases_.rows() == (weights_ * input).rows());
    return (weights_ * input).colwise() + biases_;
}

Matrix LinearLayer::backward(const Matrix& gradOutput, Optimizer& opt,
                             double learningRate) {
    assert(cache_ != nullptr);

    const Matrix& X = cache_->input;
    Index batchSize = static_cast<Index>(X.cols());

    auto gradW = gradOutput * X.transpose();
    auto gradB = gradOutput.rowwise().sum();
    Matrix gradInput = weights_.transpose() * gradOutput;

    opt.update(weights_, gradW);
    opt.update(biases_, gradB);

    return gradInput;
}
}  // namespace NeuralNetwork
