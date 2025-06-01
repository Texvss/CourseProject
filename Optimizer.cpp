#include "Optimizer.h"
#include <cassert>
#include <cmath>

namespace NeuralNetwork {

void sgdMatrixUpdate(Optimizer*, Matrix& params, const Matrix& grad, double learningRate, size_t index) {
    assert(learningRate > 0);
    if (params.rows() != grad.rows() || params.cols() != grad.cols()) {
        throw std::invalid_argument("Parameter and gradient size mismatch");
    }
    params.noalias() -= learningRate * grad;
}

void sgdVectorUpdate(Optimizer*, Vector& params, const Vector& grad, double learningRate, size_t index) {
    assert(learningRate > 0);
    if (params.size() != grad.size()) {
        throw std::invalid_argument("Parameter and gradient size mismatch");
    }
    params.noalias() -= learningRate * grad;
}

void adamUpdate(
    Matrix& params,
    const Matrix& grad,
    double learningRate,
    Momentum& momentum,
    double beta1,
    double beta2,
    double epsilon,
    size_t& t
) {
    assert(learningRate > 0);
    assert(beta1 > 0 && beta1 < 1);
    assert(beta2 > 0 && beta2 < 1);
    assert(epsilon > 0);

    if (params.rows() != grad.rows() || params.cols() != grad.cols()) {
        throw std::invalid_argument("Parameter and gradient size mismatch");
    }

    if (momentum.M.rows() != params.rows() || momentum.M.cols() != params.cols() ||
        momentum.V.size() != params.rows() * params.cols()) {
        momentum.M = Matrix::Zero(params.rows(), params.cols());
        momentum.V = Vector::Zero(params.rows() * params.cols());
    }

    ++t;

    momentum.M = beta1 * momentum.M + (1 - beta1) * grad;

    Vector gradVec = Eigen::Map<const Vector>(grad.data(), grad.size());
    Vector gradSquared = gradVec.cwiseProduct(gradVec);
    momentum.V = beta2 * momentum.V + (1 - beta2) * gradSquared;

    double corr1 = 1.0 - std::pow(beta1, t);
    double corr2 = 1.0 - std::pow(beta2, t);
    Matrix mHat = momentum.M / corr1;
    Vector vHat = momentum.V / corr2;

    Matrix vHatMat = Eigen::Map<const Matrix>(vHat.data(), params.rows(), params.cols());

    params.noalias() -= learningRate * (mHat.array() / (vHatMat.array().sqrt() + epsilon)).matrix();
}

Optimizer::Optimizer(
    VectorUpdateFunc&& vectorUpdateFn,
    MatrixUpdateFunc&& matrixUpdateFn,
    double learningRate
) : vectorUpdateFn_(std::move(vectorUpdateFn)),
    matrixUpdateFn_(std::move(matrixUpdateFn)),
    learningRate_(learningRate),
    beta1_(0.9),
    beta2_(0.999),
    epsilon_(1e-8),
    t_(0) {
}

Optimizer Optimizer::SGD(double learningRate) {
    return Optimizer(
        [](Optimizer* opt, Vector& params, const Vector& grad, double learningRate, size_t index) {
            sgdVectorUpdate(opt, params, grad, learningRate, index);
        },
        [](Optimizer* opt, Matrix& params, const Matrix& grad, double learningRate, size_t index) {
            sgdMatrixUpdate(opt, params, grad, learningRate, index);
        },
        learningRate
    );
}

Optimizer Optimizer::Adam(double learningRate, double beta1, double beta2, double epsilon) {
    return Optimizer(
        [learningRate, beta1, beta2, epsilon](Optimizer* opt, Vector& params, const Vector& grad, double learningRateOverride, size_t index) mutable {
            if (index >= opt->moments_.size()) {
                opt->moments_.resize(index + 1);
            }
            double effectiveLearningRate = (learningRateOverride > 0.0) ? learningRateOverride : learningRate;
            Matrix paramsMat = Eigen::Map<Matrix>(params.data(), params.size(), 1);
            Matrix gradMat = Eigen::Map<const Matrix>(grad.data(), grad.size(), 1);
            adamUpdate(paramsMat, gradMat, effectiveLearningRate, opt->moments_[index], beta1, beta2, epsilon, opt->t_);
            params = Eigen::Map<Vector>(paramsMat.data(), paramsMat.size());
        },
        [learningRate, beta1, beta2, epsilon](Optimizer* opt, Matrix& params, const Matrix& grad, double learningRateOverride, size_t index) mutable {
            if (index >= opt->moments_.size()) {
                opt->moments_.resize(index + 1);
            }
            double effectiveLearningRate = (learningRateOverride > 0.0) ? learningRateOverride : learningRate;
            adamUpdate(params, grad, effectiveLearningRate, opt->moments_[index], beta1, beta2, epsilon, opt->t_);
        },
        learningRate
    );
}

void Optimizer::update(Vector& params, const Vector& grad, size_t index, double learningRate) {
    vectorUpdateFn_(this, params, grad, learningRate, index);
}

void Optimizer::update(Matrix& params, const Matrix& grad, size_t index, double learningRate) {
    matrixUpdateFn_(this, params, grad, learningRate, index);
}

} // namespace NeuralNetwork
