#include "LossFunction.h"

#include <cassert>
#include <iostream>
#include <limits>

namespace NeuralNetwork {

LossFunction::LossFunction(LossFunc&& lossFn, GradFunc&& gradFn)
    : lossFn_(std::move(lossFn)), gradFn_(std::move(gradFn)) {
}

LossFunction LossFunction::MSE() {
    return LossFunction(
        [](const Matrix& predictions, const Matrix& actualOutput) {
            Matrix difference = actualOutput - predictions;
            return difference.array().square().mean();
        },
        [](const Matrix& predictions, const Matrix& actualOutput) {
            Matrix difference = actualOutput - predictions;
            return 2.0 * (predictions - actualOutput) /
                   (predictions.rows() * predictions.cols());
        });
}

LossFunction LossFunction::CrossEntropy() {
    return LossFunction(
        [](const Matrix& logits, const Matrix& Y_true) -> double {
            if (logits.rows() != Y_true.rows() ||
                logits.cols() != Y_true.cols()) {
                return 0.0;
            }
            auto maxPerCol = logits.colwise().maxCoeff();
            auto shifted =
                logits.array() - maxPerCol.replicate(logits.rows(), 1).array();
            auto expShift = shifted.array().exp().matrix();
            auto sumExp = expShift.colwise().sum().eval();
            sumExp = sumExp.array().max(1e-10);
            auto P = (expShift.array().rowwise() / sumExp.array()).matrix();
            auto logP = P.array().log().matrix();
            if (!logP.array().isFinite().all()) {
                return 0.0;
            }
            double loss =
                -(Y_true.array() * logP.array()).colwise().sum().mean();
            if (!std::isfinite(loss)) {
                return 0.0;
            }
            return loss;
        },
        [](const Matrix& logits, const Matrix& Y_true) -> Matrix {
            if (logits.rows() != Y_true.rows() ||
                logits.cols() != Y_true.cols()) {
                return Matrix(Y_true.rows(), Y_true.cols());
            }
            auto maxPerCol = logits.colwise().maxCoeff();
            auto shifted =
                logits.array() - maxPerCol.replicate(logits.rows(), 1).array();
            auto expShift = shifted.array().exp().matrix();
            auto sumExp = expShift.colwise().sum().eval();
            sumExp = sumExp.array().max(1e-10);
            auto P = (expShift.array().rowwise() / sumExp.array()).matrix();
            double B = static_cast<double>(logits.cols());
            auto grad = (P - Y_true) / B;
            if (!grad.array().isFinite().all()) {
                return Matrix::Zero(Y_true.rows(), Y_true.cols()).eval();
            }
            return grad;
        });
}

double LossFunction::computeLoss(const Matrix& predictions,
                                 const Matrix& actualOut) {
    assert(lossFn_);
    return lossFn_(predictions, actualOut);
}

Matrix LossFunction::computeGrad(const Matrix& predictions,
                                 const Matrix& actualOut) {
    assert(gradFn_);
    return gradFn_(predictions, actualOut);
}
}  // namespace NeuralNetwork
