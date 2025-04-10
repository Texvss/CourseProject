#include "LossFunction.h"
#include <cassert>

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
