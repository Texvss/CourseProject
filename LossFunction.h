#pragma once

#include <functional>

#include "neunet.h"

namespace NeuralNetwork {

class LossFunction {
private:
    using LossSignature = double(const Matrix&, const Matrix&);
    using GradSignature = Matrix(const Matrix&, const Matrix&);
    using LossFunc = std::function<LossSignature>;
    using GradFunc = std::function<GradSignature>;

    LossFunc lossFn_;
    GradFunc gradFn_;

public:
    LossFunction(LossFunc&& lossFn, GradFunc&& gradFn);

    static LossFunction MSE();
    static LossFunction CrossEntropy();
    double computeLoss(const Matrix& predictions, const Matrix& actualOut);
    Matrix computeGrad(const Matrix& predictions, const Matrix& actualOut);
};
}  // namespace NeuralNetwork
