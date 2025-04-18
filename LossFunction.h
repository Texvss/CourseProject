#ifndef LOSS_FUNCTION_H
#define LOSS_FUNCTION_H
#include <functional>
#include "neunet.h"

namespace NeuralNetwork {

using LossSignature = double(const Matrix&, const Matrix&);
using GradSignature = Matrix(const Matrix&, const Matrix&);
using LossFunc = std::function<LossSignature>;
using GradFunc = std::function<GradSignature>;

class LossFunction {
private:
    std::function<double(const Matrix&, const Matrix&)> lossFn_;
    std::function<Matrix(const Matrix&, const Matrix&)> gradFn_;

public:
    LossFunction(LossFunc&& lossFn, GradFunc&& gradFn);
    static LossFunction MSE();
    double computeLoss(const Matrix& predictions, const Matrix& actualOut);
    Matrix computeGrad(const Matrix& predictions, const Matrix& actualOut);
};
}  // namespace NeuralNetwork
#endif
