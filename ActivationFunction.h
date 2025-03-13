#ifndef ACTIVATION_FUNCTION_H
#define ACTIVATION_FUNCTION_H
#include <functional>

#include "neunet.h"

namespace NeuralNetwork {
class ActivationFunction {
private:
    using Function = std::function<double(double)>;

    const Function activationFn_;
    const Function derivativeFn_;

public:
    ActivationFunction(Function&& activationFn, Function&& derivativeFn);
    Matrix forward(const Matrix& input);
    Matrix evaluate0(const Matrix& arg) const;
    Matrix evaluate1(const Matrix& arg) const;
    double evaluate0(double arg) const;
    double evaluate1(double arg) const;
    Matrix backward(const Matrix& input, const Matrix& gradOutput);
    static ActivationFunction ReLU();
    static ActivationFunction Sigmoid();
    static ActivationFunction LeakyReLU();
};
}  // namespace NeuralNetwork
#endif
