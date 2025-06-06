#include "ActivationFunction.h"

#include <cmath>

namespace NeuralNetwork {

ActivationFunction::ActivationFunction(Function&& activationFn,
                                       Function&& derivativeFn)
    : activationFn_(std::move(activationFn)),
      derivativeFn_(std::move(derivativeFn)) {
}

Matrix ActivationFunction::forward(const Matrix& input) {
    auto result =
        input.unaryExpr([this](double x) { return activationFn_(x); });
    return result;
}

Matrix ActivationFunction::backward(const Matrix& input,
                                    const Matrix& gradOutput) {
    return input.unaryExpr([this](double x) { return derivativeFn_(x); })
        .cwiseProduct(gradOutput);
}

Matrix ActivationFunction::evaluate0(const Matrix& arg) const {
    return arg.unaryExpr(activationFn_);
}

Matrix ActivationFunction::evaluate1(const Matrix& arg) const {
    return arg.unaryExpr(derivativeFn_);
}

double ActivationFunction::evaluate0(double arg) const {
    return activationFn_(arg);
}

double ActivationFunction::evaluate1(double arg) const {
    return derivativeFn_(arg);
}

ActivationFunction ActivationFunction::ReLU() {
    return ActivationFunction([](double x) { return x > 0 ? x : 0; },
                              [](double x) { return x > 0 ? 1.0 : 0.0; });
}

ActivationFunction ActivationFunction::Sigmoid() {
    return ActivationFunction([](double x) { return 1 / (1 + std::exp(-x)); },
                              [](double x) {
                                  double sigma = 1 / (1 + std::exp(-x));
                                  return sigma * (1 - sigma);
                              });
}

ActivationFunction ActivationFunction::LeakyReLU(double alpha) {
    return ActivationFunction([=](double x) { return x > 0 ? x : alpha * x; },
                              [=](double x) { return x > 0 ? 1.0 : alpha; });
}
}  // namespace NeuralNetwork
