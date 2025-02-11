#include "ActivationFunction.h"
#include <functional>
#include <cmath>

ActivationFunction::ActivationFunction(std::function<double(double)>&& activationFn, std::function<double(double)>&& derivativeFn) : 
            activationFn_(std::move(activationFn)), derivativeFn_(std::move(derivativeFn)) {}

// MatrixXd ActivationFunction::forward(const MatrixXd& input)
// {
//     return input.unaryExpr([this](double x) {return activationFn(x);});
// }

// MatrixXd ActivationFunction::backward(const MatrixXd& input, const MatrixXd& gradOutput)
// {
//     return input.unaryExpr([this](double x) {return derivativeFn(x);}).cwiseProduct(gradOutput);
// }

ActivationFunction ActivationFunction::ReLU()
{
    return ActivationFunction([] (double x) {return x > 0 ? x : 0; }, [] (double x) {return x > 0 ? 1.0 : 0.0; });
}

ActivationFunction ActivationFunction::Sigmoid()
{
    return ActivationFunction([] (double x) {return 1/(1 + std::exp(-x));}, [] (double x) {double sigma = 1/(1 + std::exp(-x)); return sigma * (1 - sigma);});
}

ActivationFunction ActivationFunction::LeakyReLU()
{
    return ActivationFunction([] (double x) {return x > 0 ? x : leakyReLUAlpha * x;}, [] (double x) {return x > 0 ? 1.0 : leakyReLUAlpha;});
}