#ifndef ACTIVATION_FUNCTION_H
#define ACTIVATION_FUNCTION_H

#include <Eigen/Dense>

#include <functional>
using Eigen::MatrixXd;

class ActivationFunction
{
    private:
        static constexpr double leakyReLUAlpha = 0.01;
    //     std::function<double(double)> activationFn_;
    //     std::function<double(double)> derivativeFn_;
    public:
    ActivationFunction(std::function<double(double)>&& activationFn, std::function<double(double)>&& derivativeFn);
    // MatrixXd forward(const MatrixXd& input);
    // MatrixXd backward(const MatrixXd& input, const MatrixXd& gradOutput);
    std::function<double(double)> activationFn_;
    std::function<double(double)> derivativeFn_;
    static ActivationFunction ReLU();
    static ActivationFunction Sigmoid();
    static ActivationFunction LeakyReLU();
};

#endif