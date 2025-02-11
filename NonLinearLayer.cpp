#include "NonLinearLayer.h"
#include <iostream>
#include <Eigen/Dense>

NonLinearLayer::NonLinearLayer(ActivationFunction&& activateF) : activateF_(std::move(activateF)) {}

MatrixXd NonLinearLayer::forward(const MatrixXd& input)
{
    this->inputStore = input;
    MatrixXd output;
    output = input.unaryExpr([this](double x) {return activateF_.activationFn_(x); });
    return output;
}

MatrixXd NonLinearLayer::backward(const MatrixXd& gradOutput)
{
    return inputStore.unaryExpr([this](double x) { return activateF_.derivativeFn_(x); }).cwiseProduct(gradOutput);
}