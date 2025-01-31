#include "NonLinearLayer.h"
#include <iostream>
#include <Eigen/Dense>

NonLinearLayer::NonLinearLayer(const ActivationFunction& activationFunction) : activationFunction(activationFunction) {}

MatrixXd NonLinearLayer::forward(const MatrixXd& input)
{
    this->inputStore = input;
    MatrixXd output;
    output = activationFunction.forward(input);
    return output;
}

MatrixXd NonLinearLayer::backward(const MatrixXd& gradOutput)
{
    MatrixXd gradInput;
    gradInput = activationFunction.backward(this->inputStore, gradOutput);
    return gradInput;
}