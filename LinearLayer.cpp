#include <Eigen/Dense>
#include "LinearLayer.h"

LinearLayer::LinearLayer(InputSize x, OutputSize y) : x_(x), y_(y) 
{
    weights = MatrixXd::Random(y, x) * sqrt(2.0 / x);;
    biases = VectorXd::Zero(y);
    lastInput = MatrixXd::Zero(x, 1);
    gradWeights = MatrixXd::Zero(y,x);
    gradBiases = VectorXd::Zero(y);
}
VectorXd LinearLayer::forward(const VectorXd& input)
{
    this->lastInput = input;
    VectorXd output = weights * input + biases;
    
    return output;
}


VectorXd LinearLayer::backward(const VectorXd& gradOutput)
{
    VectorXd gradInput = weights.transpose() * gradOutput;

    gradWeights = gradOutput * lastInput.transpose();
    gradBiases = gradOutput;

    return gradInput;
}


// void LinearLayer::updateParametrs(double learningSpeed)
// {
//     weights = weights - learningSpeed * gradWeights;
//     biases = biases - learningSpeed * gradBiases;
// }