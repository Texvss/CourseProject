#include <iostream>
#include <Eigen/Dense>
#include "LinearLayer.h"

LinearLayer::LinearLayer(int _inputSize, int _outputSize) : inputSize(_inputSize), outputSize(_outputSize) 
{
    weights = MatrixXd::Random(outputSize, inputSize) * sqrt(2.0 / inputSize);;
    biases = VectorXd::Zero(outputSize);
    lastInput = MatrixXd::Zero(inputSize, 1);
    gradWeights = MatrixXd::Zero(outputSize,inputSize);
    gradBiases = VectorXd::Zero(outputSize);
}

LinearLayer::~LinearLayer() {}

VectorXd LinearLayer::forward(const VectorXd& input)
{
    if (input.size() != inputSize)
    {
        throw std::invalid_argument("Разный размер");
    }
    this->lastInput = input;
    VectorXd output = weights * input + biases;
    
    return output;
}


VectorXd LinearLayer::backward(const VectorXd& gradOutput)
{
    if (gradOutput.size() != outputSize)
    {
        throw std::invalid_argument("Разный размер");
    }
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