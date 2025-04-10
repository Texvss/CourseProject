#include "NeuralNetwork.h"

#include <iostream>
#include <random>

NeuralNetwork::NeuralNetwork(double W_, double b_, double learningSpeed_)
    : W(W_), b(b_), learningSpeed(learningSpeed_) {
}
double NeuralNetwork::Forward(double number) {
    double expectedvalue;
    expectedvalue = this->W * number + this->b;

    return expectedvalue;
}

double NeuralNetwork::Error(double predicted, double actual) {
    return pow(actual - predicted, 2);
    // return abs(actual - predicted)
}

void NeuralNetwork::Train(std::vector<double> inputsX,
                          std::vector<double> outPutsY) {
    for (int epoch = 0; epoch < epoch; ++epoch) {
        for (size_t j = 0; j < inputsX.size(); ++j) {
            double expetedValue = this->Forward(inputsX[j]);
            double gradW = -2 * (outPutsY[j] - expetedValue) * inputsX[j];
            double gradB = -2 * (outPutsY[j] - expetedValue);
            this->W = this->W - this->learningSpeed * gradW;
            this->b = this->b - this->learningSpeed * gradB;
        }
    }
}

std::vector<double> NeuralNetwork::Predict(std::vector<double> inputsX) {
    std::vector<double> predictions;
    for (size_t i = 0; i < inputsX.size(); ++i) {
        double outputY = this->W * inputsX[i] + this->b;
        predictions.push_back(outputY);
    }
    return predictions;
}
