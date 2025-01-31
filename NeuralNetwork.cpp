#include "NeuralNetwork.h"
#include <random>
#include <iostream>

NeuralNetwork::NeuralNetwork(double _W, double _b, double _learningSpeed)
    : W(_W), b(_b), learningSpeed(_learningSpeed) {}

NeuralNetwork::~NeuralNetwork() {}

void NeuralNetwork::Paramets()
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-0.01, 0.01);
    this->W = dis(gen); // параметру W присваевается случайное число в этом диапозоне (-0.01, 0.01)
    this->b = 0; // присваиваем параметру b число ноль
}

double NeuralNetwork::Forward(double number)
{
    double expectedvalue;
    expectedvalue = this->W * number + this->b;
    
    return expectedvalue;
}

double NeuralNetwork::Error(double predicted, double actual)
{
    return pow(actual - predicted, 2); // среднеквадратичная ошибка MSE
    // return abs(actual - predicted) // модуль ошибка MAE
}

void NeuralNetwork::Train(std::vector<double> inputsX, std::vector<double> outPutsY)
{
    // this->learningSpeed = 0.01;
    int epochs = 10000; // количество циклов обучения
    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        for (size_t j = 0; j < inputsX.size(); ++j)
        {
            double expetedValue = this->Forward(inputsX[j]);
            // double error = this->Error(expetedValue, outPutsY[j]);
            double gradW = -2 * (outPutsY[j] - expetedValue) * inputsX[j];
            double gradB = -2 * (outPutsY[j] - expetedValue);
            this->W = this->W - this->learningSpeed * gradW;
            this->b = this->b - this->learningSpeed * gradB;
        }
    }
}

std::vector<double> NeuralNetwork::Predict(std::vector<double> inputsX)
{
    std::vector<double> predictions;
    for (size_t i = 0; i < inputsX.size(); ++i)
    {
        double outputY = this->W * inputsX[i] + this->b;
        predictions.push_back(outputY);
    }
    return predictions;
}
