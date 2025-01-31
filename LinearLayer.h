#ifndef LINEAR_LAYER_H
#define LINEAR_LAYER_H

#include <Eigen/Dense>

using Eigen::MatrixXd;
using Eigen::VectorXd;

class LinearLayer
{
    private:
        MatrixXd weights; // матрица весов
        VectorXd biases; // вектор смещений
        MatrixXd gradWeights; // используется для обновления весов
        VectorXd gradBiases; // используется для оюновления смещений
        int inputSize; // размер входа
        int outputSize; // размер выхода
        MatrixXd lastInput;
        double learningSpeed = 0.01;
    public:
        LinearLayer(int _inputSize, int _outputSize);
        ~LinearLayer();
        VectorXd forward(const VectorXd& input); // метод для прямого прохода, вычисление output'а
        VectorXd backward(const VectorXd& gradOutput); // метод для обратного проход, вычисдение градиента output'ов
        // void updateParametrs(double learningSpeed);
};


#endif