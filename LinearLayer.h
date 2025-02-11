#ifndef LINEAR_LAYER_H
#define LINEAR_LAYER_H

#include <Eigen/Dense>

using Eigen::MatrixXd;
using Eigen::VectorXd;


enum InputSize : int {};
enum OutputSize : int {};

class LinearLayer
{
    private:
        MatrixXd weights; // матрица весов
        VectorXd biases; // вектор смещений
        MatrixXd gradWeights; // используется для обновления весов
        VectorXd gradBiases; // используется для оюновления смещений
        InputSize x_; // размер входа
        OutputSize y_; // размер выхода
        MatrixXd lastInput;
        // double learningSpeed = 0.01;
    public:
        LinearLayer(InputSize x, OutputSize y);
        VectorXd forward(const VectorXd& input); // метод для прямого прохода, вычисление output'а
        VectorXd backward(const VectorXd& gradOutput); // метод для обратного проход, вычисдение градиента output'ов
        // void updateParametrs(double learningSpeed);
};


#endif