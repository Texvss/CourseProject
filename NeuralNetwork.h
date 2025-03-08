#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <vector>


class NeuralNetwork
{
  private:
    double W; // weight
    double b; // по сути это просто смещение
    double learningSpeed = 0.01;
  public:
    NeuralNetwork(double _W, double _b, double _learningSpeed);
    ~NeuralNetwork();
    
    void Paramets(); // в самом начале нужно что-то присвоить W и b для этого и нужен этот метод
    double Forward(double number); // этот метод будет вычислять предсказание
    double Error(double predicted, double actual); // этот метод нужен для вычисления погрешности
    void Train(std::vector<double> inputsX, std::vector<double> outPutsY); // метод для обучения
    std::vector<double> Predict(std::vector<double> inputsX); // метод для использования обученной модели
};

#endif