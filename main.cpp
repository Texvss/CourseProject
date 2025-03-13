#include <iostream>

#include "ActivationFunction.h"
#include "LinearLayer.h"
#include "LossFunction.h"
#include "neunet.h"
#include "NonLinearLayer.h"

using namespace NeuralNetwork;

int main() {
    Matrix inputs(3, 2);
    inputs << 1.0, 4.0, 2.0, 5.0, 3.0, 6.0;

    Matrix targets(2, 2);
    targets << 1.0, 0.0, 0.0, 1.0;

    LinearLayer linearLayer(Input(3), Output(2));
    NonLinearLayer nonLinearLayer(ActivationFunction::ReLU());
    LossFunction lossFunction = LossFunction::MSE();

    std::cout << "Прямой проход:" << '\n';
    Matrix linearOutput(2, 2);
    for (int i = 0; i < inputs.cols(); ++i) {
        linearOutput.col(i) = linearLayer.forward(inputs.col(i));
    }
    std::cout << "Результат после линейного слоя:\n" << linearOutput << '\n';

    Matrix activationOutput = nonLinearLayer.forward(linearOutput);
    std::cout << "Результат после нелинейного слоя:\n"
              << activationOutput << '\n';

    double loss = lossFunction.computeLoss(activationOutput, targets);
    std::cout << "Значение функции потерь: " << loss << '\n';

    std::cout << "Обратный проход:" << '\n';
    Matrix gradLoss = lossFunction.computeGrad(activationOutput, targets);
    std::cout << "Градиент функции потерь:\n" << gradLoss << '\n';

    Matrix gradActivation = nonLinearLayer.backward(gradLoss);
    std::cout << "Градиент после нелинейного слоя:\n" << gradActivation << '\n';

    Matrix gradLinear(2, 3);
    for (int i = 0; i < gradActivation.cols(); ++i) {
        Vector grad = gradActivation.col(i);
        gradLinear.col(i) = linearLayer.backward(grad, 0.01);
    }
    std::cout << "Градиент после линейного слоя:\n" << gradLinear << '\n';

    return 0;
}
