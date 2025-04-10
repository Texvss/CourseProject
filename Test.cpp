#include "Test.h"

namespace NeuralNetwork{

    void test1(){
    Matrix inputs(3, 3);
    inputs << 1.0, 4.0, 2.0, 5.0, 3.0, 6.0, 8.0, 5.0, 7.0;

    Matrix targets(3, 3);
    targets << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0;

    LinearLayer linearLayer(X(3), Y(3));
    NonLinearLayer nonLinearLayer(ActivationFunction::Sigmoid());
    LossFunction lossFunction = LossFunction::MSE();

    std::cout << "Прямой проход:" << '\n';
    std::vector<Vector> columns;
    for (int i = 0; i < inputs.cols(); ++i) {
        columns.push_back(linearLayer.forward(inputs.col(i)));
    }
    Matrix linearOutput(columns[0].size(), inputs.cols());
    for (int i = 0; i < inputs.cols(); ++i) {
        linearOutput.col(i) = columns[i];
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
        gradLinear = linearLayer.backward(grad, 0.01);
    }
    std::cout << "Градиент после линейного слоя:\n" << gradLinear << '\n';
    }
} // namespace 