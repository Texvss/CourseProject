#include "Test.h"
#include "Random.h"

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

    void testSGD() {
        std::cout << "=== testSGD() ===\n";
        Matrix grad(2,2);
        grad << 1.0, 2.0,
                3.0, 4.0;

        SGDOptimizer opt(0.5);
        Matrix update = opt.update(grad);
        std::cout << "Исходный градиент:\n" << grad << "\n\n";
        std::cout << "SGD update (η=0.5):\n" << update << "\n\n";
    }
    void testTrain() {
        std::cout << "=== testTrain() Linear Regression ===\n";

        Matrix inputs(1, 4);
        inputs << 1.0, 2.0, 3.0, 4.0;
        Matrix targets(1, 4);
        targets << 3.0, 5.0, 7.0, 9.0;

        // 2) Создаём слой, оптимизатор и функцию потерь
        Random rnd(42);
        LinearLayer layer(X(1), Y(1), rnd);
        SGDOptimizer opt(0.1);
        LossFunction loss = LossFunction::MSE();

        const int epochs = 150;
        for (int epoch = 1; epoch <= epochs; ++epoch) {
            Matrix pred = layer.forward(inputs);
            double L = loss.computeLoss(pred, targets);

            Matrix gradLoss = loss.computeGrad(pred, targets);
            layer.backward(gradLoss, /*learningSpeed=*/0.1);

            if (epoch % 20 == 0) {
                std::cout << "Epoch " << epoch
                          << " Loss=" << L << '\n';
            }
        }

        Matrix finalPred = layer.forward(inputs);
        std::cout << "\nAfter training, predictions:\n"
                  << finalPred << "\n";
        std::cout << "True targets:\n" << targets << "\n";
    }


} // namespace 
