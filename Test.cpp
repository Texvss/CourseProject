#include "Test.h"
#include "Random.h"
#include <utility>

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

        Random rnd(42);
        LinearLayer layer(X(1), Y(1), rnd);
        SGDOptimizer opt(0.1);
        LossFunction loss = LossFunction::MSE();

        const int epochs = 1000;
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

    int globalTest() {
        std::vector<Matrix> trainImages;
        std::vector<Vector> trainLabels;
        std::vector<Matrix> testImages;
        std::vector<Vector> testLabels;

        if (!MNISTLoader::load("../data/train-images-idx3-ubyte", 
                           "../data/train-labels-idx1-ubyte", 
                           trainImages, trainLabels) ||
        !MNISTLoader::load("../data/t10k-images-idx3-ubyte", 
                           "../data/t10k-labels-idx1-ubyte", 
                           testImages, testLabels)) {
            std::cerr << "Failed to load MNIST data" << std::endl;
            return -1;
    }

        Random rnd(42345);
        DataLoader trainLoader(trainImages, trainLabels, /*batchSize=*/64, rnd);
        DataLoader testLoader(testImages,  testLabels,  /*batchSize=*/64, rnd);

        NeuralNetwork model;
        model.addLayer<LinearLayer>(X(784), Y(128), rnd);
        model.addLayer<NonLinearLayer>(ActivationFunction::ReLU());
        model.addLayer<LinearLayer>(X(128), Y(10), rnd);

        LossFunction loss = LossFunction::MSE();
        Train trainer(model, loss, /*lr=*/0.01);
        trainer.fit(trainLoader, /*epochs=*/10, /*shuffle=*/true);

        double sumLoss = 0.0;
        int correct = 0;
        int total = 0;

        while (testLoader.isNext()) {
            Batch batch = testLoader.nextBatch();

            Matrix X(batch.inputs[0].rows(), batch.inputs.size());
            for (size_t i = 0; i < batch.inputs.size(); ++i) {
                X.col(i) = batch.inputs[i];
            }

            Matrix preds = model.forward(X);
            sumLoss += loss.computeLoss(preds,
                                    [&](){
                                        Matrix Y(batch.targets[0].size(), batch.targets.size());
                                        for (size_t i = 0; i < batch.targets.size(); ++i)
                                            Y.col(i) = batch.targets[i];
                                        return Y;
                                    }());

            for (int i = 0; i < preds.cols(); ++i) {
                int p; preds.col(i).maxCoeff(&p);
                int t; batch.targets[i].maxCoeff(&t);
                if (p == t) ++correct;
                ++total;
            }
    }

    std::cout << "Test Loss: " << (sumLoss / (total / 64))
              << ", Accuracy: " << (100.0 * correct / total) << "%" << std::endl;

    return 0;
    }


} // namespace Test
