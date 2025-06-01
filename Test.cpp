#include "Test.h"
#include "Random.h"
#include <vector>
#include <random>
#include <iostream>
#include <iomanip>

namespace NeuralNetwork {

int globalTest() {
    std::cout << "Initializing random number generator with fixed seed 42345..." << '\n';
    Random rnd(42345);

    std::cout << "Loading MNIST data..." << '\n';
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
        std::cerr << "Error: Failed to load MNIST data" << '\n';
        return -1;
    }

    std::cout << "Loaded " << trainImages.size() << " training samples and "
              << testImages.size() << " test samples" << '\n';

    if (!trainImages.empty() && !trainLabels.empty()) {
        std::cout << "Training image dimensions: " << trainImages[0].rows() << "x" << trainImages[0].cols()
                  << ", label dimensions: " << trainLabels[0].rows() << "x" << trainLabels[0].cols() << '\n';
    }

    std::cout << "Creating data loaders with batch size 64..." << '\n';
    DataLoader trainLoader(trainImages, trainLabels, 64, rnd);
    DataLoader testLoader(testImages, testLabels, 64, rnd);

    std::cout << "Building neural network..." << '\n';
    NeuralNetwork model;
    try {
        model.addLayer<LinearLayer>(X(784), Y(128), rnd);
        std::cout << "Added LinearLayer: 784 -> 128" << '\n';
        model.addLayer<NonLinearLayer>(ActivationFunction::ReLU());
        std::cout << "Added NonLinearLayer: ReLU" << '\n';
        model.addLayer<LinearLayer>(X(128), Y(10), rnd);
        std::cout << "Added LinearLayer: 128 -> 10" << '\n';
        model.addLayer<NonLinearLayer>(ActivationFunction::LeakyReLU(0.05));
        std::cout << "Added NonLinearLayer: LeakyReLU(0.05)" << '\n';
    } catch (const std::exception& e) {
        std::cerr << "Error: Failed to build model: " << e.what() << '\n';
        return -1;
    }

    LossFunction lossFunc = LossFunction::CrossEntropy();
    Optimizer optimizer = Optimizer::Adam(0.001);
    double learningRate = 0.001;

    std::cout << "Configuring trainer with CrossEntropy loss and optimizer (learning rate: " << learningRate << ")..." << '\n';
    Train trainer(model, lossFunc, optimizer, learningRate);

    std::cout << "Starting training for 10 epochs..." << '\n';
    try {
        trainer.fit(trainLoader, 10, true);
    } catch (const std::exception& e) {
        std::cerr << "Error: Training failed: " << e.what() << '\n';
        return -1;
    }
    std::cout << "Training completed" << '\n';

    std::cout << "Evaluating on test set..." << '\n';
    double totalLoss = 0.0;
    int correct = 0;
    int total = 0;

    testLoader.reset();
    int batchCount = 0;
    while (testLoader.isNext()) {
        Batch batch = testLoader.nextBatch();
        batchCount++;
        total += batch.inputs.size();
        std::cout << "Processing test batch " << batchCount << " (" << total << " samples processed)..." << '\n';

        Matrix X(batch.inputs[0].rows(), batch.inputs.size());
        for (size_t i = 0; i < batch.inputs.size(); ++i) {
            X.col(i) = batch.inputs[i];
        }

        Matrix preds = model.forward(X);

        Matrix Y(batch.targets[0].rows(), batch.targets.size());
        for (size_t i = 0; i < batch.targets.size(); ++i) {
            Y.col(i) = batch.targets[i];
        }

        totalLoss += lossFunc.computeLoss(preds, Y);

        for (int i = 0; i < preds.cols(); ++i) {
            Index predLabel;
            preds.col(i).maxCoeff(&predLabel);
            Index trueLabel;
            batch.targets[i].maxCoeff(&trueLabel);
            if (predLabel == trueLabel) {
                ++correct;
            }
        }
    }

    double avgLoss = totalLoss / (total / 64.0);
    double accuracy = 100.0 * correct / total;

    std::cout << "Test Loss: " << std::fixed << std::setprecision(6) << avgLoss
              << ", Accuracy: " << std::fixed << std::setprecision(2) << accuracy << "%" << '\n';

    if (accuracy < 80.0) {
        std::cerr << "Warning: Accuracy below 80%, model may not have trained correctly" << '\n';
        return 1;
    }

    return 0;
}

} // namespace NeuralNetwork
