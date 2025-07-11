#include "Test.h"

#include <cassert>
#include <filesystem>
#include <iomanip>
#include <iostream>

namespace NeuralNetwork {

NeuralNetwork makeModel1(Random rnd) {
    NeuralNetwork model;
    model.addLayer<LinearLayer>(In(784), Out(256), rnd);
    model.addLayer<NonLinearLayer>(ActivationFunction::ReLU());
    model.addLayer<LinearLayer>(In(256), Out(128), rnd);
    model.addLayer<NonLinearLayer>(ActivationFunction::LeakyReLU(0.05));
    model.addLayer<LinearLayer>(In(128), Out(10), rnd);
    model.addLayer<NonLinearLayer>(ActivationFunction::LeakyReLU(0.05));
    return model;
}

NeuralNetwork makeModel2(Random rnd) {
    NeuralNetwork model;
    model.addLayer<LinearLayer>(In(784), Out(128), rnd);
    model.addLayer<NonLinearLayer>(ActivationFunction::ReLU());
    model.addLayer<LinearLayer>(In(128), Out(10), rnd);
    model.addLayer<NonLinearLayer>(ActivationFunction::LeakyReLU(0.05));
    return model;
}

NeuralNetwork makeModel3(Random rnd) {
    NeuralNetwork model;
    model.addLayer<LinearLayer>(In(784), Out(512), rnd);
    model.addLayer<NonLinearLayer>(ActivationFunction::LeakyReLU(0.01));
    model.addLayer<LinearLayer>(In(512), Out(64), rnd);
    model.addLayer<NonLinearLayer>(ActivationFunction::Sigmoid());
    model.addLayer<LinearLayer>(In(64), Out(10), rnd);
    model.addLayer<NonLinearLayer>(ActivationFunction::ReLU());
    return model;
}

Stats trainModelAlgo1(NeuralNetwork& model, DataLoader& trainLoader,
                      DataLoader& testLoader) {
    LossFunction lossFunc = LossFunction::CrossEntropy();
    Optimizer opt = Optimizer(SGD(0.01));
    double learningRate = 0.01;
    int epochs = 10;
    Train trainer(lossFunc, std::move(opt), learningRate);
    std::cout << "Training for " << epochs << " epochs...\n";
    trainer.fit(model, trainLoader, testLoader, static_cast<Epoches>(epochs),
                Shuffle::Enable);

    double totalLoss = 0.0;
    int correct = 0;
    int total = 0;
    testLoader.rewind();
    while (testLoader.hasNext()) {
        auto batch = testLoader.nextBatch();
        total += batch.X.cols();
        Matrix preds = model.forward(std::move(batch.X));
        totalLoss += lossFunc.computeLoss(preds, batch.Y) * batch.X.cols();
        for (int i = 0; i < preds.cols(); ++i) {
            Index predLabel;
            preds.col(i).maxCoeff(&predLabel);
            Index trueLabel;
            batch.Y.col(i).maxCoeff(&trueLabel);
            if (predLabel == trueLabel) {
                ++correct;
            }
        }
    }
    return {totalLoss / total, 100.0 * correct / total};
}

void printStats(const Stats& stats) {
    std::cout << "Test loss: " << std::fixed << std::setprecision(6)
              << stats.trainLoss << ", Accuracy: " << std::setprecision(2)
              << stats.testAccuracy << "%\n";
}

void globalTest() {
    std::cout << "Running global test...\n";
    Random rnd(42345);

    const std::filesystem::path trainImages = "../data/train-images-idx3-ubyte";
    const std::filesystem::path trainLabels = "../data/train-labels-idx1-ubyte";
    const std::filesystem::path testImages = "../data/t10k-images-idx3-ubyte";
    const std::filesystem::path testLabels = "../data/t10k-labels-idx1-ubyte";

    assert(std::filesystem::exists(trainImages));
    assert(std::filesystem::exists(trainLabels));
    auto trainLoader =
        DataLoader::makeMnistLoader(trainImages, trainLabels, 64, rnd);
    assert(trainLoader.has_value());

    assert(std::filesystem::exists(testImages));
    assert(std::filesystem::exists(testLabels));
    auto testLoader =
        DataLoader::makeMnistLoader(testImages, testLabels, 64, rnd);
    assert(testLoader.has_value());

    std::cout << "Loaded " << trainLoader->rewind() << " training and "
              << testLoader->rewind() << " test samples\n";

    NeuralNetwork model = makeModel1(rnd);
    // NeuralNetwork model = makeModel2(rnd);
    // NeuralNetwork model = makeModel3(rnd);

    auto statsSGD = trainModelAlgo1(model, *trainLoader, *testLoader);
    std::cout << "SGD: ";
    printStats(statsSGD);

    LossFunction lossFunc = LossFunction::CrossEntropy();
    Optimizer opt = Optimizer(Adam(0.001, 0.9, 0.999, 1e-8));
    double learningRate = 0.001;
    Train trainer(lossFunc, std::move(opt), learningRate);
    std::cout << "Training for 10 epochs with Adam...\n";
    trainer.fit(model, *trainLoader, *testLoader, static_cast<Epoches>(10),
                Shuffle::Enable);

    double totalLoss = 0.0;
    int correct = 0;
    int total = 0;
    testLoader->rewind();
    while (testLoader->hasNext()) {
        auto batch = testLoader->nextBatch();
        total += batch.X.cols();
        Matrix preds = model.forward(std::move(batch.X));
        totalLoss += lossFunc.computeLoss(preds, batch.Y) * batch.X.cols();
        for (int i = 0; i < preds.cols(); ++i) {
            Index predLabel;
            preds.col(i).maxCoeff(&predLabel);
            Index trueLabel;
            batch.Y.col(i).maxCoeff(&trueLabel);
            if (predLabel == trueLabel) {
                ++correct;
            }
        }
    }
    Stats statsAdam = {totalLoss / total, 100.0 * correct / total};
    std::cout << "Adam: ";
    printStats(statsAdam);

    assert(statsSGD.testAccuracy >= 80.0 || statsAdam.testAccuracy >= 80.0);
    std::cout << "Global test passed\n";
}

void run_all_tests() {
    std::cout << "Running all tests...\n";
    testDataLoader();
    testNeuralNetwork();
    globalTest();
    std::cout << "All tests passed!\n";
}

void testDataLoader() {
    std::cout << "Testing DataLoader...\n";
    Random rnd(42);
    const std::filesystem::path trainImages = "../data/train-images-idx3-ubyte";
    const std::filesystem::path trainLabels = "../data/train-labels-idx1-ubyte";
    const std::filesystem::path testImages = "../data/t10k-images-idx3-ubyte";
    const std::filesystem::path testLabels = "../data/t10k-labels-idx1-ubyte";

    assert(std::filesystem::exists(trainImages));
    assert(std::filesystem::exists(trainLabels));
    auto trainLoader =
        DataLoader::makeMnistLoader(trainImages, trainLabels, 64, rnd);
    assert(trainLoader.has_value());
    assert(trainLoader->hasNext());

    auto batch = trainLoader->nextBatch();
    assert(batch.X.cols() <= 64);
    assert(batch.X.rows() == 784);
    assert(batch.Y.cols() == batch.X.cols());
    assert(batch.Y.rows() == 10);

    size_t totalSamples = 0;
    trainLoader->rewind();
    while (trainLoader->hasNext()) {
        auto b = trainLoader->nextBatch();
        totalSamples += b.X.cols();
    }
    assert(totalSamples == 60000);

    auto invalidLoader = DataLoader::makeMnistLoader(
        "invalid-images.idx3-ubyte", "invalid-labels.idx1-ubyte", 64, rnd);
    assert(!invalidLoader.has_value());

    std::cout << "DataLoader tests passed\n";
}

void testNeuralNetwork() {
    std::cout << "Testing NeuralNetwork...\n";
    Random rnd(42);
    NeuralNetwork model = makeModel1(rnd);
    // NeuralNetwork model = makeModel2(rnd);
    // NeuralNetwork model = makeModel3(rnd);

    Matrix X(784, 2);
    X.setRandom();
    Matrix Y = model.forward(std::move(X));
    assert(Y.rows() == 10 && Y.cols() == 2);

    Matrix gradOutput(10, 2);
    gradOutput.setRandom();
    Optimizer opt = Optimizer(Adam(0.01, 0.9, 0.999, 1e-8));
    Matrix gradInput = model.backward(gradOutput, opt, 0.01);
    assert(gradInput.rows() == 784 && gradInput.cols() == 2);

    std::cout << "NeuralNetwork tests passed\n";
}

}  // namespace NeuralNetwork
