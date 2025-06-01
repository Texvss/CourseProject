#include "Train.h"

namespace NeuralNetwork {

Train::Train(NeuralNetwork& model, LossFunction& loss, Optimizer& optimizer, double learningRate)
    : model_(model), loss_(loss), optimizer_(optimizer), learningRate_(learningRate) {}

void Train::fit(DataLoader& loader, int epochs, bool shuffle) {
    for (int ep = 1; ep <= epochs; ++ep) {
        if (shuffle) {
            loader.reset();
        }
        double totalLoss = 0.0;
        int batchCount = 0;

        while (loader.isNext()) {
            Batch batch = loader.nextBatch();
            totalLoss += trainStep(batch);
            ++batchCount;
        }
        std::cout << "Epoch " << ep << "/" << epochs
                      << " - Loss: " << (totalLoss / batchCount) << '\n';
    }
}

double Train::trainStep(const Batch& batch) {
    if (batch.inputs.empty() || batch.inputs.size() != batch.targets.size()) {
        throw std::runtime_error("Invalid batch: empty or mismatched inputs/targets");
    }

    size_t batchSize = batch.inputs.size();
    size_t features = batch.inputs[0].rows();
    size_t outputDim = batch.targets[0].rows();

    for (size_t i = 0; i < batchSize; ++i) {
        if (batch.inputs[i].rows() != 784 || batch.inputs[i].cols() != 1) {
            throw std::runtime_error("Invalid input dimensions");
        }
        if (batch.targets[i].rows() != 10 || batch.targets[i].cols() != 1) {
            throw std::runtime_error("Invalid target dimensions");
        }
    }

    Matrix X(features, batchSize);
    Matrix Y(outputDim, batchSize);

    for (size_t i = 0; i < batchSize; ++i) {
        X.col(i) = batch.inputs[i];
        Y.col(i) = batch.targets[i];
    }

    auto predictions = model_.forward(X);
    double lossValue = loss_.computeLoss(predictions, Y);
    auto grad = loss_.computeGrad(predictions, Y);
    model_.backward(grad, optimizer_, learningRate_);

    return lossValue;
}

} // namespace NeuralNetwork
