#include "Train.h"

#include <cassert>

namespace NeuralNetwork {

Train::Train(LossFunction loss, Optimizer optimizer, double learningRate)
    : loss_(std::move(loss)),
      optimizer_(std::move(optimizer)),
      learningRate_(learningRate) {
}

double Train::trainStep(NeuralNetwork& model, const Batch& batch) {
    assert(batch.X.rows() > 0 && batch.Y.rows() > 0 &&
           batch.X.cols() == batch.Y.cols() &&
           "Invalid batch: empty or mismatched inputs/targets");

    auto predictions = model.forward(batch.X);
    double lossValue = loss_.computeLoss(predictions, batch.Y);
    auto grad = loss_.computeGrad(predictions, batch.Y);
    model.backward(grad, optimizer_, learningRate_);

    return lossValue;
}

void Train::progress(int currentEpoch, Epoches totalEpochs, double trainLoss,
                     double trainAccuracy, double testLoss,
                     double testAccuracy) const {
    std::cout << "Epoch " << currentEpoch << "/"
              << static_cast<int>(totalEpochs)
              << " - Train: Loss = " << trainLoss
              << ", Accuracy = " << trainAccuracy * 100 << "%"
              << " | Test: Loss = " << testLoss
              << ", Accuracy = " << testAccuracy * 100 << "%\n";
}

EpochStatistics Train::trainOneEpoch(NeuralNetwork& model, DataLoader& loader,
                                     int currentEpoch) {
    double totalLoss = 0.0;
    size_t batchCount = 0;

    while (loader.hasNext()) {
        Batch batch = loader.nextBatch();
        totalLoss += trainStep(model, batch);
        ++batchCount;
    }

    return EpochStatistics{currentEpoch,
                           batchCount > 0 ? totalLoss / batchCount : -1.0, 0.0};
}

EpochStatistics Train::evaluate(NeuralNetwork& model, DataLoader& loader,
                                int currentEpoch) {
    double totalLoss = 0.0;
    double totalAccuracy = 0.0;
    size_t batchCount = 0;

    loader.reset();
    while (loader.hasNext()) {
        Batch batch = loader.nextBatch();
        auto predictions = model.forward(batch.X);
        totalLoss += loss_.computeLoss(predictions, batch.Y);

        size_t correct = 0;
        for (Eigen::Index i = 0; i < predictions.cols(); ++i) {
            Eigen::Index predClass;
            Eigen::Index trueClass;
            predictions.col(i).maxCoeff(&predClass);
            batch.Y.col(i).maxCoeff(&trueClass);
            if (predClass == trueClass) {
                ++correct;
            }
        }
        totalAccuracy += static_cast<double>(correct) / predictions.cols();
        ++batchCount;
    }

    return EpochStatistics{currentEpoch,
                           batchCount > 0 ? totalLoss / batchCount : -1.0,
                           batchCount > 0 ? totalAccuracy / batchCount : 0.0};
}

void Train::fit(NeuralNetwork& model, DataLoader& loader,
                DataLoader& testLoader, Epoches epochs, Shuffle status) {
    for (int ep = 0; ep < epochs; ++ep) {
        if (status == Shuffle::Enable) {
            loader.shuffle();
        } else {
            loader.reset();
        }

        EpochStatistics trainStats = trainOneEpoch(model, loader, ep + 1);
        EpochStatistics testStats = evaluate(model, testLoader, ep + 1);

        EpochStatistics stats = trainOneEpoch(model, loader, ep + 1);
        if (trainStats.totalLoss >= 0 && testStats.totalLoss >= 0) {
            progress(trainStats.currentEpoch, epochs, trainStats.totalLoss,
                     trainStats.accuracy, testStats.totalLoss,
                     testStats.accuracy);
        } else {
            std::cerr << "Epoch " << stats.currentEpoch << "/"
                      << static_cast<int>(epochs)
                      << " - No batches to process\n";
        }
    }
}
}  // namespace NeuralNetwork
