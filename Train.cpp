#include "Train.h"

namespace NeuralNetwork
{
    Train::Train(NeuralNetwork& model, LossFunction& loss, double learningRate) : model_(model), loss_(loss), learningRate_(learningRate) {}

     void Train::fit(DataLoader& loader, int epochs, bool shuffle)
    {
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
    size_t batchSize = batch.inputs.size();
    size_t features = batch.inputs[0].rows();
    
    Matrix X(features, batchSize);
    Matrix Y(batch.targets[0].size(), batchSize);

    for (size_t i = 0; i < batchSize; ++i) {
        X.col(i) = batch.inputs[i];
        Y.col(i) = batch.targets[i];
    }

    auto predictions = model_.forward(X);

    double lossValue = loss_.computeLoss(predictions, Y);

    auto grad = loss_.computeGrad(predictions, Y);
    model_.backward(grad, learningRate_);

    return lossValue;
}
} // namespace NeuralNetwork
