#pragma once

#include <iostream>

#include "DataLoader.h"
#include "LossFunction.h"
#include "neunet.h"
#include "NeuralNetwork.h"

namespace NeuralNetwork {
enum class Shuffle { Disable, Enable };
enum Epoches : int;

struct EpochStatistics {
    int currentEpoch;
    double totalLoss;
    double accuracy;
};

class Train {
private:
    LossFunction loss_;
    Optimizer optimizer_;
    double learningRate_;

    double trainStep(NeuralNetwork& model, const Batch& batch);
    void progress(int currentEpoch, Epoches totalEpochs, double trainLoss,
                  double trainAccuracy, double testLoss,
                  double testAccuracy) const;
    EpochStatistics trainOneEpoch(NeuralNetwork& model, DataLoader& loader,
                                  int currentEpoch);
    EpochStatistics evaluate(NeuralNetwork& model, DataLoader& loader,
                             int currentEpoch);

public:
    Train(LossFunction loss, Optimizer optimizer, double learningRate);

    void fit(NeuralNetwork& model, DataLoader& loader, DataLoader& testLoader,
             Epoches epochs, Shuffle status = Shuffle::Enable);
};

}  // namespace NeuralNetwork
