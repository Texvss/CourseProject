#pragma once

#include "Adam.h"
#include "DataLoader.h"
#include "LossFunction.h"
#include "NeuralNetwork.h"
#include "SGD.h"
#include "Train.h"

namespace NeuralNetwork {

struct Stats {
    double trainLoss;
    double testAccuracy;
};

static NeuralNetwork makeModel1(Random rnd);
static NeuralNetwork makeModel2(Random rnd);
static NeuralNetwork makeModel3(Random rnd);

Stats trainModelAlgo1(NeuralNetwork& model, DataLoader& trainLoader,
                      DataLoader& testLoader);

void printStats(const Stats& stats);
void globalTest();
void run_all_tests();
void testDataLoader();
void testNeuralNetwork();

}  // namespace NeuralNetwork
