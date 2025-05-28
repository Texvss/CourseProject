#ifndef TEST_H
#define TEST_H

#include "Random.h"
#include "NeuralNetwork.h"
#include "LinearLayer.h"
#include "NonLinearLayer.h"
#include "ActivationFunction.h"
#include "LossFunction.h"
#include "DataLoader.h"
#include "Train.h"
#include "MNISTLoader.h"
#include "SGDOptimizer.h"
#include <iostream>
#include <cassert>
#include <utility>

namespace NeuralNetwork
{
    void test1();
    void testSGD();
    void testTrain();
    int globalTest();
} // namespace NeuralNetwork

#endif
