#ifndef TEST_H
#define TEST_H

#include <iostream>
#include "ActivationFunction.h"
#include "LinearLayer.h"
#include "LossFunction.h"
#include "neunet.h"
#include "NonLinearLayer.h"
#include "SGDOptimizer.h"

namespace NeuralNetwork
{
    void test1();
    void testSGD();
    void testTrain();
} // namespace NeuralNetwork

#endif
