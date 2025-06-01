#ifndef TRAIN_H
#define TRAIN_H


#include <iostream>
#include "NeuralNetwork.h"
#include "DataLoader.h"
#include "LossFunction.h"
#include "neunet.h"
#include "Optimizer.h"

namespace NeuralNetwork
{
    class Train
    {
    private:
            NeuralNetwork& model_;
            LossFunction& loss_;
            Optimizer& optimizer_;
            double learningRate_;          
            double trainStep(const Batch& batch);
    public:
        Train(NeuralNetwork& model, LossFunction& loss, Optimizer& optimizer, double learningRate);
        void fit(DataLoader& loader, int epochs, bool shuffle = true);
    };
    
} // namespace NeuralNetwork


#endif
