#pragma once

#include "ActivationFunction.h"
#include "Cache.h"
#include "neunet.h"

namespace NeuralNetwork {

class NonLinearLayer {
private:
    std::unique_ptr<Cache> cache_;
    ActivationFunction activateF_;

public:
    NonLinearLayer(ActivationFunction&& activateF);

    void turn_on_learning_mod();
    void turn_off_learning_mod();
    Matrix forward(const Matrix& input);
    Matrix backward(const Matrix& gradOutput);
};
}  // namespace NeuralNetwork
