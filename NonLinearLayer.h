#ifndef NON_LINEAR_LAYER_H
#define NON_LINEAR_LAYER_H

#include "ActivationFunction.h"
#include "neunet.h"

namespace NeuralNetwork {
class NonLinearLayer {
public:
    struct Cache {
        Matrix input;
    };
    NonLinearLayer(ActivationFunction&& activateF);
    Matrix forward(const Matrix& input);
    Matrix backward(const Matrix& gradOutput);

private:
    std::unique_ptr<Cache> cache_;
    ActivationFunction activateF_;
    Matrix inputStore_;
};
}  // namespace NeuralNetwork
#endif
