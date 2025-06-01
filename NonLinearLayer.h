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

    void turn_on_learning_mod() {
        cache_ = std::make_unique<Cache>();
    }

    void turn_off_learning_mod() {
        cache_.reset();
    }
    NonLinearLayer(ActivationFunction&& activateF);
    Matrix forward(const Matrix& input);
    // Matrix backward(const Matrix& gradOutput);
    Matrix computeGradients(const Matrix& gradOutput);
private:
    std::unique_ptr<Cache> cache_;
    ActivationFunction activateF_;
    Matrix inputStore_;
};
}  // namespace NeuralNetwork
#endif
