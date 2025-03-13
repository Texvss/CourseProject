#ifndef LINEAR_LAYER_H
#define LINEAR_LAYER_H

#include "neunet.h"

namespace NeuralNetwork {
enum Input : Index;
enum Output : Index;

class LinearLayer {
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

    LinearLayer(Input x, Output y);
    Matrix forward(const Matrix& input);
    Matrix backward(const Vector& gradOutput, double learningSpeed);

private:
    Matrix weights_;
    Vector biases_;
    std::unique_ptr<Cache> cache_;
};
}  // namespace NeuralNetwork
#endif
