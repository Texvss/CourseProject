#include "NeuralNetwork.h"

namespace NeuralNetwork {
Matrix NeuralNetwork::forward(const Matrix&& X) {
    Matrix out = std::move(X);
    for (auto& layer : layers_) {
        out = layer.forward(std::move(out));
    }
    return out;
}

Matrix NeuralNetwork::backward(const Matrix& gradOutput, Optimizer& optimizer,
                               double learningRate) {
    Matrix grad = gradOutput;
    for (auto it = layers_.rbegin(); it != layers_.rend(); ++it) {
        grad = it->backward(grad, optimizer, learningRate);
    }
    return grad;
}

}  // namespace NeuralNetwork
