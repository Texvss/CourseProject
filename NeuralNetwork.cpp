#include "NeuralNetwork.h"

namespace NeuralNetwork
{
    Matrix NeuralNetwork::forward(const Matrix& X) {
        Matrix out = X;
        for (auto& layer : layers_) {
            out = layer.forward(out);
        }
        return out;
    }

    Matrix NeuralNetwork::backward(const Matrix& gradOutput, double learningRate) {
        Matrix grad = gradOutput;
        for (auto it = layers_.rbegin(); it != layers_.rend(); ++it) {
            grad = it->backward(grad, learningRate);
        }
        return grad;
    }
} // namespace NeuralNetwork
