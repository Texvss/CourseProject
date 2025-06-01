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

    Matrix NeuralNetwork::backward(const Matrix& gradOutput, Optimizer& optimizer, double learningRate) {
    Matrix grad = gradOutput;
    size_t layerIndex = 0;
    for (auto it = layers_.rbegin(); it != layers_.rend(); ++it) {
        grad = it->computeGradients(grad);
        it->applyParameterUpdate(optimizer, learningRate, layerIndex);
        ++layerIndex;
    }
    return grad;
    }
} // namespace NeuralNetwork
