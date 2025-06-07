#include "NeuralNetwork.h"

namespace NeuralNetwork {
Matrix NeuralNetwork::forward(const Matrix& X) {
    Matrix out = X;
    for (auto& layer : layers_) {
        out = layer.forward(out);
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

NeuralNetwork NeuralNetwork::makeModel1(Random& rnd) {
    NeuralNetwork model;
    model.addLayer<LinearLayer>(In(784), Out(256), rnd);
    model.addLayer<NonLinearLayer>(ActivationFunction::ReLU());
    model.addLayer<LinearLayer>(In(256), Out(128), rnd);
    model.addLayer<NonLinearLayer>(ActivationFunction::ReLU());
    model.addLayer<LinearLayer>(In(128), Out(10), rnd);
    model.addLayer<NonLinearLayer>(ActivationFunction::LeakyReLU(0.05));
    return model;
}

NeuralNetwork NeuralNetwork::makeModel2(Random& rnd) {
    NeuralNetwork model;
    model.addLayer<LinearLayer>(In(784), Out(128), rnd);
    model.addLayer<NonLinearLayer>(ActivationFunction::ReLU());
    model.addLayer<LinearLayer>(In(128), Out(10), rnd);
    model.addLayer<NonLinearLayer>(ActivationFunction::LeakyReLU(0.05));
    return model;
}

NeuralNetwork NeuralNetwork::makeModel3(Random& rnd) {
    NeuralNetwork model;
    model.addLayer<LinearLayer>(In(784), Out(512), rnd);
    model.addLayer<NonLinearLayer>(ActivationFunction::LeakyReLU(0.01));
    model.addLayer<LinearLayer>(In(512), Out(64), rnd);
    model.addLayer<NonLinearLayer>(ActivationFunction::Sigmoid());
    model.addLayer<LinearLayer>(In(64), Out(10), rnd);
    model.addLayer<NonLinearLayer>(ActivationFunction::ReLU());
    return model;
}
}  // namespace NeuralNetwork
