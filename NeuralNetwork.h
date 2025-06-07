#pragma once

#include <utility>
#include <vector>

#include "AnyLayer.h"
#include "LossFunction.h"
#include "neunet.h"
#include "Random.h"

namespace NeuralNetwork {
class NeuralNetwork {
private:
    std::vector<CAnyLayer> layers_;

public:
    NeuralNetwork() = default;

    template <typename LayerT, typename... Args>
    void addLayer(Args&&... args) {
        layers_.emplace_back(std::in_place_type<LayerT>,
                             std::forward<Args>(args)...);
    }
    Matrix forward(const Matrix& X);
    Matrix backward(const Matrix& gradOutput, Optimizer& opt,
                    double learningRate);
    static NeuralNetwork makeModel1(Random& rnd);
    static NeuralNetwork makeModel2(Random& rnd);
    static NeuralNetwork makeModel3(Random& rnd);
};

}  // namespace  NeuralNetwork
