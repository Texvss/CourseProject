#pragma once

#include "neunet.h"

namespace NeuralNetwork {

class Adam {
    struct Momentum {
        Matrix M;
        Matrix V;
    };

private:
    double alpha_;
    double beta1_;
    double beta2_;
    double epsilon_;
    double beta1_t_;
    double beta2_t_;
    std::unordered_map<void*, Momentum> moments_;
    std::unordered_map<void*, size_t> t_;

public:
    Adam(double alpha, double beta1, double beta2, double epsilon);

    void update(Matrix& tetta, const Matrix& grad);
    void update(Vector& tetta, const Vector& grad);
};

}  // namespace NeuralNetwork
