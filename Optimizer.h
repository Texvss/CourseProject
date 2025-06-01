#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "neunet.h"
#include <functional>
#include <vector>

namespace NeuralNetwork {

struct Momentum {
    Matrix M;
    Vector V;
};

class Optimizer {
public:
    using VectorUpdateFunc = std::function<void(Optimizer*, Vector&, const Vector&, double, size_t)>;
    using MatrixUpdateFunc = std::function<void(Optimizer*, Matrix&, const Matrix&, double, size_t)>;

    Optimizer(
        VectorUpdateFunc&& vectorUpdateFn,
        MatrixUpdateFunc&& matrixUpdateFn,
        double learningRate
    );

    static Optimizer SGD(double learningRate);
    static Optimizer Adam(double learningRate, double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8);

    void update(Vector& params, const Vector& grad, size_t index, double learningRate);
    void update(Matrix& params, const Matrix& grad, size_t index, double learningRate);

private:
    VectorUpdateFunc vectorUpdateFn_;
    MatrixUpdateFunc matrixUpdateFn_;
    double learningRate_;
    double beta1_;
    double beta2_;
    double epsilon_;
    size_t t_;
    std::vector<Momentum> moments_;
};

} // namespace NeuralNetwork

#endif
