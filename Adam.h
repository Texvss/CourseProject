#pragma once

#include <map>

#include "neunet.h"

namespace NeuralNetwork {

class Adam {
    struct MatrixMomentum {
        Matrix M;
        Matrix V;
    };

    struct VectorMomentum {
        Vector M;
        Vector V;
    };

private:
    double alpha_;
    double beta1_;
    double beta2_;
    double epsilon_;
    double beta1_t_;
    double beta2_t_;
    std::map<Matrix*, size_t> matrixT_;
    std::map<Matrix*, MatrixMomentum> matrixMoments_;
    std::map<Vector*, size_t> vectorT_;
    std::map<Vector*, VectorMomentum> vectorMoments_;

public:
    Adam(double alpha, double beta1, double beta2, double epsilon);

    void update(Matrix& tetta, const Matrix& grad);
    void update(Vector& tetta, const Vector& grad);
};

}  // namespace NeuralNetwork
