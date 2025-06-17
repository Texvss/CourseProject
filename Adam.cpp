#include "Adam.h"

#include <cassert>
#include <cmath>

namespace NeuralNetwork {

Adam::Adam(double alpha, double beta1, double beta2, double epsilon)
    : alpha_(alpha), beta1_(beta1), beta2_(beta2), epsilon_(epsilon) {
    assert(alpha > 0 && "Learning rate must be positive");
    assert(beta1 > 0 && beta1 < 1 && "Beta1 must be in (0, 1)");
    assert(beta2 > 0 && beta2 < 1 && "Beta2 must be in (0, 1)");
    assert(epsilon > 0 && "Epsilon must be positive");
}

void Adam::update(Matrix& tetta, const Matrix& grad) {
    Matrix* paramPtr = &tetta;

    if (matrixT_.find(paramPtr) == matrixT_.end()) {
        matrixT_[paramPtr] = 0;
        matrixMoments_[paramPtr] = {Matrix::Zero(tetta.rows(), tetta.cols()),
                                    Matrix::Zero(tetta.rows(), tetta.cols())};
    }

    size_t& t = matrixT_[paramPtr];
    MatrixMomentum& mom = matrixMoments_[paramPtr];

    ++t;

    mom.M = beta1_ * mom.M + (1 - beta1_) * grad;
    mom.V = beta2_ * mom.V + (1 - beta2_) * grad.array().square().matrix();

    double corr1 = 1.0 - std::pow(beta1_, t);
    double corr2 = 1.0 - std::pow(beta2_, t);

    Matrix m_hat = mom.M / corr1;
    Matrix v_hat = mom.V / corr2;

    tetta.array() -=
        alpha_ * (m_hat.array() / (v_hat.array().sqrt() + epsilon_));
}

void Adam::update(Vector& tetta, const Vector& grad) {
    Vector* paramPtr = &tetta;

    if (vectorT_.find(paramPtr) == vectorT_.end()) {
        vectorT_[paramPtr] = 0;
        vectorMoments_[paramPtr] = {Vector::Zero(tetta.size()),
                                    Vector::Zero(tetta.size())};
    }

    size_t& t = vectorT_[paramPtr];
    VectorMomentum& mom = vectorMoments_[paramPtr];
    ++t;
    mom.M = beta1_ * mom.M + (1 - beta1_) * grad;
    mom.V = beta2_ * mom.V.array() + (1 - beta2_) * grad.array().square();

    double corr1 = 1.0 - std::pow(beta1_, t);
    double corr2 = 1.0 - std::pow(beta2_, t);

    Vector m_hat = mom.M / corr1;
    Vector v_hat = mom.V / corr2;

    auto denominator = v_hat.array().sqrt() + epsilon_;

    auto update_term = (m_hat.array() / denominator).eval();

    tetta.array() -= alpha_ * update_term;
}
}  // namespace NeuralNetwork
