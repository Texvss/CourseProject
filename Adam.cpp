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
    void* param_ptr = static_cast<void*>(&tetta);
    auto& t = t_[param_ptr];
    auto& mom = moments_[param_ptr];

    if (t == 0) {
        mom.M = Matrix::Zero(tetta.rows(), tetta.cols());
        mom.V = Matrix::Zero(tetta.rows(), tetta.cols());
    }

    ++t;

    mom.M = beta1_ * mom.M + (1 - beta1_) * grad;
    mom.V = beta2_ * mom.V + (1 - beta2_) * grad.array().square().matrix();

    double corr1 = 1.0 - std::pow(beta1_, t);
    double corr2 = 1.0 - std::pow(beta2_, t);

    Matrix m_hat = mom.M / corr1;
    Matrix v_hat = mom.V / corr2;

    tetta -=
        alpha_ * (m_hat.array() / (v_hat.array().sqrt() + epsilon_)).matrix();
}

void Adam::update(Vector& tetta, const Vector& grad) {
    void* param_ptr = static_cast<void*>(&tetta);
    auto& t = t_[param_ptr];
    auto& mom = moments_[param_ptr];

    Matrix tettaMat = tetta;
    Matrix gradMat = grad;

    if (t == 0) {
        mom.M = Matrix::Zero(tettaMat.rows(), tettaMat.cols());
        mom.V = Matrix::Zero(tettaMat.rows(), tettaMat.cols());
    }

    ++t;

    mom.M = beta1_ * mom.M + (1 - beta1_) * gradMat;
    mom.V = beta2_ * mom.V + (1 - beta2_) * gradMat.array().square().matrix();

    double corr1 = 1.0 - std::pow(beta1_, t);
    double corr2 = 1.0 - std::pow(beta2_, t);

    Matrix m_hat = mom.M / corr1;
    Matrix v_hat = mom.V / corr2;

    tettaMat -=
        alpha_ * (m_hat.array() / (v_hat.array().sqrt() + epsilon_)).matrix();

    for (int i = 0; i < tetta.size(); ++i) {
        tetta[i] = tettaMat(i, 0);
    }
}
}  // namespace NeuralNetwork
