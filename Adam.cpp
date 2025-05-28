#include "Adam.h"
#include <cassert>
#include <cmath>

namespace NeuralNetwork {
    Adam::Adam(double alpha, double beta1, double beta2, double epsilon) : alpha_(alpha)
    , beta1_(beta1)
    , beta2_(beta2)
    , epsilon_(epsilon)
    , t_(0) 
    {
        assert(alpha > 0);
        assert(beta1 > 0 && beta1 < 1);
        assert(beta2 > 0 && beta2 < 1);
        assert(epsilon > 0);
    }

    void Adam::update(Matrix& tetta, const Matrix& grad){
        ++t_;
        if (moments_.size() <= t_) {
            moments_.resize(t_ + 1, {Matrix::Zero(tetta.rows(), tetta.cols()), Matrix::Zero(tetta.rows(), tetta.cols())});
        }

        moments_.at(t_).M = beta1_ * moments_.at(t_ - 1).M + (1 - beta1_) * grad;
        moments_.at(t_).V = beta2_ * moments_.at(t_ - 1).V + (1 - beta2_) * grad.array().square().matrix();

        double corr1 = 1.0 - std::pow(beta1_, t_);
        double corr2 = 1.0 - std::pow(beta2_, t_);

        Matrix m_hat = moments_.at(t_).M / corr1;
        Matrix v_hat = moments_.at(t_).V / corr2;
        
        tetta -= alpha_ * (m_hat.array() / (v_hat.array().sqrt() + epsilon_)).matrix();

    }

    void Adam::update(Vector& tetta, const Vector& grad) {
        ++t_;
        Matrix g = grad;
        if (moments_.size() <= t_) {
            moments_.resize(t_ + 1, {Matrix::Zero(g.rows(), g.cols()), Matrix::Zero(g.rows(), g.cols())});
        }

        moments_.at(t_).M = beta1_ * moments_.at(t_ - 1).M + (1 - beta1_) * g;
        moments_.at(t_).V = beta2_ * moments_.at(t_ - 1).V + (1 - beta2_) * g.array().square().matrix();

        double corr1 = 1.0 - std::pow(beta1_, t_);
        double corr2 = 1.0 - std::pow(beta2_, t_);

        Matrix m_hat = moments_.at(t_).M / corr1;
        Matrix v_hat = moments_.at(t_).V / corr2;

        for (int i = 0; i < tetta.size(); ++i) {
            tetta[i] -= alpha_ * (m_hat(i,0) / (std::sqrt(v_hat(i,0)) + epsilon_));
        }
    }
} // namespace NeuralNetwork 
