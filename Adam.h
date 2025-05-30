#ifndef ADAM_H
#define ADAM_H
 
#include "neunet.h"

namespace NeuralNetwork {
    
    class Adam {
        struct Momentum {
            Matrix M;
            Vector V;
        };

    private:
        double alpha_;
        double beta1_;
        double beta2_;
        double epsilon_;
        double beta1_t_;
        double beta2_t_;
        std::vector<Momentum> moments_;
        // std::vector<Momentum> v_;
        size_t t_;

    public:
        Adam(double alpha, double beta1, double beta2, double epsilon);
        void update(Matrix& tetta, const Matrix& grad);
        void update(Vector& tetta, const Vector& grad);
    };
    
} // namespace NeuralNetwork

#endif
