#ifndef SGD_OPTIMIZER_H
#define SGD_OPTIMIZER_H

#include "neunet.h"

namespace NeuralNetwork
{
    struct SGDOptimizer {
        double learningSpeed;
        explicit SGDOptimizer(double ls) : learningSpeed(ls) {}
        Matrix update(const Matrix& grad) const {
            return learningSpeed * grad;
        }
        Vector update(const Vector& grad) const {
            return learningSpeed * grad;
        }
    };

} // namespace 



#endif