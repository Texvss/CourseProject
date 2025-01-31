#ifndef NON_LINEAR_LAYER_H
#define NON_LINEAR_LAYER_H

#include "ActivationFunction.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

class NonLinearLayer
{
    private:
        ActivationFunction activationFunction;
        MatrixXd inputStore;
        
    public:
        NonLinearLayer(const ActivationFunction& activationFunction);
        MatrixXd forward(const MatrixXd& input);
        MatrixXd backward(const MatrixXd& gradOutput);
};

#endif