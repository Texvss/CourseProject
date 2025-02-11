#ifndef NON_LINEAR_LAYER_H
#define NON_LINEAR_LAYER_H

#include "ActivationFunction.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

class NonLinearLayer
{
    private:
        ActivationFunction activateF_;
        MatrixXd inputStore;
        
    public:
        NonLinearLayer(ActivationFunction&& activateF);
        MatrixXd forward(const MatrixXd& input);
        MatrixXd backward(const MatrixXd& gradOutput);
};

#endif