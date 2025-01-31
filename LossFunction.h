#ifndef LOSS_FUNCTION_H
#define LOSS_FUNCTION_H

#include <Eigen/Dense>
#include <functional>
using Eigen::MatrixXd;
using Eigen::VectorXd;

class LossFunction
{
    private:
        std::function<double(const MatrixXd&, const MatrixXd&)> lossFn;
        std::function<MatrixXd(const MatrixXd&, const MatrixXd&)> gradFn;
    public:
        LossFunction(std::function<double(const MatrixXd&, const MatrixXd&)> lossFn, std::function<MatrixXd(const MatrixXd&, const MatrixXd&)> gradFn);
        static LossFunction MSE();
        double computeLoss(const MatrixXd& predictions, const MatrixXd& actualOut);
        MatrixXd computeGrad(const MatrixXd& predictions, const MatrixXd& actualOut);
};

#endif