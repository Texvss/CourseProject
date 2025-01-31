#include "LossFunction.h"

LossFunction::LossFunction(std::function<double(const MatrixXd&, const MatrixXd&)> lossFn, std::function<MatrixXd(const MatrixXd&, const MatrixXd&)> gradFn) : lossFn(lossFn), gradFn(gradFn) {}


LossFunction LossFunction::MSE()
{
    return LossFunction(
        [] (const MatrixXd& predictions, const MatrixXd& actualOutput) 
        {
            MatrixXd difference = actualOutput - predictions;
            return difference.array().square().mean();
        },
        [] (const MatrixXd& predictions, const MatrixXd& actualOutput) 
        {
            MatrixXd difference = actualOutput - predictions;
            return 2.0 * difference / predictions.size();          
        }
    );
}


double LossFunction::computeLoss(const MatrixXd& predictions, const MatrixXd& actualOut)
{
    return (actualOut - predictions).array().square().mean();
}

MatrixXd LossFunction::computeGrad(const MatrixXd& predictions, const MatrixXd& actualOut)
{
    return 2.0 * (predictions - actualOut) / predictions.size();
}