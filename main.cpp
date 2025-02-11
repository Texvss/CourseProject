// #include <iostream>
// #include <vector>
// #include "NeuralNetwork.h"
// #include <Eigen/Dense>


// int main(int, char**)
// {
//     std::vector<double> inputsX(5);
//     std::vector<double> actualOutputs(5);
//     NeuralNetwork NN(0, 0, 0.01);
//     for (size_t i = 0; i < inputsX.size(); ++i)
//     {
//         double input;

//         std::cin >> input;
//         inputsX[i] = input;
//         double y = 100 * input + 1;
//         actualOutputs[i] = y;
//     }
//     std::cout << "Actual Outputs: ";
//     for (size_t i = 0; i < inputsX.size(); ++i)
//     {
//         std::cout << actualOutputs[i] << " ";
//     }
//     std::cout << '\n';

//     NN.Train(inputsX, actualOutputs);
//     std::cout << "Predictions: ";
//     std::vector<double> predictions = NN.Predict(inputsX);
//     for (size_t i = 0; i < inputsX.size(); ++i)
//     {
//         std::cout << predictions[i] << " ";
//     }
//     // std::srand(std::time(nullptr));
//     // int random_value = std::rand();
//     // std::cout << random_value;
// }


// // using Eigen::Matrix;
// // using Eigen::VectorXd;
// // using Eigen::Matrix3d;
// // using Eigen::Vector3d;
 
// // int main()
// // {
// // //     MatrixXd m = MatrixXd::Random(3,3);
// // //     m = (m + MatrixXd::Constant(3,3,1.2)) * 50;
// // //     std::cout << "m =" << std::endl << m << std::endl;
// // // //   VectorXd v(3);
// // //     MatrixXd v = MatrixXd::Random(3,3);
// // //      std::cout << "v =" << std::endl << v << std::endl;
// // // //   v << 1, 2, 3;
// // //     std::cout << "m * v =" << std::endl << m * v << std::endl;
// // }
 
 
// // int main()
// // {
// //   Matrix3d m = Matrix3d::Random();
// //   m = (m + Matrix3d::Constant(1.2)) * 50;
// //   std::cout << "m =" << std::endl << m << std::endl;
// //   Vector3d v(1,2,3);
  
// //   std::cout << "m * v =" << std::endl << m * v << std::endl;
// // }

#include <iostream>
#include <Eigen/Dense>
#include "ActivationFunction.h"
#include "LinearLayer.h"
#include "NonLinearLayer.h"
#include "LossFunction.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

enum InputSize : int;
enum OutputSize : int;

int main() {
    MatrixXd inputs(2, 3);
    inputs << 1.0, 2.0, 3.0,
              4.0, 5.0, 6.0;

    MatrixXd targets(2, 2);
    targets << 1.0, 0.0,
               0.0, 1.0;

    LinearLayer linearLayer(InputSize (3), OutputSize (2));

    NonLinearLayer nonLinearLayer(ActivationFunction::ReLU());

    LossFunction lossFunction = LossFunction::MSE();

    std::cout << "Прямой проход:" << '\n';
    MatrixXd linearOutput(2, 2);
    for (int i = 0; i < inputs.rows(); ++i) {
        linearOutput.row(i) = linearLayer.forward(inputs.row(i).transpose());
    }
    std::cout << "Результат после линейного слоя:\n" << linearOutput << '\n';

    MatrixXd activationOutput(2, 2);
    activationOutput = nonLinearLayer.forward(linearOutput);
    std::cout << "Результат после нелинейного слоя:\n" << activationOutput << '\n';

    double loss = lossFunction.computeLoss(activationOutput, targets);
    std::cout << "Значение функции потерь: " << loss << '\n';


    std::cout << "Обратный проход:" << '\n';
    MatrixXd gradLoss = lossFunction.computeGrad(activationOutput, targets);
    std::cout << "Градиент функции потерь:\n" << gradLoss << '\n';

    MatrixXd gradActivation = nonLinearLayer.backward(gradLoss);
    std::cout << "Градиент после нелинейного слоя:\n" << gradActivation << '\n';

    MatrixXd gradLinear(2, 3);
    for (int i = 0; i < gradActivation.rows(); ++i) {
        gradLinear.row(i) = linearLayer.backward(gradActivation.row(i).transpose()).transpose();
    }
    std::cout << "Градиент после линейного слоя:\n" << gradLinear << '\n';

    return 0;
}