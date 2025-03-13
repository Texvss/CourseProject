#include "Random.h"

#include <EigenRand/EigenRand>

namespace NeuralNetwork {
Random::Random(int seed) : generator_(seed) {
}

Matrix Random::uniformMatrix(Index rows, Index cols, double a, double b) {
    Eigen::ArrayXXd randMat =
        Eigen::Rand::uniformReal<Eigen::ArrayXXd>(rows, cols, generator_, a, b);
    return randMat.matrix();
}
}  // namespace NeuralNetwork
