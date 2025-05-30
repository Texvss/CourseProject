#include "Random.h"
#include <EigenRand/EigenRand>

namespace NeuralNetwork {

Random::Random(int seed) : generator_(seed) {}

Matrix Random::uniformMatrix(Index rows, Index cols, double a, double b) {
        return Eigen::Rand::uniformReal<Matrix>(rows, cols, generator_, a, b);
}

Vector Random::uniformVector(Index rows, double a, double b){
    Matrix result = Eigen::Rand::uniformReal<Matrix>(rows, 1, generator_, a, b);
    return result.col(0);
}

std::mt19937& Random::engine() {
    return generator_;
}
}  // namespace NeuralNetwork
