#ifndef MNIST_LOADER_H
#define MNIST_LOADER_H

#include <string>
#include <vector>
#include "neunet.h"

namespace NeuralNetwork {

class MNISTLoader {
public:
    static bool load(const std::string& imageFile,
                     const std::string& labelFile,
                     std::vector<Matrix>& images,
                     std::vector<Vector>& labels);
};

} // namespace NeuralNetwork
#endif
