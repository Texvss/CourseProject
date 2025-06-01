// MNISTLoader.cpp
#include "MNISTLoader.h"
#include <fstream>
#include <iostream>

namespace NeuralNetwork {

static uint32_t readUint32(std::ifstream& stream) {
    uint32_t val = 0;
    stream.read(reinterpret_cast<char*>(&val), 4);
    return __builtin_bswap32(val);
}

bool MNISTLoader::load(const std::string& imageFile,
                       const std::string& labelFile,
                       std::vector<Matrix>& images,
                       std::vector<Vector>& labels)
{
    std::ifstream img(imageFile, std::ios::binary);
    std::ifstream lbl(labelFile, std::ios::binary);
    if (!img || !lbl) return false;

    uint32_t magicImg = readUint32(img);
    uint32_t numImgs = readUint32(img);
    uint32_t rows = readUint32(img);
    uint32_t cols = readUint32(img);

    uint32_t magicLbl = readUint32(lbl);
    uint32_t numLbls = readUint32(lbl);

    if (numImgs != numLbls || magicImg != 2051 || magicLbl != 2049) return false;

    images.clear();
    labels.clear();

    for (uint32_t i = 0; i < numImgs; ++i) {
        Matrix m(rows * cols, 1);
        for (int j = 0; j < rows * cols; ++j) {
            unsigned char pixel;
            img.read(reinterpret_cast<char*>(&pixel), 1);
            m(j, 0) = pixel / 255.0;
        }
        images.push_back(m);

        unsigned char label;
        lbl.read(reinterpret_cast<char*>(&label), 1);
        Vector v = Vector::Zero(10);
        v(label) = 1.0;
        labels.push_back(v);
    }

    return true;
}
} // namespace NeuralNetwork
