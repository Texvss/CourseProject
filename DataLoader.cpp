#include "DataLoader.h"

#include <algorithm>
#include <cassert>
#include <fstream>

namespace NeuralNetwork {

DataLoader::DataLoader(std::vector<Matrix>&& data, std::vector<Vector>&& labels,
                       size_t batchSize, Random& rnd)
    : data_(std::move(data)),
      labels_(std::move(labels)),
      batchSize_(batchSize),
      currentPosition_(0),
      rnd_(rnd) {
    assert(data_.size() == labels_.size() &&
           "Data and labels must have the same size");
    assert(!data_.empty() && "Data must not be empty");
    assert(batchSize > 0 && "Batch size must be positive");

    indices_.resize(data_.size());
    for (size_t i = 0; i < indices_.size(); ++i) {
        indices_[i] = i;
    }
}

size_t DataLoader::shuffle() {
    rnd_.shuffle(indices_.begin(), indices_.end());
    currentPosition_ = 0;
    return data_.size();
}

size_t DataLoader::reset() {
    currentPosition_ = 0;
    return data_.size();
}

bool DataLoader::hasNext() const {
    return currentPosition_ < data_.size();
}

Batch DataLoader::nextBatch() {
    Batch batch;
    size_t endPosition = std::min(currentPosition_ + batchSize_, data_.size());
    size_t actualBS = endPosition - currentPosition_;
    size_t features = data_[0].rows();
    size_t outputDim = labels_[0].rows();

    Matrix X(features, actualBS);
    Matrix Y(outputDim, actualBS);

    for (size_t i = 0; i < actualBS; ++i) {
        size_t idx = indices_[currentPosition_ + i];
        X.col(i) = data_[idx];
        Y.col(i) = labels_[idx];
    }
    currentPosition_ = endPosition;
    return Batch{std::move(X), std::move(Y)};
}

static uint32_t readUint32(std::ifstream& stream) {
    uint32_t val = 0;
    stream.read(reinterpret_cast<char*>(&val), 4);
    return __builtin_bswap32(val);
}

std::optional<MnistData> DataLoader::loadMnistData(
    const std::filesystem::path& imagesFile,
    const std::filesystem::path& labelsFile) {
    std::ifstream img(imagesFile, std::ios::binary);
    std::ifstream lbl(labelsFile, std::ios::binary);

    if (!img.is_open() || !lbl.is_open()) {
        return std::nullopt;
    }

    uint32_t magicImg = readUint32(img);
    uint32_t numImgs = readUint32(img);
    uint32_t rows = readUint32(img);
    uint32_t cols = readUint32(img);

    uint32_t magicLbl = readUint32(lbl);
    uint32_t numLbls = readUint32(lbl);

    if (magicImg != 2051 || magicLbl != 2049 || numImgs != numLbls ||
        numImgs == 0) {
        return std::nullopt;
    }

    MnistData data;
    data.images.resize(numImgs);
    data.labels.resize(numImgs);

    for (uint32_t i = 0; i < numImgs; ++i) {
        Matrix m(rows * cols, 1);
        for (int j = 0; j < rows * cols; ++j) {
            unsigned char pixel;
            img.read(reinterpret_cast<char*>(&pixel), 1);
            if (!img) {
                return std::nullopt;
            }
            m(j, 0) = pixel / 255.0;
        }
        data.images[i] = std::move(m);

        unsigned char label;
        lbl.read(reinterpret_cast<char*>(&label), 1);
        if (!lbl) {
            return std::nullopt;
        }
        Vector v = Vector::Zero(10);
        v(label) = 1.0;
        data.labels[i] = std::move(v);
    }

    img.close();
    lbl.close();
    return data;
}

std::optional<DataLoader> DataLoader::makeMnistLoader(
    const std::filesystem::path& imagesFile,
    const std::filesystem::path& labelsFile, size_t batchSize, Random& rnd) {
    auto data = loadMnistData(imagesFile, labelsFile);
    if (!data) {
        return std::nullopt;
    }
    auto loader = DataLoader(std::move(data->images), std::move(data->labels),
                             batchSize, rnd);
    return loader;
}

}  // namespace NeuralNetwork
