#pragma once

#include <algorithm>
#include <filesystem>
#include <vector>

#include "neunet.h"
#include "Random.h"

namespace NeuralNetwork {
struct Batch {
    Matrix X;
    Matrix Y;
};

struct MnistData {
    std::vector<Matrix> images;
    std::vector<Vector> labels;
};

class DataLoader {
private:
    std::vector<Matrix> data_;
    std::vector<Vector> labels_;
    std::vector<size_t> indices_;
    size_t batchSize_;
    size_t currentPosition_;
    Random& rnd_;

public:
    DataLoader(std::vector<Matrix>&& data, std::vector<Vector>&& labels,
               size_t batchSize, Random& rnd);

    size_t shuffle();
    size_t reset();
    bool hasNext() const;
    Batch nextBatch();
    static std::optional<MnistData> loadMnistData(
        const std::filesystem::path& imagesFile,
        const std::filesystem::path& labelsFile);

    static std::optional<DataLoader> makeMnistLoader(
        const std::filesystem::path& imagesFile,
        const std::filesystem::path& labelsFile, size_t batchSize, Random& rnd);
};

}  // namespace NeuralNetwork
