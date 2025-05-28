#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include "neunet.h"
#include <vector>
#include <algorithm>
#include "Random.h"

namespace NeuralNetwork
{
    struct Batch
    {
        std::vector<Matrix> inputs;
        std::vector<Vector> targets;
    };
    
    class DataLoader
    {
    private:
        std::vector<Matrix> data_;
        std::vector<Vector> labels_;
        std::vector<size_t> indices_;
        size_t batchSize_;
        size_t currentPosition_;
        Random& rnd_;

    public:
        DataLoader(const std::vector<Matrix>& data, const std::vector<Vector>& labels, size_t batchSize, Random& rnd);
        size_t reset();
        bool isNext() const;
        Batch nextBatch();
    };
    
    
} // namespace NeuralNetwork
#endif
