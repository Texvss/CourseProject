#include "DataLoader.h"

namespace NeuralNetwork
{
    DataLoader::DataLoader(const std::vector<Matrix>& data, const std::vector<Vector>& labels, size_t batchSize, Random& rnd) : data_(data), 
    labels_(labels),
    batchSize_(batchSize),
    currentPosition_(0),
    rnd_(rnd) 
    {
        indices_.resize(data_.size());
        for (size_t i = 0; i < indices_.size(); ++i) {
            indices_[i] = i;
        }
    }

    size_t DataLoader::reset() {
        std::shuffle(indices_.begin(), indices_.end(), rnd_.engine());
        currentPosition_ = 0;
        return data_.size();
    }

    bool DataLoader::isNext() const {
        return currentPosition_ < data_.size();
    }

    Batch DataLoader::nextBatch() {
        Batch batch;
        size_t endPosition = std::min(currentPosition_ + batchSize_, data_.size());
        size_t actualBS =endPosition - currentPosition_;

        batch.inputs.reserve(actualBS);
        batch.targets.reserve(actualBS);

    for (size_t i = currentPosition_; i < endPosition; ++i) {
        size_t idx = indices_[i];
        batch.inputs.push_back(data_[idx]);
        batch.targets.push_back(labels_[idx]);
    }

    currentPosition_ = endPosition;
    return batch;
    }

} // namespace NeuralNetwork

// actualBS - actual batch size
