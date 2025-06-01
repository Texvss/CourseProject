#ifndef ANYLAYER_H
#define ANYLAYER_H

#include "neunet.h"
#include "Optimizer.h"
#include "NonLinearLayer.h"
#include <memory>
#include <type_traits>
#include <tuple>

namespace NeuralNetwork {

class ILayerInterface {
public:
    virtual ~ILayerInterface() = default;
    virtual Matrix forward(const Matrix& input) = 0;
    virtual Matrix computeGradients(const Matrix& gradOutput) = 0;
    virtual void applyParameterUpdate(Optimizer& opt, double learningRate, size_t layerIndex) = 0;
    virtual void turn_on_learning_mod() = 0;
    virtual void turn_off_learning_mod() = 0;
};

template <typename LayerT>
class LayerHolder : public ILayerInterface {
public:
    template <typename... Args>
    explicit LayerHolder(Args&&... args)
        : layer_(std::forward<Args>(args)...) {}

    Matrix forward(const Matrix& input) override {
        return layer_.forward(input);
    }

    Matrix computeGradients(const Matrix& gradOutput) override {
        auto [gradIn, dW, dB] = layer_.computeGradientsWithParams(gradOutput);
        lastGradWeights_ = std::move(dW);
        lastGradBiases_ = std::move(dB);
        return gradIn;
    }

    void applyParameterUpdate(Optimizer& opt, double learningRate, size_t layerIndex) override {
        opt.update(layer_.getWeight(), lastGradWeights_, layerIndex * 2, learningRate);
        opt.update(layer_.getBias(), lastGradBiases_, layerIndex * 2 + 1, learningRate);
    }

    void turn_on_learning_mod() override {
        layer_.turn_on_learning_mod();
    }

    void turn_off_learning_mod() override {
        layer_.turn_off_learning_mod();
    }

private:
    LayerT layer_;
    Matrix lastGradWeights_;
    Vector lastGradBiases_;
};

template <>
class LayerHolder<NonLinearLayer> : public ILayerInterface {
public:
    template <typename... Args>
    explicit LayerHolder(Args&&... args)
        : layer_(std::forward<Args>(args)...) {}

    Matrix forward(const Matrix& input) override {
        return layer_.forward(input);
    }

    Matrix computeGradients(const Matrix& gradOutput) override {
        return layer_.computeGradients(gradOutput);
    }

    void applyParameterUpdate(Optimizer&, double, size_t) override {
    }

    void turn_on_learning_mod() override {
        layer_.turn_on_learning_mod();
    }

    void turn_off_learning_mod() override {
        layer_.turn_off_learning_mod();
    }

private:
    NonLinearLayer layer_;
};

class CAnyLayer {
public:
    template <typename LayerT, typename... Args>
    explicit CAnyLayer(std::in_place_type_t<LayerT>, Args&&... args)
        : ptr_(std::make_unique<LayerHolder<LayerT>>(std::forward<Args>(args)...)) {}

    Matrix forward(const Matrix& input) {
        return ptr_->forward(input);
    }

    Matrix computeGradients(const Matrix& gradOutput) {
        return ptr_->computeGradients(gradOutput);
    }

    void applyParameterUpdate(Optimizer& opt, double learningRate, size_t layerIndex) {
        ptr_->applyParameterUpdate(opt, learningRate, layerIndex);
    }

    void turn_on_learning_mod() {
        ptr_->turn_on_learning_mod();
    }

    void turn_off_learning_mod() {
        ptr_->turn_off_learning_mod();
    }

private:
    std::unique_ptr<ILayerInterface> ptr_;
};

} // namespace NeuralNetwork

#endif
