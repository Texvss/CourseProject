#pragma once

#include "LinearLayer.h"
#include "neunet.h"
#include "NonLinearLayer.h"
#include "Optimizer.h"

namespace NeuralNetwork {
namespace detail {

class ILayerInterface {
public:
    virtual ~ILayerInterface() = default;
    virtual Matrix forward(const Matrix&& input) = 0;
    virtual Matrix backward(const Matrix& gradOutput, Optimizer& opt,
                            double learningRate) = 0;
    virtual void turn_on_learning_mod() = 0;
    virtual void turn_off_learning_mod() = 0;
};

template <typename LayerT>
class LayerHolder : public ILayerInterface {
private:
    LayerT layer_;

public:
    template <typename... Args>
    explicit LayerHolder(Args&&... args) : layer_(std::forward<Args>(args)...) {
    }

    Matrix forward(const Matrix&& input) override {
        return layer_.forward(std::move(input));
    }

    Matrix backward(const Matrix& gradOutput, Optimizer& opt,
                    double learningRate) override {
        return layer_.backward(gradOutput, opt, learningRate);
    }

    void turn_on_learning_mod() override {
        layer_.turn_on_learning_mod();
    }

    void turn_off_learning_mod() override {
        layer_.turn_off_learning_mod();
    }
};

template <>
class LayerHolder<NonLinearLayer> : public ILayerInterface {
private:
    NonLinearLayer layer_;

public:
    template <typename... Args>
    explicit LayerHolder(Args&&... args) : layer_(std::forward<Args>(args)...) {
    }

    Matrix forward(const Matrix&& input) override {
        return layer_.forward(std::move(input));
    }

    Matrix backward(const Matrix& gradOutput, Optimizer&, double) override {
        return layer_.backward(gradOutput);
    }

    void turn_on_learning_mod() override {
        layer_.turn_on_learning_mod();
    }

    void turn_off_learning_mod() override {
        layer_.turn_off_learning_mod();
    }
};

}  // namespace detail

class CAnyLayer {
private:
    std::unique_ptr<detail::ILayerInterface> ptr_;

public:
    template <typename LayerT, typename... Args>
    explicit CAnyLayer(std::in_place_type_t<LayerT>, Args&&... args)
        : ptr_(std::make_unique<detail::LayerHolder<LayerT>>(
              std::forward<Args>(args)...)) {
    }

    Matrix forward(const Matrix&& input) {
        return ptr_->forward(std::move(input));
    }

    Matrix backward(const Matrix& gradOutput, Optimizer& opt,
                    double learningRate) {
        return ptr_->backward(gradOutput, opt, learningRate);
    }

    void turn_on_learning_mod() {
        ptr_->turn_on_learning_mod();
    }

    void turn_off_learning_mod() {
        ptr_->turn_off_learning_mod();
    }
};

}  // namespace NeuralNetwork
