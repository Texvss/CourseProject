#pragma once

#include <memory>

#include "Adam.h"
#include "neunet.h"
#include "SGD.h"

namespace NeuralNetwork {

namespace detail {

class IOptimizerInterface {
public:
    virtual ~IOptimizerInterface() = default;
    virtual void update(Matrix& params, const Matrix& grad) = 0;
    virtual void update(Vector& params, const Vector& grad) = 0;
};

template <typename OptT>
class OptimizerHolder final : public IOptimizerInterface {
public:
    explicit OptimizerHolder(OptT opt) : opt_(std::move(opt)) {
    }

    void update(Matrix& params, const Matrix& grad) override {
        opt_.update(params, grad);
    }

    void update(Vector& params, const Vector& grad) override {
        opt_.update(params, grad);
    }

private:
    OptT opt_;
};

}  // namespace detail

class Optimizer {
private:
    std::unique_ptr<detail::IOptimizerInterface> ptr_;

public:
    explicit Optimizer(SGD sgd)
        : ptr_(std::make_unique<detail::OptimizerHolder<SGD>>(std::move(sgd))) {
    }

    explicit Optimizer(Adam adam)
        : ptr_(std::make_unique<detail::OptimizerHolder<Adam>>(
              std::move(adam))) {
    }

    void update(Matrix& params, const Matrix& grad) {
        ptr_->update(params, grad);
    }

    void update(Vector& params, const Vector& grad) {
        ptr_->update(params, grad);
    }
};

}  // namespace NeuralNetwork
