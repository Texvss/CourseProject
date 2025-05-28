#ifndef ANYLAYER_H
#define ANYLAYER_H

#include "neunet.h"
#include <memory>
#include <type_traits>

namespace NeuralNetwork {

    class ILayerInterface {
    public:
        virtual ~ILayerInterface() = default;
        virtual Matrix forward(const Matrix& input) = 0;
        virtual Matrix backward(const Matrix& gradOutput, double learningSpeed) = 0;
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

        Matrix backward(const Matrix& gradOutput, double learningSpeed) override {
            // Если у слоя есть backward с learningSpeed
            if constexpr (std::is_invocable_v<decltype(&LayerT::backward), LayerT, const Matrix&, double>) {
                return layer_.backward(gradOutput, learningSpeed);
            } else {
                return layer_.backward(gradOutput);
            }
        }

        void turn_on_learning_mod() override {
            layer_.turn_on_learning_mod();
        }

        void turn_off_learning_mod() override {
            layer_.turn_off_learning_mod();
        }

    private:
        LayerT layer_;
    };

    // Универсальный слой с type-erasure
    class CAnyLayer {
    public:
        template <typename LayerT, typename... Args>
        explicit CAnyLayer(std::in_place_type_t<LayerT>, Args&&... args)
            : ptr_(std::make_unique<LayerHolder<LayerT>>(std::forward<Args>(args)...)) {}

        Matrix forward(const Matrix& input) {
            return ptr_->forward(input);
        }

        Matrix backward(const Matrix& gradOutput, double learningSpeed) {
            return ptr_->backward(gradOutput, learningSpeed);
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

#endif // ANYLAYER_H
