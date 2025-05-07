#ifndef ANYLAYER_H
#define ANYLAYER_H

#include "neunet.h"

namespace NeuralNetwork {

    template<class TBase>
    class ILayerInterface : public TBase {
    public:
        virtual Matrix forward(const Matrix&) = 0;
        virtual Matrix backward(const Matrix&, double) = 0;
        virtual void turn_on_learning_mod() = 0;
        virtual void turn_off_learning_mod() = 0;
        virtual ~ILayerInterface() = default;
    };
    
    template<class TBase, class TLayer>
    class CLayerImpl : public TBase {
        using CBase = TBase;
    public:
        using CBase::CBase;
    
        Matrix forward(const Matrix& input) override {
            return this->Object().forward(input);
        }
    
        Matrix backward(const Matrix& gradOutput, double learningSpeed) override {
            return this->Object().backward(gradOutput, learningSpeed);
        }
    
        void turn_on_learning_mod() override {
            this->Object().turn_on_learning_mod();
        }
    
        void turn_off_learning_mod() override {
            this->Object().turn_off_learning_mod();
        }
    };
    
    class CAnyLayer : public NSLibrary::CAnyMovable<ILayerInterface, CLayerImpl> {
        using CBase = NSLibrary::CAnyMovable<ILayerInterface, CLayerImpl>;
    public:
        using CBase::CBase;
    
        Matrix forward(const Matrix& input) {
            return operator->()->forward(input);
        }
    
        Matrix backward(const Matrix& gradOutput, double learningSpeed) {
            return operator->()->backward(gradOutput, learningSpeed);
        }
    
        void turn_on_learning_mod() {
            operator->()->turn_on_learning_mod();
        }
    
        void turn_off_learning_mod() {
            operator->()->turn_off_learning_mod();
        }
    };
    
} // namespace NeuralNetwork

#endif
