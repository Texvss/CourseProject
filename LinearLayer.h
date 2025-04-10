#ifndef LINEAR_LAYER_H
#define LINEAR_LAYER_H

#include "neunet.h"
#include "Random.h"

namespace NeuralNetwork {

enum X : Index;
enum Y : Index;

class LinearLayer {
public:
    struct Cache {
        Matrix input;
    };

    void turn_on_learning_mod() {
        cache_ = std::make_unique<Cache>();
    }

    void turn_off_learning_mod() {
        cache_.reset();
    }

    LinearLayer(X x, Y y, Random& rnd = globalRandom());
    Matrix forward(const Matrix& input);
    Matrix backward(const Matrix& gradOutput, double learningSpeed);

private:
    Matrix weights_;
    Vector biases_;
    std::unique_ptr<Cache> cache_;
    
    static Random& globalRandom() {
        static Random rnd(42);
        return rnd;
    }

    static Matrix initializeMatrix(Index rows, Index cols, Random& rnd) {
        if (rows <= 0 || cols <= 0) {
            throw std::invalid_argument("Размеры матрицы должны быть положительными");
        }
        return rnd.uniformMatrix(rows, cols, -1, 1); // так оставлять константы тоже не очень хорошо, но я просто экономлю строчки кода
      }
      static Vector initializeVector(Index rows, Random& rnd) {
        if (rows <= 0) {
            throw std::invalid_argument("Размеры матрицы должны быть положительными");
        }
        return rnd.uniformVector(rows, -1, 1);
      }
};
}  // namespace NeuralNetwork
#endif
