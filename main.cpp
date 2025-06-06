#include "Except.h"
#include "NeuralNetwork.h"
#include "Test.h"

using namespace NeuralNetwork;

int main() {
    try {
        run_all_tests();
    } catch (...) {
        except::react();
    }
    return 0;
}
