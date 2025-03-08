#include "activation.h"
#include <stdio.h>
#include <assert.h>
#include <math.h>

#define FLOAT_EQ(a, b) (fabs((a) - (b)) < 1e-6)

void test_sigmoid() {
    assert(FLOAT_EQ(sigmoid(0.0f), 0.5f));
    assert(sigmoid(100.0f) > 0.999f);
    assert(sigmoid(-100.0f) < 0.001f);
    printf("Sigmoid test passed!\n");
}

void test_relu() {
    assert(FLOAT_EQ(relu(5.0f), 5.0f));
    assert(FLOAT_EQ(relu(-5.0f), 0.0f));
    assert(FLOAT_EQ(relu_df(5.0f), 1.0f));
    assert(FLOAT_EQ(relu_df(-5.0f), 0.0f));
    printf("ReLU tests passed!\n");
}

void test_softmax() {
    float input[3] = {1.0f, 2.0f, 3.0f};
    float output[3];
    softmax(input, output, 3);
    
    float sum = output[0] + output[1] + output[2];
    assert(FLOAT_EQ(sum, 1.0f));
    assert(output[2] > output[1] && output[1] > output[0]);
    printf("Softmax tests passed!\n");
}