#include "metric.h"
#include <stdio.h>
#include <assert.h>
#include <math.h>

void test_accuracy() {
    float predicted[] = {0.7f, 0.3f, 0.6f};
    float target[] = {1.0f, 0.0f, 1.0f};
    float acc = compute_accuracy(predicted, target, 3);
    assert(fabs(acc - 1.0f) < 1e-6); // semua benar
    printf("Accuracy test passed!\n");
}

void test_mse() {
    float predicted[] = {0.5f, 1.0f, 0.0f};
    float target[] = {1.0f, 1.0f, 0.0f};
    float mse = compute_mse(predicted, target, 3);
    assert(fabs(mse - (1.0f / 12.0f)) < 1e-6);
    printf("MSE test passed!\n");
}

void test_mae() {
    float predicted[] = {0.5f, 1.5f, 0.0f};
    float target[] = {1.0f, 1.0f, 0.0f};
    float mae = compute_mae(predicted, target, 3);
    assert(fabs(mae - 0.333333f) < 1e-6);
    printf("MAE test passed!\n");
}

void test_metric() {
    test_accuracy();
    test_mse();
    test_mae();
}