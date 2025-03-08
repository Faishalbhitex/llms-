#include "optimizer.h"
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <stdlib.h>

void test_sgd() {
    Optimizer* opt = create_optimizer(SGD, 0.1f);

    // Inisialisi dummy parameter
    int rows = 2, cols = 2;
    float** weights = (float**)malloc(rows * sizeof(float*));
    float* biases = (float*)malloc(rows * sizeof(float));
    float** weight_grads = (float**)malloc(rows * sizeof(float*));
    float* bias_grads = (float*)malloc(rows * sizeof(float));

    for(int i = 0; i < rows; i++) {
        weights[i] = (float*)malloc(cols * sizeof(float));
        weight_grads[i] = (float*)malloc(cols * sizeof(float));
        for(int j = 0; j < cols; j++) {
            weights[i][j] = 1.0f;
            weight_grads[i][j] = 0.5f;
        }
        biases[i] = 0.5f;
        bias_grads[i] = 0.2f;
    }

    initialize_optimizer_parameters(opt, rows, cols);
    optimizer_update(opt, weights, biases, weight_grads, bias_grads, rows, cols);

    // Verifikasi SGD update
    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            assert(fabs(weights[i][j] - (1.0f - 0.1f * 0.5f)) < 1e-6);
        }
        assert(fabs(biases[i] - (0.5f - 0.1f * 0.2f)) < 1e-6);
    }

    // bebaskan
    free_optimizer(opt);
    for(int i = 0; i < rows; i++) {
        free(weights[i]);
        free(weight_grads[i]);
    }
    free(weights);
    free(weight_grads);
    free(biases);
    free(bias_grads);

    printf("SGD test passed!\n");
}

void test_sgd_momentum() {
    Optimizer* opt = create_optimizer(SGD, 0.1f);
    opt->momentum = 0.9f;

    int rows = 1, cols = 1;
    float** weights = (float**)malloc(rows * sizeof(float*));
    weights[0] = (float*)malloc(cols * sizeof(float));
    weights[0][0] = 1.0f;

    float* biases = (float*)malloc(rows * sizeof(float));
    biases[0] = 0.5f;

    float** weight_grads = (float**)malloc(rows * sizeof(float*));
    weight_grads[0] = (float*)malloc(cols * sizeof(float));
    weight_grads[0][0] = 0.5f;

    float* bias_grads = (float*)malloc(rows * sizeof(float));
    bias_grads[0] = 0.2f;

    initialize_optimizer_parameters(opt, rows, cols);

    // Update pertama
    optimizer_update(opt, weights, biases, weight_grads, bias_grads, rows, cols);
    
    // Setelah update pertama, velocity = -lr * grad = -0.1 * 0.5 = -0.05
    float velocity = -0.1f * 0.5f;
    float expected_weight = 1.0f + velocity; // 1.0 - 0.05 = 0.95
    assert(fabs(weights[0][0] - expected_weight) < 1e-6);

    // Update kedua
    optimizer_update(opt, weights, biases, weight_grads, bias_grads, rows, cols);
    
    // velocity baru = momentum * velocity_lama - lr * grad
    velocity = 0.9f * velocity - 0.1f * 0.5f;
    expected_weight += velocity;
    
    assert(fabs(weights[0][0] - expected_weight) < 1e-6);

    // Bebaskan
    free_optimizer(opt);
    free(weights[0]);
    free(weights);
    free(weight_grads[0]);
    free(weight_grads);
    free(biases);
    free(bias_grads);

    printf("SGD with momentum test passed!\n");
}

void test_adam() {
    Optimizer* opt = create_optimizer(ADAM, 0.001f);
    opt->beta1 = 0.9f;
    opt->beta2 = 0.999f;
    opt->epsilon = 1e-8f;

    int rows = 1, cols = 1;
    float** weights = (float**)malloc(rows * sizeof(float*));
    weights[0] = (float*)malloc(cols * sizeof(float));
    weights[0][0] = 1.0f;

    float* biases = (float*)malloc(rows * sizeof(float));
    biases[0] = 0.5f;

    float** weight_grads = (float**)malloc(rows * sizeof(float*));
    weight_grads[0] = (float*)malloc(cols * sizeof(float));
    weight_grads[0][0] = 0.5f;

    float* bias_grads = (float*)malloc(rows * sizeof(float));
    bias_grads[0] = 0.2f;

    initialize_optimizer_parameters(opt, rows, cols);

    // Update pertama
    optimizer_update(opt, weights, biases, weight_grads, bias_grads, rows, cols);
    float m_w = 0.9f * 0.0f + 0.1f * 0.5f;
    float v_w = 0.999f * 0.0f + 0.001f * 0.5f * 0.5f;
    float m_hat_w = m_w / (1 - 0.9f);
    float v_hat_w = v_w / (1 - 0.999f);
    float expected_weight = 1.0f - 0.001f * m_hat_w / (sqrtf(v_hat_w) + 1e-8f);

    float m_b = 0.9f * 0.0f + 0.1f * 0.2f;
    float v_b = 0.999f * 0.0f + 0.001f * 0.2f * 0.2f;
    float m_hat_b = m_b / (1 - 0.9f);
    float v_hat_b = v_b / (1 - 0.999f);
    float expected_bias = 0.5f - 0.001f * m_hat_b / (sqrtf(v_hat_b) + 1e-8f);

    assert(fabs(weights[0][0] - expected_weight) < 1e-6);
    assert(fabs(biases[0] - expected_bias) < 1e-6);

    // Bebaskan
    free_optimizer(opt);
    free(weights[0]);
    free(weights);
    free(weight_grads[0]);
    free(weight_grads);
    free(biases);
    free(bias_grads);

    printf("Adam test passed!\n");
}

void test_optimizer() {
    test_sgd();
    test_sgd_momentum();
    test_adam();
}