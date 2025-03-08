#include "loss.h"
#include <stdio.h>
#include <assert.h>
#include <math.h>

void test_mse_loss() {
    Loss* loss = create_loss(MSE, 0);
    
    float predicted[] = {0.5f, 1.0f, 0.0f};
    float target[] = {1.0f, 1.0f, 0.0f};
    
    float loss_value = compute_loss(loss, predicted, target, 3);
    assert(fabs(loss_value - (0.25f + 0.0f + 0.0f)/2/3) < 1e-5);
    
    float gradient[3];
    compute_loss_gradient(loss, predicted, target, gradient, 3);
    assert(fabs(gradient[0] - (0.5f - 1.0f)/3) < 1e-5);
    assert(fabs(gradient[1] - (1.0f - 1.0f)/3) < 1e-5);
    assert(fabs(gradient[2] - (0.0f - 0.0f)/3) < 1e-5);
    
    free_loss(loss);
    printf("MSE loss test passed!.\n");
}

void test_cross_entropy_loss() {
    Loss* loss = create_loss(CROSS_ENTROPY, 0);
    
    float predicted[] = {0.7f, 0.2f, 0.1f}; // Asumsi sudah softmax
    float target[] = {1.0f, 0.0f, 0.0f};
    
    float loss_value = compute_loss(loss, predicted, target, 3);
    float expected = -logf(0.7f + EPSILON);
    assert(fabs(loss_value - expected/3) < 1e-5);
    
    float gradient[3];
    compute_loss_gradient(loss, predicted, target, gradient, 3);
    assert(fabs(gradient[0] - (0.7f - 1.0f)/3) < 1e-5);
    assert(fabs(gradient[1] - (0.2f - 0.0f)/3) < 1e-5);
    assert(fabs(gradient[2] - (0.1f - 0.0f)/3) < 1e-5);
    
    free_loss(loss);
    printf("Cross Entropy loss test passed!\n");    
}

void test_huber_loss() {
    Loss* loss = create_loss(HUBER, 1.0f);
    
    float predicted[] = {1.5f, 0.5f};
    float target[] = {1.0f, 1.0f};
    
    float loss_value = compute_loss(loss, predicted, target, 2);
    // (0.5*(0.5^2) + 1.0*(0.5 - 0.5*1.0)) / 2
    float expected = (0.5 * 0.25 + 0.5 * 0.25) / 2;
    assert(fabs(loss_value - expected) < 1e-5);
    
    float gradient[2];
    compute_loss_gradient(loss, predicted, target, gradient, 2);
    assert(fabs(gradient[0] - (1.5f - 1.0f)/2) < 1e-5);   // delta=1, error=0.5 < 1
    assert(fabs(gradient[1] - -0.5f/2) < 1e-5);  // error=0.5, grad= -0.5/2
    
    free_loss(loss);
    printf("Huber loss test passed!\n");
}

void test_loss() {
    test_mse_loss();
    test_cross_entropy_loss();
    test_huber_loss();
}