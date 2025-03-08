#include <stdio.h>

// 1. Test initializer
void test_initializer();

// 2. Test Activation
void test_sigmoid();
void test_relu();
void test_softmax();

// 3. Test Linear
void test_linear();

// 4. Test Loss
void test_loss();

// 5. Test Optimizer
void test_optimizer();

// 6. Test Batch
void test_batch();

// 7. Test Metric
void test_metric();

int main() {
    // 1
    test_initializer();
    
    // 2
    test_sigmoid();
    test_relu();
    test_softmax();
    
    // 3
    test_linear();
    
    // 4
    test_loss();
    
    // 5
    test_optimizer();
    
    // 6
    test_batch();
    
    // 7
    test_metric();

    printf("All tests passed!\n");
    return 0;
}