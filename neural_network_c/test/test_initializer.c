#include "initializer.h"
#include "utils.h"
#include <stdio.h>
#include <assert.h>
#include <math.h>

void test_initializer() {
    // Test  Xavier initialization
    Initializer* xavier = create_initializer(XAVIER);
    float** weights = allocate_matrix(100, 50);
    initialize_weights(weights, 100, 50, xavier);
    
    // Verify mean ~0
    float sum = 0.0f;
    for(int i = 0; i < 100; i++) {
        for(int j = 0; j < 50; j++) {
            sum += weights[i][j];
        }
    }
    assert(fabs(sum / (100*50)) < 0.01f);
    
    // Test Zeros initialization
    Initializer* zeros = create_initializer(ZEROS);
    float* biases = allocate_array(100);
    initialize_biases(biases, 100, zeros);
    
    for(int i = 0; i < 100; i++) {
        assert(biases[i] == 0.0f);
    }
    
    printf("All initializer tests passed! \n");
    
    free_initializer(xavier);
    free_initializer(zeros);
    free_matrix(weights, 100);
    free_array(biases);
}