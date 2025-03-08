#include "linear.h"
#include <stdio.h>
#include <assert.h>
#include <math.h>

void test_linear_layer_creation() {
    Initializer* weight_init = create_initializer(ZEROS);
    Initializer* bias_init = create_initializer(ONES);
    
    LinearLayer* layer = create_linear_layer(3, 2, weight_init, bias_init);
    
    assert(layer->input_size == 3);
    assert(layer->output_size == 2);
    
    for(int i = 0; i < 2; i++) {
        for(int j = 0; j < 3; j++) {
            assert(fabs(layer->weights[i][j] - 0.0f) < 1e-6);
        }
        assert(fabs(layer->biases[i] - 1.0f) < 1e-6);
    }
    
    free_linear_layer(layer);
    free_initializer(weight_init);
    free_initializer(bias_init);
    printf("Linear layer creation test passed!\n");
}

void test_linear_forward() {
    Initializer* weight_init = create_initializer(ONES);
    Initializer* bias_init = create_initializer(ZEROS);
    
    LinearLayer* layer = create_linear_layer(2, 2, weight_init, bias_init);
    
    float input[] = {1.0f, 2.0f};
    float* output = linear_forward(layer, input, 1);
    
    assert(output[0]  == 3.0f);
    assert(output[1] == 3.0f);
    
    free(output);
    free_linear_layer(layer);
    free_initializer(weight_init);
    free_initializer(bias_init);
    
    printf("Linear forward test passed!\n");
}

void test_linear_backward() {
    Initializer* weight_init = create_initializer(ONES);
    Initializer* bias_init = create_initializer(ZEROS);
    
    LinearLayer* layer = create_linear_layer(2, 2, weight_init, bias_init);
    
    float input[] = {1.0f, 2.0f};
    float grad_output[] = {0.5f, 0.5f};
    
    linear_backward(layer, grad_output, input, 1, 0.1f);
    
    assert(fabs(layer->weights[0][0] - 0.95f) < 1e-5);
    assert(fabs(layer->weights[0][1] - 0.9f) < 1e-5);
    assert(fabs(layer->weights[1][0] - 0.95f) < 1e-5);
    assert(fabs(layer->weights[1][1] - 0.9f) < 1e-5);
    
    assert(fabs(layer->biases[0] -  (-0.05f)) < 1e-5);
    assert(fabs(layer->biases[1] - (-0.05f)) < 1e-5);
    
    free_linear_layer(layer);
    free_initializer(weight_init);
    free_initializer(bias_init);
    
    printf("Linear backward test passed!\n");
}

void test_linear_save_load() {
    Initializer* weight_init = create_initializer(ONES);
    Initializer* bias_init = create_initializer(ZEROS);
    
    LinearLayer* layer = create_linear_layer(2, 2, weight_init, bias_init);
    
    save_linear_layer(layer, "linear_layer.bin");
    LinearLayer* loaded_layer = load_linear_layer("linear_layer.bin");
    
    // Pastikan tidak menggunkan initializer saat load
    assert(loaded_layer->weight_initializer == NULL);
    assert(loaded_layer->bias_initializer == NULL);
    
    assert(loaded_layer->input_size == 2);
    assert(loaded_layer->output_size == 2);
    
    for(int i = 0; i < 2; i++) {
        for(int j = 0; j < 2; j++) {
            assert(loaded_layer->weights[i][j] == 1.0f);
        }
        assert(loaded_layer->biases[i] == 0.0f);
    }
    
    free_linear_layer(layer);
    free_linear_layer(loaded_layer);
    free_initializer(weight_init);
    free_initializer(bias_init);
    
    printf("Linear save/load test passed!\n");
}

void test_linear() {
    test_linear_layer_creation();
    test_linear_forward();
    test_linear_backward();
    test_linear_save_load();
}