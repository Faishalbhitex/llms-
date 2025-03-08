#ifndef LINEAR_H
#define LINEAR_H

#include "initializer.h"

typedef struct {
    int input_size;
    int output_size;
    float** weights;
    float* biases;
    float** weight_gradients;
    float* bias_gradients;
    Initializer* weight_initializer;
    Initializer* bias_initializer;
} LinearLayer;

LinearLayer* create_linear_layer(int input_size, int output_size, Initializer* weight_init, Initializer* bias_init);
void free_linear_layer(LinearLayer* layer);
float* linear_forward(LinearLayer* layer, float* input, int batch_size);
void linear_backward(LinearLayer* layer, float* grad_output, float* input, int batch_size, float learning_rate);
void save_linear_layer(LinearLayer* layer, const char* filename);
LinearLayer* load_linear_layer(const char* filename);

#endif // LINEAR_H