#include "activation.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

// Sigmoid
float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

float sigmoid_df(float x) {
    float s = sigmoid(x);
    return s * (1 - s);
}

// ReLU
float relu(float x) {
    return x > 0 ? x : 0;
}

float relu_df(float x) {
    return x > 0 ? 1.0f : 0.0f;
}

// Leaky ReLU
float leaky_relu(float x, float alpha) {
    return x > 0 ? x : alpha * x;
}

float leaky_relu_df(float x, float alpha) {
    return x > 0 ? 1.0f : alpha;
}

// Tanh
float tanh_activation(float x) {
    return tanhf(x);
}

float tanh_activation_df(float x) {
    float t = tanh(x);
    return 1 - t * t;
}

// ELU
float elu(float x, float alpha) {
    return x > 0 ? x : alpha * (expf(x)  - 1);
}

float elu_df(float x, float alpha) {
    return x > 0 ? 1.0f : alpha * (expf(x));
}

// Softmax dengan stabilisasi numerik
void softmax(float* input, float* output, int size) {
    float max = input[0];
    for(int i = 0; i < size; i++) {
        if(input[i] > max) max = input[i];
    }
    
    float sum = 0.0f;
    for(int i = 0; i < size; i++) {
        output[i] = expf(input[i] - max);
        sum += output[i];
    }
    
    for(int i = 0; i < size; i++) {
        output[i] /= sum;
    }
}

void softmax_df(float* output, float* gradient, int size) {
    for(int i = 0; i < size; i++) {
        gradient[i] = output[i] * (1 - output[i]);
        for(int j = 0; j < size; j++) {
            if(i != j) {
                gradient[i] -= output[i] * output[j];
            }
        }
    }
}

// Manajemen layer
ActivationLayer* create_activation_layer(ActivationType type, float alpha) {
    ActivationLayer* layer = malloc(sizeof(ActivationLayer));
    layer->type = type;
    layer->alpha = alpha;
    return layer;
}

void free_activation_layer(ActivationLayer* layer) {
    return free(layer);
}