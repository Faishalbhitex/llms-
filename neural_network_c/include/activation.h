#ifndef ACTIVATION_H
#define ACTIVATION_H

typedef enum { 
    SIGMOID,
    RELU, 
    LEAKY_RELU, 
    TANH, 
    SOFTMAX, 
    ELU 
} ActivationType;

typedef struct {
    ActivationType type;
    float alpha;  // For LeakyReLU and ELU
} ActivationLayer;

// Forward declarations
float sigmoid(float x);
float sigmoid_df(float x);

float relu(float x);
float relu_df(float x);

float leaky_relu(float x, float alpha);
float leaky_relu_df(float x, float alpha);

float tanh_activation(float x);
float tanh_activation_df(float x);

float elu(float x, float alpha);
float elu_df(float x, float alpha);

void softmax(float* input, float* output, int size);
void softmax_df(float* output, float* gradient, int size);

ActivationLayer* create_activation_layer(ActivationType type, float alpha);

void free_activation_layer(ActivationLayer* layer);

#endif // ACTIVATION_H