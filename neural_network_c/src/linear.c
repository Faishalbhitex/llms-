#include "linear.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

LinearLayer* create_linear_layer(int input_size, int output_size, Initializer* weight_init, Initializer* bias_init) {
    LinearLayer* layer = (LinearLayer *)malloc(sizeof(LinearLayer));
    layer->input_size = input_size;
    layer->output_size = output_size;
    layer->weight_initializer = weight_init;
    layer->bias_initializer = bias_init;
    
    // Alokasi memori untuk weights
    layer->weights = (float**)malloc(output_size * sizeof(float*));
    for(int i = 0; i < output_size; i++) {
        layer->weights[i] = (float*)malloc(input_size * sizeof(float));
    }
    initialize_weights(layer->weights, output_size, input_size, weight_init);
    
    // Alokasi memori untuk biases
    layer->biases = (float*)malloc(output_size * sizeof(float));
    initialize_biases(layer->biases, output_size, bias_init);
    
    // Alokasi memori untuk gradients
    layer->weight_gradients = (float**)malloc(output_size * sizeof(float*));
    for(int i = 0; i < output_size; i++) {
        layer->weight_gradients[i] = (float*)calloc(input_size, sizeof(float));
    }
    layer->bias_gradients = (float*)calloc(output_size, sizeof(float));
    
    return layer;
}

void free_linear_layer(LinearLayer* layer) {
    // Bebaskan memori weights
    for(int i = 0; i < layer->output_size; i++) {
        free(layer->weights[i]);
        free(layer->weight_gradients[i]);
    }
    free(layer->weights);
    free(layer->weight_gradients);
    
    // Bebaskan memori biases
    free(layer->biases);
    free(layer->bias_gradients);
    
    free(layer);
}

float* linear_forward(LinearLayer* layer, float* input, int batch_size) {
    float* output = (float*)malloc(batch_size * layer->output_size * sizeof(float));
    
    for(int b = 0; b < batch_size; b++) {
        for(int i = 0; i < layer->output_size; i++) {
            output[b * layer->output_size + i] = layer->biases[i];
            for(int j = 0; j < layer->input_size; j++) {
                output[b * layer->output_size + i] += input[b * layer->input_size + j] * layer->weights[i][j];
            }
        }
    }
    return output;
}

void linear_backward(LinearLayer* layer, float* grad_output, float* input, int batch_size, float learning_rate) {
    // Reset gradient ke nol
    for(int i = 0; i < layer->output_size; i++) {
        memset(layer->weight_gradients[i], 0, layer->input_size * sizeof(float));
    }
    memset(layer->bias_gradients, 0, layer->output_size * sizeof(float));
    
    // Kalkulasi gradient terhadap weights dan biases
    for(int b = 0; b < batch_size; b++) {
        for(int i = 0; i < layer->output_size; i++) {
            for(int j = 0; j < layer->input_size; j++) {
                layer->weight_gradients[i][j] += input[b * layer->input_size + j] * grad_output[b * layer->output_size + i];
            }
            layer->bias_gradients[i] += grad_output[b * layer->output_size + i];
        }
    }
    
    // Update weights dan biases
    for(int i = 0; i < layer->output_size; i++) {
        for(int j = 0; j < layer->input_size; j++) {
            layer->weights[i][j] -= learning_rate * layer->weight_gradients[i][j] / batch_size;
        }
        layer->biases[i] -= learning_rate * layer->bias_gradients[i] / batch_size;
    }
}

void save_linear_layer(LinearLayer* layer, const char* filename) {
    FILE* file = fopen(filename, "wb");
    if (file == NULL) {
        printf("Error opening file for writing.\n");
        return;
    }
    
    fwrite(&layer->input_size, sizeof(int), 1, file);
    fwrite(&layer->output_size, sizeof(int), 1, file);
    
    for(int i = 0; i < layer->output_size; i++) {
        fwrite(layer->weights[i], sizeof(float), layer->input_size, file);
    }
    fwrite(layer->biases, sizeof(float), layer->output_size, file);
    
    fclose(file);
}

LinearLayer* load_linear_layer(const char* filename) {
    FILE* file = fopen(filename, "rb");
    if(file == NULL) {
        printf("Error opening file for reading.\n");
        return NULL;
    }
    
    int input_size, output_size;
    fread(&input_size, sizeof(int), 1, file);
    fread(&output_size, sizeof(int), 1, file);
    
    // Dummy Initializer untuk menghindari NULL
    Initializer* dummy_weight = create_initializer(ZEROS);
    Initializer* dummy_bias = create_initializer(ZEROS);
    
    LinearLayer* layer = create_linear_layer(input_size, output_size, dummy_weight, dummy_bias);
    
    for(int i = 0; i < output_size; i++) {
        fread(layer->weights[i], sizeof(float), input_size, file);
    }
    fread(layer->biases, sizeof(float), output_size, file);
    
    // Bersihkan dummy initializer
    free_initializer(dummy_weight);
    free_initializer(dummy_bias);
    layer->weight_initializer = NULL;
    layer->bias_initializer = NULL;
    
    fclose(file);
    return layer;
}