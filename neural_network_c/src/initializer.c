#include "initializer.h"
#include "utils.h"
#include <math.h>
#include <stdlib.h>
#include <time.h>

Initializer* create_initializer(InitializerType type) {
    Initializer* init = (Initializer *)malloc(sizeof(Initializer));
    init->type = type;
    
    switch(type) {
        case RANDOM_UNIFORM:
            init->min_value = -0.05;
            init->max_value = 0.05f;
            break;
        case RANDOM_NORMAL:
            init->mean = 0.0f;
            init->std = 0.05f;
            break;
        default:
            break;
    }
    return init;
}

void initialize_weights(float** weights, int rows, int cols, Initializer* init) {
    switch(init->type) {
        case XAVIER: {
            float std = sqrtf(2.0f / (rows + cols));
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    weights[i][j] = (rand() / (float)RAND_MAX) * (2 * std) - std;
                }
            }
            break;
        }
        case HE: {
            float std = sqrtf(2.0f / rows);
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    weights[i][j] = (rand() / (float)RAND_MAX) * (2 * std) - std;
                }
            }
            break;
        }
        case RANDOM_UNIFORM:
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    weights[i][j] = init->min_value + (init->max_value - init->min_value) * (rand() / (float)RAND_MAX);
                }
            }
            break;
        case RANDOM_NORMAL: {
            // Box-Muller transform untuk distribusi normal
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    float u1 = (rand() + 1.0f) / ((float)RAND_MAX + 1.0f);
                    float u2 = (rand() + 1.0f) / ((float)RAND_MAX + 1.0f);
                    float z = sqrtf(-2.0f * logf(u1)) * cosf(2 * M_PI * u2);
                    weights[i][j] = init->mean + init->std * z;
                }
            }
            break;
        }
        case ONES:
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    weights[i][j] = 1.0f;
                }
            }
            break;
        case ZEROS:
            for (int i = 0; i < rows; i++) {
                memset(weights[i], 0, cols * sizeof(float));
            }
            break;
    }
}

void initialize_biases(float* biases, int size, Initializer* init) {
    switch(init->type) {
        case ZEROS:
            memset(biases, 0, size * sizeof(float));
            break;
        case ONES:
            for(int i = 0; i < size; i++) {
                biases[i] = 1.0f;
            }
            break;
        default:
            // Default inisialisasi kecil random
            for(int i = 0; i < size; i++) {
                biases[i] = 0.01f * rand() / (float)RAND_MAX;
            }
    }
}

void free_initializer(Initializer* init) {
    free(init);
}