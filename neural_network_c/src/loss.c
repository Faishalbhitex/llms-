#include "loss.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>


Loss* create_loss(LossType type, float delta) {
    Loss* loss = (Loss*)malloc(sizeof(Loss));
    loss->type = type;
    loss->delta = delta;
    return loss;
}

float compute_loss(Loss* loss, float* predicted, float*  target, int size) {
    float total = 0.0f;
    switch(loss->type) {
        case MSE:
            for(int i = 0; i < size; i++) {
                float diff = predicted[i] - target[i];
                total += diff * diff;
            }
            return total / (2 * size);
        
        case CROSS_ENTROPY:
            for(int i = 0; i < size; i++) {
                float p = predicted[i];
                total += -target[i] * logf(p + EPSILON);
            }
            return total / size;
        
        case BINARY_CROSS_ENTROPY:
            for(int i = 0; i < size; i++) {
                float p = predicted[i];
                 total += -target[i] * logf(p + EPSILON) - (1 - target[i]) * logf(1 - p + EPSILON);
            }
            return total / size;
        
        case HUBER:
            for(int i = 0; i < size; i++) {
                float diff = fabs(predicted[i] - target[i]);
                if(diff <= loss->delta) {
                    total += 0.5 * diff * diff;
                } else {
                    total += loss->delta * (diff - 0.5 * loss->delta);
                }
            }
            return total / size;
        
        default:
            return 0.0f;
    }
}

void compute_loss_gradient(Loss* loss, float* predicted, float* target, float* gradient, int size) {
    switch(loss->type) {
        case MSE:
            for(int i = 0; i < size; i++) {
                gradient[i] = (predicted[i] - target[i]) / size;
            }
            break;
        
        case CROSS_ENTROPY:
            for(int i = 0; i < size; i++) {
                // Asumsi predicted sudah melalui softmax
                gradient[i] = (predicted[i] - target[i]) / size;
            }
            break;
        
        case BINARY_CROSS_ENTROPY:
            for(int i = 0; i < size; i++) {
                float p = predicted[i];
                gradient[i] = (-target[i] / (p + EPSILON) + (1 - target[i]) / (1 - p + EPSILON)) / size;
            }
            break;
        
        case HUBER:
            for(int i = 0; i < size; i++) {
                float diff = predicted[i] - target[i];
                if(fabs(diff) <= loss->delta) {
                    gradient[i] = diff / size;
                } else {
                    gradient[i] = (diff > 0 ? loss->delta : -loss->delta) / size;
                }
            }
            break;
    }
}

void free_loss(Loss* loss) {
    free(loss);
}