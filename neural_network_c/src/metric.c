#include "metric.h"
#include <stdlib.h>
#include <math.h>

Metric* create_metric(MetricType type) {
	Metric* metric = (Metric*)malloc(sizeof(Metric));
	metric->type = type;
	return metric;
}

float compute_metric(Metric* metric, float* predicted, float* target, int size) {
    switch(metric->type) {
        case ACCURACY:
            return compute_accuracy(predicted, target, size);
        case MSE:
            return compute_mse(predicted, target, size);
        case MAE:
            return compute_mae(predicted, target, size);
        default:
            return 0.0f;
    }
}

void free_metric(Metric* metric) {
    free(metric);
}

float compute_accuracy(float* predicted, float* target, int size) {
    int correct = 0;
    for(int i = 0; i < size; i++) {
        // Threshold 0.5 untuk klasifikasi biner
        if(fabs(predicted[i] - target[i]) < 0.5f){
            correct++;
        }
    }
    return (float)correct / size;
}

float compute_mse(float* predicted, float* target, int size) {
    float sum = 0.0f;
    for(int i = 0; i < size; i++) {
        float diff = predicted[i] - target[i];
        sum += diff * diff;
    }
    return sum / size;
}

float compute_mae(float* predicted, float* target, int size) {
    float sum = 0.0f;
    for(int i = 0; i < size; i++) {
        sum += fabs(predicted[i] - target[i]);
    }
    return sum / size;
}