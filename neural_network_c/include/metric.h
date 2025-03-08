#ifndef METRIC_H
#define METRIC_H

//#include "loss.h"  // Jika ingin menggunakan MSE dari loss

typedef enum { 
    ACCURACY,
    MSE, 
    MAE 
 } MetricType;

typedef struct {
    MetricType type;
} Metric;

Metric* create_metric(MetricType type);
float compute_metric(Metric* metric, float* predicted, float* target, int size);
void free_metric(Metric* metric);

// Fungsi spesifik untuk masing-masing metric
float compute_accuracy(float* predicted, float* target, int size);
float compute_mse(float* predicted, float* target, int size);
float compute_mae(float* predicted, float* target, int size);

#endif // METRIC_H
