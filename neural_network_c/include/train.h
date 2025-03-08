#ifndef TRAIN_H
#define TRAIN_H

#include "sequential.h"
#include "batch.h"

typedef struct {
    float** train_data;
    float** train_labels;
    float** val_data;
    float** val_labels;
    int train_size;
    int val_size;
    int input_size;
    int output_size;
} Dataset;

typedef struct {
    int epochs;
    int batch_size;
    float early_stopping_patience;
    float early_stopping_min_delta;
    int verbose;
    int shuffle;
} TrainingConfig;

Dataset* load_dataset(const char* filename);
void free_dataset(Dataset* dataset);
void train_model(SequentialModel* model, Dataset* dataset, TrainingConfig config);
float evaluate_model(SequentialModel* model, Dataset* dataset);
float* predict(SequentialModel* model, float* input);

#endif // TRAIN_H