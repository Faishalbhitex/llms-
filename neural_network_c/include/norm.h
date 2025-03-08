#ifndef BATCH_H
#define BATCH_H

typedef struct {
    float** data;
    float** labels;
    int batch_size;
    int input_size;
    int output_size;
    int current_index;
} Batch;

typedef struct {
    float** data;
    float** labels;
    int total_samples;
    int input_size;
    int output_size;
    int batch_size;
    int num_batches;
    int current_batch;
} BatchIterator;

Batch* create_batch(int batch_size, int input_size, int output_size);
void free_batch(Batch* batch);
BatchIterator* create_batch_iterator(float** data, float** labels, int total_samples, int input_size, int output_size, int batch_size);
void free_batch_iterator(BatchIterator* iterator);
Batch* get_next_batch(BatchIterator* iterator);
void shuffle_batch_iterator(BatchIterator* iterator);

#endif // BATCH_H