#include "batch.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>

Batch* create_batch(int batch_size, int input_size, int output_size) {
    Batch* batch = (Batch*)malloc(sizeof(Batch));
    batch->batch_size = batch_size;
    batch->input_size = input_size;
    batch->output_size = output_size;
    batch->current_index = 0;
    
    batch->data = (float**)malloc(batch_size * sizeof(float*));
    batch->labels = (float**)malloc(batch_size * sizeof(float*));
    
    for(int i = 0; i < batch_size; i++) {
        batch->data[i] = (float*)malloc(input_size * sizeof(float));
        batch->labels[i] = (float*)malloc(output_size * sizeof(float));
    }
    
    return batch;
}

void free_batch(Batch* batch) {
    for(int i = 0; i < batch->batch_size; i++) {
        free(batch->data[i]);
        free(batch->labels[i]);
    }
    free(batch->data);
    free(batch->labels);
    free(batch);
}

BatchIterator* create_batch_iterator(float** data, float** labels, int total_samples, int input_size, int output_size, int batch_size) {
    BatchIterator* iterator = (BatchIterator*)malloc(sizeof(BatchIterator));
    iterator->data = data;
    iterator->labels = labels;
    iterator->total_samples = total_samples;
    iterator->input_size = input_size;
    iterator->output_size = output_size;
    iterator->batch_size = batch_size;
    iterator->num_batches = (total_samples + batch_size - 1) / batch_size;
    iterator->current_batch = 0;
    
    return iterator;
}

void free_batch_iterator(BatchIterator* iterator) {
    // tidak free data dan labels karena milik caller
    free(iterator);
}

Batch* get_next_batch(BatchIterator* iterator) {
    if(iterator->current_batch >= iterator->num_batches) return NULL;
    
    int start_idx = iterator->current_batch * iterator->batch_size;
    int end_idx = start_idx + iterator->batch_size;
    if(end_idx > iterator->total_samples) end_idx = iterator->total_samples;
    
    Batch* batch = create_batch(end_idx - start_idx, iterator->input_size, iterator->output_size);
    
    for(int i = 0; i < batch->batch_size; i++) {
        int src_idx = start_idx + i;
        memcpy(batch->data[i], iterator->data[src_idx], iterator->input_size * sizeof(float));
        memcpy(batch->labels[i], iterator->labels[src_idx], iterator->output_size * sizeof(float));
    }
    
    iterator->current_batch++;
    return batch;
}

void swap_float_array(float** a, float** b) {
    float* temp = *a;
    *a = *b;
    *b = temp;
}

void shuffle_batch_iterator(BatchIterator* iterator) {
    // Fisher-Yates shuffle
    srand(time(NULL));
    for(int i = iterator->total_samples - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        swap_float_array(&iterator->data[i], &iterator->data[j]);
        swap_float_array(&iterator->labels[i], &iterator->labels[j]);
    }
    iterator->current_batch = 0;
}