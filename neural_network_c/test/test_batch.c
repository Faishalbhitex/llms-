#include "batch.h"
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>

void test_batch_creation() {
    Batch* batch = create_batch(32, 784, 10);
    
    assert(batch->batch_size == 32);
    assert(batch->input_size == 784);
    assert(batch->output_size == 10);
    
    free_batch(batch);
    printf("Batch creation test passed!\n");
}

void test_batch_iterator() {
    // Membuat data dummy
    int total_samples = 100;
    int input_size = 2;
    int output_size = 1;
    float* data[100];
    float* labels[100];
    
    for(int i = 0; i < 100; i++) {
        data[i] = (float*)malloc(input_size * sizeof(float));
        labels[i] = (float*)malloc(output_size * sizeof(float));
        data[i][0] = i;
        data[i][1] = i*2;
        labels[i][0] = i*3;
    }
    
    BatchIterator* iterator = create_batch_iterator(data, labels, total_samples, input_size, output_size, 32);
    
    assert(iterator->num_batches == 4); // 100/32 = 3.125 → 4 batches
    assert(iterator->current_batch == 0);
    
    // Test batch pertama
    Batch* batch = get_next_batch(iterator);
    assert(batch->batch_size == 32);
    assert(batch->data[0][0] == 0);
    assert(batch->data[31][1] == 62); // 31*2 = 62
    
    // Test batch terakhir
    while(get_next_batch(iterator)) {} // Lewati sampai akhir
    iterator->current_batch = 3;
    batch = get_next_batch(iterator);
    assert(batch->batch_size == 4); // 100 - 3*32 = 4
    
    // Cleanup
    free_batch(batch);
    free_batch_iterator(iterator);
    for(int i = 0; i < 100; i++) {
        free(data[i]);
        free(labels[i]);
    }
    printf("Batch iterator test passed!\n");
}

void test_shuffle() {
    // Buat data dengan pola terurut
    int total_samples = 10;
    float* data[10];
    float* labels[10];
    for(int i = 0; i < 10; i++) {
        data[i] = (float*)malloc(sizeof(float));
        labels[i] = (float*)malloc(sizeof(float));
        *data[i] = (float)i;
        *labels[i] = (float)i;
    }
    
    BatchIterator* iterator = create_batch_iterator(data, labels, total_samples, 1, 1, 10);
    shuffle_batch_iterator(iterator);
    
    // Verifikasi shuffle
    int same_order = 1;
    for(int i = 0; i < 10; i++) {
        if(*iterator->data[i] != (float)i) {
            same_order = 0;
            break;
        }
    }
    assert(!same_order);
    
    // Cleanup
    free_batch_iterator(iterator);
    for(int i = 0; i < 10; i++) {
        free(data[i]);
        free(labels[i]);
    }
    printf("Shuffle test passed!\n");
}

void test_batch() {
    test_batch_creation();
    test_batch_iterator();
    test_shuffle();
}