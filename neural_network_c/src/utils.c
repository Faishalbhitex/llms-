#include "utils.h"
#include <stdlib.h>
#include <time.h>
#include <string.h>

// Memory managemenr
float* allocate_array(size_t size) {
    return (float*)malloc(size * sizeof(float));
}

float** allocate_matrix(size_t rows, size_t cols) {
    float** matrix = (float**)malloc(rows * sizeof(float*));
    for(size_t i = 0; i < rows; i++) {
        matrix[i] = allocate_array(cols);
    }
    return matrix;
}

void free_array(float* arr) {
    if(arr != NULL) free(arr);
}

void free_matrix(float** matrix, size_t rows) {
    if(matrix == NULL) return;
    for(size_t i = 0; i < rows; i++) {
        free_array(matrix[i]);
    }
    free(matrix);
}

// Math utilities
float clip(float value, float min, float max) {
    if(value < min) return min;
    if(value > max) return max;
    return value;
}

void shuffle_indices(int* indices, int size) {
    srand(time(NULL));
    for(int i = size-1; i > 0; i--) {
        int j = rand() % (i+1);
        int temp = indices[i];
        indices[i] = indices[j];
        indices[j] = temp;
        
    }
}