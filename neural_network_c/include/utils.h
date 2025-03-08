#ifndef UTILS_H
#define UTILS_H

#include <stddef.h>

// Memory management
float* allocate_array(size_t size);
float** allocate_matrix(size_t rows, size_t cols);
void free_array(float* arr);
void free_matrix(float** matrix, size_t rows);

// Math utilities
float clip(float value, float min, float max);
void shuffle_indices(int* indices, int size);

#endif // UTILS_H