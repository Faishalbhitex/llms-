#ifndef INITIALIZER_H
#define INITIALIZER_H

typedef enum { XAVIER, HE, RANDOM_UNIFORM, RANDOM_NORMAL, ZEROS, ONES } InitializerType;

typedef struct {
    InitializerType type;
    float min_value;  // For uniform distribution
    float max_value;  // For uniform distribution
    float mean;       // For normal distribution
    float std;        // For normal distribution
} Initializer;

Initializer* create_initializer(InitializerType type);
void initialize_weights(float** weights, int rows, int cols, Initializer* init);
void initialize_biases(float* biases, int size, Initializer* init);
void free_initializer(Initializer* init);

#endif // INITIALIZER_H