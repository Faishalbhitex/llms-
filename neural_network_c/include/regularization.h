#ifndef REGULARIZATION_H
#define REGULARIZATION_H

typedef enum { L1, L2, DROPOUT, ELASTIC_NET } RegularizationType;

typedef struct {
    RegularizationType type;
    float lambda;       // L1/L2 strength
    float dropout_rate; // For dropout
    float* dropout_mask;
    int mask_size;
} Regularizer;

Regularizer* create_regularizer(RegularizationType type, float strength);
void apply_regularization(Regularizer* reg, float** weights, float* biases, int rows, int cols);
void compute_regularization_gradient(Regularizer* reg, float** weights, float** gradients, int rows, int cols);
void free_regularizer(Regularizer* reg);

#endif // REGULARIZATION_H