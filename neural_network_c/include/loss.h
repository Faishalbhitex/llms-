#ifndef LOSS_H
#define LOSS_H

#define EPSILON 1e-8  // Untuk mencegah log(0)

typedef enum { MSE, CROSS_ENTROPY, BINARY_CROSS_ENTROPY, HUBER } LossType;

typedef struct {
    LossType type;
    float delta;  // For Huber loss
} Loss;

Loss* create_loss(LossType type, float delta);
float compute_loss(Loss* loss, float* predicted, float* target, int size);
void compute_loss_gradient(Loss* loss, float* predicted, float* target, float* gradient, int size);
void free_loss(Loss* loss);

#endif // LOSS_H