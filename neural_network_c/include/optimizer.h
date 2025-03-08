#ifndef OPTIMIZER_H
#define OPTIMIZER_H

typedef enum { SGD, ADAM, RMSPROP, ADAGRAD } OptimizerType;

// Struct SGD dengan momentum
typedef struct {
    float** weight_velocities;
    float* bias_velocities;
} SGDParameters;

// Struct Adam
typedef struct {
    float** m_weights;
    float* m_biases;
    float** v_weights;
    float* v_biases;
} AdamParameters;

// Struct RMSprop
typedef struct {
    float** squared_grad_weights;
    float* squared_grad_biases;
} RMSpropParameters;

// Struct Adagrad
typedef struct {
    float** cache_weights;
    float* cache_biases;
} AdagradParameters;

typedef struct {
    OptimizerType type;
    float learning_rate;
    float beta1;        // For Adam
    float beta2;        // For Adam
    float epsilon;      // For Adam/RMSprop
    float momentum;     // For SGD with momentum
    float decay;        // Learning rate decay
    int t;             // Time step
    int rows, cols;
    void* parameters;   // Optimizer-specific parameters
} Optimizer;

Optimizer* create_optimizer(OptimizerType type, float learning_rate);
void initialize_optimizer_parameters(Optimizer* opt, int rows, int cols);
void optimizer_update(Optimizer* opt, float** weights, float* biases, float** weight_gradients, float* bias_gradients, int rows, int cols);
void free_optimizer(Optimizer* opt);

#endif // OPTIMIZER_H