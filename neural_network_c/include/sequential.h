#ifndef SEQUENTIAL_H
#define SEQUENTIAL_H

#include "activation.h"
#include "linear.h"
#include "loss.h"
#include "optimizer.h"
#include "regularization.h"

typedef struct LayerNode {
    void* layer;
    char* layer_type;
    struct LayerNode* next;
} LayerNode;

typedef struct {
    LayerNode* first;
    LayerNode* last;
    int num_layers;
    Optimizer* optimizer;
    Loss* loss;
    Regularizer* regularizer;
} SequentialModel;

SequentialModel* create_sequential_model(Optimizer* opt, Loss* loss, Regularizer* reg);
void add_layer(SequentialModel* model, void* layer, const char* layer_type);
float* forward_pass(SequentialModel* model, float* input, int batch_size);
void backward_pass(SequentialModel* model, float* gradient, float* input, int batch_size);
void free_sequential_model(SequentialModel* model);
void save_model(SequentialModel* model, const char* filename);
SequentialModel* load_model(const char* filename);

#endif // SEQUENTIAL_H