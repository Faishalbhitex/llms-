#include "optimizer.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>


Optimizer* create_optimizer(OptimizerType type, float learning_rate) {
    Optimizer* opt = (Optimizer*)malloc(sizeof(Optimizer));
    opt->type = type;
    opt->learning_rate = learning_rate;
    opt->beta1 = 0.9f;
    opt->beta2 = 0.999f;
    opt->epsilon = 1e-8;
    opt->momentum = 0.0f;
    opt->decay = 0.0f;
    opt->t = 0;
    opt->parameters = NULL;
    
    return opt;
}

void initialize_optimizer_parameters(Optimizer* opt, int rows, int cols) {
    opt->rows = rows;
    opt->cols = cols;
    switch(opt->type) {
        case SGD:
            if(opt->momentum > 0) {
                SGDParameters* params = (SGDParameters*)malloc(sizeof(SGDParameters));
                params->weight_velocities = (float**)malloc(rows * sizeof(float*));
                params->bias_velocities = (float*)malloc(rows * sizeof(float));
                for(int i = 0; i < rows; i++) {
                    params->weight_velocities[i] = (float*)calloc(cols, sizeof(float)); 
                }
                memset(params->bias_velocities, 0, rows * sizeof(float));
                opt->parameters = params;
            }
            break;
        
        case ADAM: {
            AdamParameters* params = (AdamParameters*)malloc(sizeof(AdamParameters));
            params->m_weights = (float**)malloc(rows * sizeof(float*));
            params->v_weights = (float**)malloc(rows * sizeof(float*));
            params->m_biases = (float*)malloc(rows * sizeof(float));
            params->v_biases = (float*)malloc(rows * sizeof(float));
            
            for(int i = 0; i < rows; i++) {
                params->m_weights[i] = (float*)calloc(cols, sizeof(float));
                params->v_weights[i] = (float*)calloc(cols, sizeof(float));
            }
            memset(params->m_biases, 0, rows * sizeof(float));
            memset(params->v_biases, 0, rows * sizeof(float));
            
            opt->parameters = params;
            break;
      }
    
    case RMSPROP: {
        RMSpropParameters* params = (RMSpropParameters*)malloc(sizeof(RMSpropParameters));
        params->squared_grad_weights = (float**)malloc(rows * sizeof(float*));
        params->squared_grad_biases = (float*)malloc(rows * sizeof(float));
        
        for(int i = 0; i < rows; i++) {
            params->squared_grad_weights[i] = (float*)calloc(cols, sizeof(float));
        }
        memset(params->squared_grad_biases, 0, rows * sizeof(float));
        
        opt->parameters = params;
        break;
    }
    
    case ADAGRAD: {
        AdagradParameters* params = (AdagradParameters*)malloc(sizeof(AdagradParameters));
        params->cache_weights = (float**)malloc(rows * sizeof(float*));
        params->cache_biases = (float*)malloc(rows * sizeof(float));
        
        for(int i = 0; i < rows; i++) {
            params->cache_weights[i] = (float*)calloc(cols, sizeof(float));
        }
        memset(params->cache_biases, 0, rows * sizeof(float));
        
        opt->parameters = params;
        break;
      }
   }
}

void optimizer_update(Optimizer* opt, float** weights, float* biases, float** weight_gradients, float* bias_gradients, int rows, int cols) {
    opt->t++;
    float lr = opt->learning_rate;
    
    // Aplikasikan learning rate decay
    if(opt->decay > 0) {
        lr = lr * (1.0f / (1.0f + opt->decay * opt->t));
    }
    
    switch(opt->type) {
        case SGD: {
            if(opt->momentum > 0) {
                SGDParameters* params = (SGDParameters*)opt->parameters;
                for(int i = 0; i < rows; i++) {
                    // Update weights
                    for(int j = 0; j < cols; j++) {
                        float new_velocity = opt->momentum * params->weight_velocities[i][j] - lr * weight_gradients[i][j];
                        params->weight_velocities[i][j] = new_velocity;
                        
                        weights[i][j] += new_velocity;
                    }
                    // Update biases
                    params->bias_velocities[i] = opt->momentum * params->bias_velocities[i] + lr * bias_gradients[i];
                    
                    biases[i] += lr * params->bias_velocities[i];
                }
            } else {
                // Vanilla SGD
                for(int i = 0; i < rows; i++) {
                    for(int j = 0; j < cols; j++) {
                        weights[i][j] -= lr * weight_gradients[i][j];
                    }
                    biases[i] -= lr * bias_gradients[i];
                }
           }
           break;
    }
    
    case ADAM: {
        AdamParameters* params = (AdamParameters*)opt->parameters;
        float beta1_t = powf(opt->beta1, opt->t);
        float beta2_t = powf(opt->beta2, opt->t);
        
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                // Update momen pertama (mean)
                params->m_weights[i][j] = opt->beta1 * params->m_weights[i][j] + (1 - opt->beta1) * weight_gradients[i][j];
                // Update momen kedua (variansi)
                params->v_weights[i][j] = opt->beta2 * params->v_weights[i][j] + (1 - opt->beta2) * weight_gradients[i][j] * weight_gradients[i][j];
                // Bias koreksi
                float m_hat = params->m_weights[i][j] / (1 - beta1_t);
                float v_hat = params->v_weights[i][j] / (1 - beta2_t);
                // Update weights
                weights[i][j] -= lr * m_hat / (sqrtf(v_hat) + opt->epsilon);
            }
            // Update biases
            params->m_biases[i] = opt->beta1 * params->m_biases[i] + (1 - opt->beta1) * bias_gradients[i];
            params->v_biases[i] = opt->beta2 * params->v_biases[i] + (1 - opt->beta2) * bias_gradients[i] * bias_gradients[i];
            float m_hat = params->m_biases[i] / (1 - beta1_t);
            float v_hat = params->v_biases[i] / (1 - beta2_t);
            biases[i] -= lr * m_hat / (sqrtf(v_hat) + opt->epsilon);
        }
        break;
    }
    
    case RMSPROP: {
        RMSpropParameters* params = (RMSpropParameters*)opt->parameters;
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                params->squared_grad_weights[i][j] = opt->beta1 * params->squared_grad_weights[i][j] + (1 - opt->beta1) * weight_gradients[i][j] * weight_gradients[i][j];
                weights[i][j] -= lr * weight_gradients[i][j] / (sqrtf(params->squared_grad_weights[i][j]) + opt->epsilon);
            }
            params->squared_grad_biases[i] = opt->beta1 * params->squared_grad_biases[i] + (1 - opt->beta1) * bias_gradients[i] * bias_gradients[i];
            biases[i] -= lr * bias_gradients[i] / (sqrtf(params->squared_grad_biases[i]) + opt->epsilon);
        }
        break;
    }
    
    case ADAGRAD: {
        AdagradParameters* params =(AdagradParameters*)opt->parameters;
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                params->cache_weights[i][j] += weight_gradients[i][j] * weight_gradients[i][j];
                weights[i][j] -= lr * weight_gradients[i][j] / (sqrtf(params->cache_weights[i][j]) + opt->epsilon);
            }
            params->cache_biases[i] += bias_gradients[i] * bias_gradients[i];
            biases[i] -= lr * bias_gradients[i] / (sqrtf(params->cache_biases[i]) + opt->epsilon);
        }
        break;
    }
    }
}

void free_optimizer(Optimizer* opt) {
    if (!opt) return;
    
    if(opt->parameters != NULL) {
        int rows = opt->rows;
        // int cols = opt->cols;
        switch(opt->type) {
            case SGD: {
                SGDParameters* params = (SGDParameters*)opt->parameters;
                for(int i = 0; i < rows; i++) {
                    free(params->weight_velocities[i]);
                }
                free(params->weight_velocities);
                free(params->bias_velocities);
                break;
            }
          case ADAM: {
              AdamParameters* params = (AdamParameters*)opt->parameters;
              for(int i = 0; i < rows; i++) {
                  free(params->m_weights[i]);
                  free(params->v_weights[i]);
              }
              free(params->m_weights);
              free(params->v_weights);
              free(params->m_biases);
              free(params->v_biases);
              break;
          }  
        case RMSPROP: {
            RMSpropParameters* params = (RMSpropParameters*)opt->parameters;
            for(int i = 0; i < rows; i++) {
                free(params->squared_grad_weights[i]);
            }
            free(params->squared_grad_weights);
            free(params->squared_grad_biases);
            break;
         }
       case ADAGRAD: {
           AdagradParameters* params = (AdagradParameters*)opt->parameters;
           for(int i = 0; i < rows; i++) {
               free(params->cache_weights[i]);
           }
           free(params->cache_weights);
           free(params->cache_biases);
           break;
         }
      }
      free(opt->parameters);
    }
    free(opt);
}