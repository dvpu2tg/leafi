/******************************************************************************
 * Copyright (c) 2024, the LeaFi author(s).
 ******************************************************************************/

#ifndef ISAX_NEURAL_FILTER_H
#define ISAX_NEURAL_FILTER_H

#ifdef __cplusplus
extern "C" {
#endif

#include "globals.h"
#include "config.h"


static float bsf_global = VALUE_MAX;


typedef struct NeuralFilter {
    int id;
    bool is_activated;

    void *net;
    void *conformal_adjustor;
    bool is_trained;

    unsigned int series_length;
    int dim_instance;

    ID_TYPE num_global_query;
    VALUE_TYPE *global_queries_shared;
    VALUE_TYPE *global_node_distances;
    VALUE_TYPE *global_bsf_distances;
    VALUE_TYPE *global_nn_distances;
    VALUE_TYPE *global_pred_distances;

    ID_TYPE num_local_query;
    VALUE_TYPE *local_queries;
    VALUE_TYPE *local_nn_distances;
//    VALUE_TYPE *local_pred_distances;

    bool is_distance_squared;

    ID_TYPE node_size;
    VALUE_TYPE sax_prune_ratio;
    VALUE_TYPE estimated_gain;
} NeuralFilter;


void initGlobalVariables(Config const *config, unsigned int num_filters);
void initInferInputs(Config const *config, VALUE_TYPE *query_series2filter);

NeuralFilter *initFilterInfo(const Config *config, ID_TYPE node_size, int filter_id);
void addFilterTrainQuery(NeuralFilter *neural_filter, VALUE_TYPE const *filter_train_queries,
                         unsigned int global_query_size, unsigned int local_query_size);
void profileFilter(Config const *config, double *gpu_ms, VALUE_TYPE *gpu_mem_mb);

int trainNeuralFilter(Config const *config, NeuralFilter *filter, unsigned int stream_id, char *sax_str);
int logNeuralFilter(Config const *config, NeuralFilter *neural_filter, unsigned int stream_id, char *sax_str);

bool isFilterActive(NeuralFilter *neural_filter);
VALUE_TYPE inferFilter(NeuralFilter *neural_filter);
VALUE_TYPE checkNInferFilter(NeuralFilter *neural_filter);

void inferBatchNonblocking(NeuralFilter **filters, unsigned int num_filters, unsigned int num_streams,
                           unsigned int unroll_degree);
void inferBatchConcurrent(NeuralFilter **filters, unsigned int num_filters, unsigned int num_streams);
VALUE_TYPE *syncPredictionsNonblocking(int device_id);
VALUE_TYPE *inferBatchFuture(NeuralFilter **filters, unsigned int num_filters, unsigned int num_streams,
                             unsigned int unroll_degree, int device_id);
VALUE_TYPE *syncPredictionsConcurrent(unsigned int num_streams);

void dumpFilter(Config const *config, NeuralFilter *neural_filter, char *filter_dump_prefix);
void loadFilter(Config const *config, char *filter_load_prefix, NeuralFilter *filter);

void freeGlobalVariables();
void freeFilter(NeuralFilter *neural_filter);

#ifdef __cplusplus
}
#endif

#endif //ISAX_NEURAL_FILTER_H
