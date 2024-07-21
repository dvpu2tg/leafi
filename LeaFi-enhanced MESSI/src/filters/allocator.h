/******************************************************************************
 * Copyright (c) 2024, the LeaFi author(s).
 ******************************************************************************/

#ifndef ISAX_ALLOCATOR_H
#define ISAX_ALLOCATOR_H

#ifdef __cplusplus
extern "C" {
#endif

#include "globals.h"
#include "config.h"

#include "neural_filter.h"


typedef struct FilterAllocator {
    NeuralFilter **filters;
    ID_TYPE num_filters;

    double cpu_ms_per_series;

    double gpu_ms;
    VALUE_TYPE gpu_mem_mb;
    VALUE_TYPE estimated_filter_pruning_ratio;

    ID_TYPE num_global_examples;
    ID_TYPE num_global_train_examples;

    VALUE_TYPE available_gpu_mem_mb;

    VALUE_TYPE *validation_recalls;
} FilterAllocator;


FilterAllocator *initAllocator(Config const *config, unsigned int num_filters);
//void updateAllocator(FilterAllocator *allocator, Config const *config);

void pushFilter(FilterAllocator *allocator, NeuralFilter *filter);

ID_TYPE selectNActivateFilters(FilterAllocator *allocator, Config const *config, bool is_trial);
ID_TYPE countActiveFilters(FilterAllocator *allocator);
void adjustErr4Recall(FilterAllocator *allocator, Config const *config);

void freeAllocator(FilterAllocator *allocator);

#ifdef __cplusplus
}
#endif

#endif //ISAX_ALLOCATOR_H
