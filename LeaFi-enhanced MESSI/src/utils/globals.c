/******************************************************************************
 * Copyright (c) 2024, the LeaFi author(s).
 ******************************************************************************/

#include "globals.h"


int clock_code;

ID_TYPE query_id_profiling = 0;
ID_TYPE leaf_counter_profiling = 0;
ID_TYPE sum2sax_counter_profiling = 0;
ID_TYPE l2square_counter_profiling = 0;

//ID node_nn_cnt = 0;

ID_TYPE neural_pruned_leaves = 0;
ID_TYPE neural_pruned_series = 0;
ID_TYPE neural_passed_leaves = 0;
ID_TYPE neural_passed_series = 0;
ID_TYPE neural_missed_leaves = 0;
ID_TYPE neural_missed_series = 0;

pthread_mutex_t *log_lock_profiling = NULL; // initialized in main.c

//unsigned int neural_filter_counter = 0;
unsigned int num_filters_learned = 0;

int log_model_size = 1;

size_t free_gpu_size = 0;
size_t total_gpu_size = 0;

__m256i M256I_1;
__m256i M256I_BREAKPOINTS_OFFSETS_0_7, M256I_BREAKPOINTS_OFFSETS_8_15;
__m256i M256I_BREAKPOINTS8_OFFSETS_0_7, M256I_BREAKPOINTS8_OFFSETS_8_15;
__m256i *M256I_OFFSETS_BY_SEGMENTS;

SAXMask root_mask;

VALUE_TYPE *split_range_cache_global = NULL;
ID_TYPE *split_segment_cache_global = NULL;

VALUE_TYPE EPSILON_GAP = 1e-4f;

int global_node_id = 0;
