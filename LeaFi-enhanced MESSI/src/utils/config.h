/******************************************************************************
 * Copyright (c) 2024, the LeaFi author(s).
 ******************************************************************************/

#ifndef ISAX_CONFIG_H
#define ISAX_CONFIG_H

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <string.h>
#include <pthread.h>
#include <stdbool.h>
#include <assert.h>

#include "globals.h"
#include "clog.h"


typedef struct Config {
    char *database_filepath;
    char *database_summarization_filepath;
    char *query_filepath;
    char *query_summarization_filepath;
    char *query_bsf_distance_filepath;
    char *log_filepath;

    ID_TYPE database_size;
    unsigned int query_size;

    unsigned int series_length;
    unsigned int sax_length;
    unsigned int sax_cardinality;

    unsigned int initial_leaf_size;
    unsigned int leaf_size;

    VALUE_TYPE leaf_min_split_ratio;
    ID_TYPE leaf_min_split_size;

    bool use_adhoc_breakpoints;
    bool share_breakpoints;
    bool exact_search;
    bool sort_leaves;
    bool split_by_summarizations;
    bool split_by_sigma;

    unsigned int k; // kNN

    unsigned int cpu_cores;
    unsigned int numa_cores;
    unsigned int max_threads_index;
    unsigned int max_threads_train;
    unsigned int max_threads_query;
    unsigned int skipped_cores;
    unsigned int numa_id;

    unsigned int index_block_size;

    unsigned int series_limitations;

    VALUE_TYPE scale_factor;

    bool leaf_compactness;
    bool lower_bounding;
    bool log_leaf_only;
    bool with_id;

    bool require_neural_filter;

    char *filter_query_load_filepath;
    unsigned int filter_query_load_size;
    VALUE_TYPE filter_train_val_split;

    char *filter_global_queries_filepath;
    char *filter_local_queries_folderpath;
    VALUE_TYPE filter_synthetic_query_min_noise_level;
    VALUE_TYPE filter_synthetic_query_max_noise_level;
    ID_TYPE filter_num_synthetic_query_global;
    ID_TYPE filter_num_synthetic_query_local;
    ID_TYPE filter_num_synthetic_query_local_min;
    ID_TYPE filter_num_synthetic_query_local_max;
    VALUE_TYPE filter_excluding_quantile;
    VALUE_TYPE filter_density_comparable_margin;
    ID_TYPE filter_synthetic_query_in_leaf_patience;

    unsigned int dim_latent;
    VALUE_TYPE dropout_p;
    VALUE_TYPE leaky_relu_negative_slope;
    VALUE_TYPE learning_rate;
    VALUE_TYPE lr_min;
    unsigned int batch_size;
    unsigned int max_epochs;

    ID_TYPE filter_min_leaf_size;
    ID_TYPE filter_min_leaf_size_default;
    ID_TYPE filter_min_leaf_size_fixed;
    VALUE_TYPE filter_run_time_multiplier;

    bool log_leaf_visits;
    int device_id;

    bool on_disk;

    bool dump_index;
    bool dump_filters;
    char *index_dump_folderpath;
    char *nodes_dump_folderpath;
    char *data_dump_folderpath;
    char *filters_dump_folderpath;
    char *breakpoints_dump_filepath;

    bool load_index;
    char *index_load_folderpath;
    char *nodes_load_folderpath;
    char *data_load_folderpath;
    char *filters_load_folderpath;
    bool load_filters;
    char *breakpoints_load_filepath;

    char *dump_filename_postfix;

    VALUE_TYPE max_norm;
    VALUE_TYPE norm_type;

    bool is_filter_conformal;
    char *filter_conformal_core_type;
    VALUE_TYPE filter_conformal_recall;

    VALUE_TYPE filter_max_gpu_mem_mb;
    bool filter_allocate_is_gain;

    bool filter_conformal_is_smoothen;
    char *filter_conformal_smoothen_core;

    char *dump_model_filename_postfix;

    VALUE_TYPE filter_trial_confidence_level;
    ID_TYPE filter_trial_iterations;
    ID_TYPE filter_trial_nnode;
    ID_TYPE filter_trial_filter_preselection_size_threshold;

    ID_TYPE allocator_cpu_trial_iterations;

    bool to_profile;
    bool is_profile_exhaustive;
} Config;


Config *initializeConfig(int argc, char **argv);

void logConfig(Config const *config);

void freeConfig(Config *config);

#endif //ISAX_CONFIG_H
