/******************************************************************************
 * Copyright (c) 2024, the LeaFi author(s).
 ******************************************************************************/

#include "config.h"

#include <sys/stat.h>
#include <sys/types.h>
#include <libgen.h>

#include "str.h"


static void _mkdir(const char *dir) {
    char tmp[PATH_MAX];
    char *p = NULL;
    size_t len;
    bool create_dir = false;

    snprintf(tmp, sizeof(tmp), "%s", dir);
    len = strlen(tmp);

    if (tmp[len - 1] == '/') {
        tmp[len - 1] = 0;
        create_dir = true;
    }

    for (p = tmp + 1; *p; p++) {
        if (*p == '/') {
            *p = 0;
            mkdir(tmp, S_IRWXU | S_IRGRP | S_IROTH);
            *p = '/';
        }
    }

    if (create_dir) {
        mkdir(tmp, S_IRWXU | S_IRGRP | S_IROTH);
    }
}


const struct option longopts[] = {
        {"database_filepath",                               required_argument, NULL, 1},
        {"database_summarization_filepath",                 required_argument, NULL, 2},
        {"query_filepath",                                  required_argument, NULL, 3},
        {"query_summarization_filepath",                    required_argument, NULL, 4},
        {"database_size",                                   required_argument, NULL, 5},
        {"query_size",                                      required_argument, NULL, 6},
        {"sax_length",                                      required_argument, NULL, 7},
        {"sax_cardinality",                                 required_argument, NULL, 8},
        {"cpu_cores",                                       required_argument, NULL, 9},
        {"log_filepath",                                    required_argument, NULL, 10},
        {"series_length",                                   required_argument, NULL, 11},
        {"adhoc_breakpoints",                               no_argument,       NULL, 12},
        {"numa_cores",                                      required_argument, NULL, 13},
        {"index_block_size",                                required_argument, NULL, 14},
        {"leaf_size",                                       required_argument, NULL, 15},
        {"initial_leaf_size",                               required_argument, NULL, 16},
        {"exact_search",                                    no_argument,       NULL, 17},
        {"k",                                               required_argument, NULL, 18},
        {"sort_leaves",                                     no_argument,       NULL, 19},
        {"split_by_summarizations",                         no_argument,       NULL, 20},
        {"scale_factor",                                    required_argument, NULL, 21},
        {"skipped_cores",                                   required_argument, NULL, 22},
        {"numa_id",                                         required_argument, NULL, 23},
        {"series_limitations",                              required_argument, NULL, 24},
        {"leaf_compactness",                                no_argument,       NULL, 25},
        {"not_lower_bounding",                              no_argument,       NULL, 26},
        {"log_leaf_only",                                   no_argument,       NULL, 27},
        {"share_breakpoints",                               no_argument,       NULL, 28},
        {"query_bsf_filepath",                              required_argument, NULL, 29},
        {"with_id",                                         no_argument,       NULL, 32},
        {"split_by_sigma",                                  no_argument,       NULL, 37},
        {"require_neural_filter",                           no_argument,       NULL, 41},
        {"train_query_size",                                required_argument, NULL, 42},
        {"filter_train_query_filepath",                     required_argument, NULL, 43},
        {"filter_min_leaf_size",                            required_argument, NULL, 44},
        {"batch_size",                                      required_argument, NULL, 45},
        {"learning_rate",                                   required_argument, NULL, 46},
        {"max_epochs",                                      required_argument, NULL, 47},
        {"log_leaf_visits",                                 no_argument,       NULL, 48},
        {"dim_latent",                                      required_argument, NULL, 49},
        {"dropout_p",                                       required_argument, NULL, 50},
        {"max_threads_index",                               required_argument, NULL, 51},
        {"max_threads_train",                               required_argument, NULL, 52},
        {"max_threads_query",                               required_argument, NULL, 53},
        {"device_id",                                       required_argument, NULL, 54},
        {"lr_min",                                          required_argument, NULL, 55},
        {"on_disk",                                         no_argument,       NULL, 56},
        {"on_disk_folderpath",                              required_argument, NULL, 57},
        {"max_norm",                                        required_argument, NULL, 58},
        {"norm_type",                                       required_argument, NULL, 59},
        {"leaky_relu_negative_slope",                       required_argument, NULL, 60},
        {"dump_index",                                      no_argument,       NULL, 61},
        {"index_dump_folderpath",                           required_argument, NULL, 62},
        {"nodes_dump_folderpath",                           required_argument, NULL, 63},
        {"data_dump_folderpath",                            required_argument, NULL, 64},
        {"filters_dump_folderpath",                         required_argument, NULL, 65},
        {"load_index",                                      no_argument,       NULL, 66},
        {"index_load_folderpath",                           required_argument, NULL, 67},
        {"nodes_load_folderpath",                           required_argument, NULL, 68},
        {"data_load_folderpath",                            required_argument, NULL, 69},
        {"filters_load_folderpath",                         required_argument, NULL, 70},
        {"load_filters",                                    no_argument,       NULL, 71},
        {"leaf_min_split_ratio",                            required_argument, NULL, 72},
        {"filter_train_val_split",                          required_argument, NULL, 73},
        {"is_filter_conformal",                             no_argument,       NULL, 74},
        {"filter_conformal_core_type",                      required_argument, NULL, 75},
        {"filter_conformal_recall",                         required_argument, NULL, 77},
        {"filter_max_gpu_mem_mb",                           required_argument, NULL, 78},
        {"filter_allocate_is_gain",                         no_argument,       NULL, 79},
        {"filter_conformal_is_smoothen",                    no_argument,       NULL, 81},
        {"filter_conformal_smoothen_core",                  required_argument, NULL, 82},
        {"dump_model_filename_postfix",                     required_argument, NULL, 83},
        {"filter_trial_confidence_level",                   required_argument, NULL, 84},
        {"filter_trial_iterations",                         required_argument, NULL, 85},
        {"filter_trial_nnode",                              required_argument, NULL, 86},
        {"filter_trial_filter_preselection_size_threshold", required_argument, NULL, 87},
        {"allocator_cpu_trial_iterations",                  required_argument, NULL, 88},
        {"dump_filters",                                    no_argument,       NULL, 89},
        {"filter_synthetic_query_min_noise_level",          required_argument, NULL, 90},
        {"filter_synthetic_query_max_noise_level",          required_argument, NULL, 91},
        {"filter_num_synthetic_query_global",               required_argument, NULL, 92},
        {"filter_num_synthetic_query_local",                required_argument, NULL, 93},
        {"filter_num_synthetic_query_local_min",            required_argument, NULL, 94},
        {"filter_num_synthetic_query_local_max",            required_argument, NULL, 95},
        {"filter_min_leaf_size_default",                    required_argument, NULL, 96},
        {"filter_excluding_quantile",                       required_argument, NULL, 97},
        {"filter_synthetic_query_in_leaf_patience",         required_argument, NULL, 98},
        {"filter_density_comparable_margin",                required_argument, NULL, 99},
        {"filter_run_time_multiplier",                      required_argument, NULL, 100},
        {"filter_min_leaf_size_fixed",                      required_argument, NULL, 101},
        {"profile",                                         no_argument,       NULL, 102},
        {"profile_exhaustive",                              no_argument,       NULL, 103},
        {NULL,                                              no_argument,       NULL, 0}
};


int initializeThreads(Config *config, unsigned int cpu_cores, unsigned int numa_cores, unsigned int skipped_cores,
                      unsigned int numa_id) {
    cpu_set_t mask, get;

    CPU_ZERO(&mask);
    CPU_ZERO(&get);

    // for andromache(Intel(R) Xeon(R) Gold 6134 CPU @ 3.20GHz), system(cpu)-dependent, check by lscpu
    unsigned int step = 3 - numa_cores;
    for (unsigned int i = 0; i < cpu_cores; ++i) {
        CPU_SET(numa_id + (skipped_cores + i) * step, &mask);
    }

    if (pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &mask) != 0) {
        fprintf(stderr, "set thread affinity failed\n");
    }

    if (pthread_getaffinity_np(pthread_self(), sizeof(cpu_set_t), &get) != 0) {
        fprintf(stderr, "get thread affinity failed\n");
    }

    return 0;
}


Config *initializeConfig(int argc, char **argv) {
    Config *config = malloc(sizeof(Config));

    config->database_filepath = NULL;
    config->database_summarization_filepath = NULL;
    config->query_filepath = NULL;
    config->query_summarization_filepath = NULL;
    config->query_bsf_distance_filepath = NULL;
    config->log_filepath = "./isax.log";

    config->series_length = 256;

    config->database_size = 0;
    config->query_size = 0;

//    config->initial_leaf_size = 1024;
    config->initial_leaf_size = 64;
    config->leaf_size = 8000;

    config->leaf_min_split_ratio = 0.05f;

    config->index_block_size = 20000;

    config->sax_cardinality = 8;
    config->sax_length = 16;
    config->scale_factor = -1;

    config->use_adhoc_breakpoints = false;
    config->share_breakpoints = false;
    config->exact_search = false;
    config->sort_leaves = false;
    config->split_by_summarizations = false;
    config->split_by_sigma = false;

    config->k = 1;

    config->cpu_cores = 1;
    config->numa_cores = 1;
    config->numa_id = 0;
    config->skipped_cores = 0;

    config->series_limitations = 0;

    config->leaf_compactness = false;
    config->lower_bounding = true;
    config->with_id = false;

    config->filter_query_load_filepath = NULL;
    config->filter_local_queries_folderpath = NULL;
    config->filter_global_queries_filepath = NULL;
    config->require_neural_filter = false;
    config->filter_query_load_size = 0;
    config->filter_min_leaf_size = 0;
    config->filter_min_leaf_size_default = 100;
    config->filter_min_leaf_size_fixed = -1;
    config->filter_run_time_multiplier = 2;

//    config->filter_train_num_synthetic_query_per_filter = 0;
    config->filter_synthetic_query_min_noise_level = 0.1f;
    config->filter_synthetic_query_max_noise_level = 0.4f;

    config->filter_num_synthetic_query_global = -1;
    config->filter_num_synthetic_query_local = -1;
    config->filter_num_synthetic_query_local_min = -1;
    config->filter_num_synthetic_query_local_max = -1;
    config->filter_excluding_quantile = 0.025f;
    config->filter_density_comparable_margin = 0.05f;
    config->filter_synthetic_query_in_leaf_patience = -1;

    config->dim_latent = 256;
    config->dropout_p = 0.5f;
    config->leaky_relu_negative_slope = 0.01f;

    config->batch_size = 0;
    config->learning_rate = 0.01f;
    config->lr_min = 0.00001f;
    config->max_epochs = 100;

    config->log_leaf_visits = false;

    config->max_threads_index = 0;
    config->max_threads_train = 0;
    config->max_threads_query = 0;

    config->device_id = 0;

    config->log_leaf_only = false;

    config->on_disk = false;
//    config->on_disk_folderpath = NULL;

    config->dump_index = false;
    config->dump_filters = false;
    config->index_dump_folderpath = NULL;
    config->nodes_dump_folderpath = NULL;
    config->data_dump_folderpath = NULL;
    config->filters_dump_folderpath = NULL;

    config->load_index = false;
    config->index_load_folderpath = NULL;
    config->nodes_load_folderpath = NULL;
    config->data_load_folderpath = NULL;
    config->filters_load_folderpath = NULL;

    config->load_filters = false;

    config->dump_filename_postfix = ".pth";

    config->max_norm = 1;
    config->norm_type = 2;

    config->filter_train_val_split = 0.8f;
    config->is_filter_conformal = true;
    config->filter_conformal_core_type = "spline";
//    config->filter_conformal_train_val_split = 1;
    config->filter_conformal_recall = 0.99f;
    config->filter_max_gpu_mem_mb = VALUE_MAX;
    config->filter_allocate_is_gain = false;
//    config->filter_node_size_threshold = 0;
    config->filter_conformal_is_smoothen = true;
    config->filter_conformal_smoothen_core = "steffen";
    config->dump_model_filename_postfix = ".mdl";
    config->filter_trial_confidence_level = 0.95f;
    config->filter_trial_iterations = 65536;
    config->filter_trial_nnode = 32;
    config->filter_trial_filter_preselection_size_threshold = 64;
    config->allocator_cpu_trial_iterations = 16384;

    config->to_profile = false;
    config->is_profile_exhaustive = false;

    char *string_parts;
    int opt, longindex = 0;
    while ((opt = getopt_long(argc, argv, "", longopts, &longindex)) != -1) {
        switch (opt) {
            case 1:
                config->database_filepath = optarg;
                break;
            case 2:
                config->database_summarization_filepath = optarg;
                break;
            case 3:
                config->query_filepath = optarg;
                break;
            case 4:
                config->query_summarization_filepath = optarg;
                break;
            case 5:
                config->database_size = (ID_TYPE) strtoull(optarg, &string_parts, 10);
                break;
            case 6:
                config->query_size = (unsigned int) strtoul(optarg, &string_parts, 10);
                break;
            case 7:
                config->sax_length = (unsigned int) strtoul(optarg, &string_parts, 10);
                break;
            case 8:
                config->sax_cardinality = (unsigned int) strtol(optarg, &string_parts, 10);
                break;
            case 9:
                config->cpu_cores = (int) strtol(optarg, &string_parts, 10);
                break;
            case 10:
                config->log_filepath = optarg;
                break;
            case 11:
                config->series_length = (unsigned int) strtol(optarg, &string_parts, 10);
                break;
            case 12:
                config->use_adhoc_breakpoints = true;
                break;
            case 13:
                config->numa_cores = (unsigned int) strtol(optarg, &string_parts, 10);
                break;
            case 14:
                config->index_block_size = (unsigned int) strtoul(optarg, &string_parts, 10);
                break;
            case 15:
                config->leaf_size = (unsigned int) strtoul(optarg, &string_parts, 10);
                break;
            case 16:
                config->initial_leaf_size = (unsigned int) strtoul(optarg, &string_parts, 10);
                break;
            case 17:
                config->exact_search = true;
                break;
            case 18:
                config->k = (unsigned int) strtol(optarg, &string_parts, 10);
                break;
            case 19:
                config->sort_leaves = true;
                break;
            case 20:
                config->split_by_summarizations = true;
                break;
            case 21:
                config->scale_factor = strtof(optarg, &string_parts);
                break;
            case 22:
                config->skipped_cores = (unsigned int) strtol(optarg, &string_parts, 10);
                break;
            case 23:
                config->numa_id = (unsigned int) strtol(optarg, &string_parts, 10);
                break;
            case 24:
                config->series_limitations = (unsigned int) strtol(optarg, &string_parts, 10);
                break;
            case 25:
                config->leaf_compactness = true;
                break;
            case 26:
                config->lower_bounding = false;
                break;
            case 27:
                config->log_leaf_only = true;
                break;
            case 28:
                config->share_breakpoints = true;
                break;
            case 29:
                config->query_bsf_distance_filepath = optarg;
                break;
            case 32:
                config->with_id = true;
                break;
            case 37:
                config->split_by_sigma = true;
                config->split_by_summarizations = true;
                break;
            case 41:
                config->require_neural_filter = true;
                break;
            case 42:
                config->filter_query_load_size = (unsigned int) strtoul(optarg, &string_parts, 10);
                break;
            case 43:
                config->filter_query_load_filepath = optarg;
                break;
            case 44:
                config->filter_min_leaf_size = (ID_TYPE) strtoul(optarg, &string_parts, 10);
                break;
            case 45:
                config->batch_size = (unsigned int) strtoul(optarg, &string_parts, 10);
                break;
            case 46:
                config->learning_rate = (VALUE_TYPE) strtof(optarg, &string_parts);
                break;
            case 47:
                config->max_epochs = (unsigned int) strtoul(optarg, &string_parts, 10);
                break;
            case 48:
                config->log_leaf_visits = true;
                break;
            case 49:
                config->dim_latent = (unsigned int) strtoul(optarg, &string_parts, 10);
                break;
            case 50:
                config->dropout_p = (VALUE_TYPE) strtof(optarg, &string_parts);
                break;
            case 51:
                config->max_threads_index = (unsigned int) strtoul(optarg, &string_parts, 10);
                break;
            case 52:
                config->max_threads_train = (unsigned int) strtoul(optarg, &string_parts, 10);
                break;
            case 53:
                config->max_threads_query = (unsigned int) strtoul(optarg, &string_parts, 10);
                break;
            case 54:
                config->device_id = (int) strtoul(optarg, &string_parts, 10);
                break;
            case 55:
                config->lr_min = (VALUE_TYPE) strtof(optarg, &string_parts);
                break;
            case 56:
                config->on_disk = true;
                break;
//            case 57:
//                config->on_disk_folderpath = optarg;
//                break;
            case 58:
                config->max_norm = (VALUE_TYPE) strtof(optarg, &string_parts);
                break;
            case 59:
                config->norm_type = (VALUE_TYPE) strtof(optarg, &string_parts);
                break;
            case 60:
                config->leaky_relu_negative_slope = (VALUE_TYPE) strtof(optarg, &string_parts);
                break;
            case 61:
                config->dump_index = true;
                break;
            case 62:
                config->index_dump_folderpath = optarg;
                break;
            case 63:
                config->nodes_dump_folderpath = optarg;
                break;
            case 64:
                config->data_dump_folderpath = optarg;
                break;
            case 65:
                config->filters_dump_folderpath = optarg;
                break;
            case 66:
                config->load_index = true;
                break;
            case 67:
                config->index_load_folderpath = optarg;
                break;
            case 68:
                config->nodes_load_folderpath = optarg;
                break;
            case 69:
                config->data_load_folderpath = optarg;
                break;
            case 70:
                config->filters_load_folderpath = optarg;
                break;
            case 71:
                config->load_filters = true;
                break;
            case 72:
                config->leaf_min_split_ratio = (VALUE_TYPE) strtof(optarg, &string_parts);
                break;
            case 73:
                config->filter_train_val_split = (VALUE_TYPE) strtof(optarg, &string_parts);
                break;
            case 74:
                config->is_filter_conformal = true;
                break;
            case 75:
                config->filter_conformal_core_type = optarg;
                break;
//            case 76:
//                config->filter_conformal_train_val_split = (VALUE_TYPE) strtof(optarg, &string_parts);
//                break;
            case 77:
                config->filter_conformal_recall = (VALUE_TYPE) strtof(optarg, &string_parts);
                break;
            case 78:
                config->filter_max_gpu_mem_mb = (VALUE_TYPE) strtof(optarg, &string_parts);
                break;
            case 79:
                config->filter_allocate_is_gain = true;
                break;
//            case 80:
//                config->filter_node_size_threshold = (int) strtoul(optarg, &string_parts, 10);
//                break;
            case 81:
                config->filter_conformal_is_smoothen = true;
                break;
            case 82:
                config->filter_conformal_smoothen_core = optarg;
                break;
            case 83:
                config->dump_model_filename_postfix = optarg;
                break;
            case 84:
                config->filter_trial_confidence_level = (VALUE_TYPE) strtof(optarg, &string_parts);
                break;
            case 85:
                config->filter_trial_iterations = (int) strtoul(optarg, &string_parts, 10);
                break;
            case 86:
                config->filter_trial_nnode = (int) strtoul(optarg, &string_parts, 10);
                break;
            case 87:
                config->filter_trial_filter_preselection_size_threshold = (VALUE_TYPE) strtof(optarg, &string_parts);
                break;
            case 88:
                config->allocator_cpu_trial_iterations = (int) strtoul(optarg, &string_parts, 10);
                break;
            case 89:
                config->dump_filters = true;
                break;
            case 90:
                config->filter_synthetic_query_min_noise_level = (VALUE_TYPE) strtof(optarg, &string_parts);
                break;
            case 91:
                config->filter_synthetic_query_max_noise_level = (VALUE_TYPE) strtof(optarg, &string_parts);
                break;
            case 92:
                config->filter_num_synthetic_query_global = (ID_TYPE) strtoul(optarg, &string_parts, 10);
                break;
            case 93:
                config->filter_num_synthetic_query_local = (ID_TYPE) strtoul(optarg, &string_parts, 10);
                break;
            case 94:
                config->filter_num_synthetic_query_local_min = (ID_TYPE) strtoul(optarg, &string_parts, 10);
                break;
            case 95:
                config->filter_num_synthetic_query_local_max = (ID_TYPE) strtoul(optarg, &string_parts, 10);
                break;
            case 96:
                config->filter_min_leaf_size_default = (ID_TYPE) strtoul(optarg, &string_parts, 10);
                break;
            case 97:
                config->filter_excluding_quantile = (VALUE_TYPE) strtof(optarg, &string_parts);
                break;
            case 98:
                config->filter_synthetic_query_in_leaf_patience = (ID_TYPE) strtoul(optarg, &string_parts, 10);
                break;
            case 99:
                config->filter_density_comparable_margin = (VALUE_TYPE) strtof(optarg, &string_parts);
                break;
            case 100:
                config->filter_run_time_multiplier = (VALUE_TYPE) strtof(optarg, &string_parts);
                break;
            case 101:
                config->filter_min_leaf_size_fixed = (ID_TYPE) strtof(optarg, &string_parts);
                break;
            case 102:
                config->to_profile = true;
                break;
            case 103:
                config->is_profile_exhaustive = true;
                break;
            default:
                exit(EXIT_FAILURE);
        }
    }

//    assert(config->series_length % config->sax_length == 0 && config->series_length % 8 == 0);
    assert(config->series_length % 8 == 0);
//    assert(config->sax_length == 8 || config->sax_length == 16);
    assert(config->sax_length >= 8 || config->sax_length <= 16);
    assert(config->sax_cardinality > 0 && config->sax_cardinality <= 8);
    assert(config->database_size > 0);
    assert(config->query_size > 0);
    assert(config->index_block_size > 0);
    assert(config->series_length > 0);
    assert(config->leaf_size > 0 && config->initial_leaf_size > 0 && config->initial_leaf_size <= config->leaf_size);
//     assert(config->k >= 0 && config->k <= 1024);
    assert(config->k >= 0);

    assert(config->leaf_min_split_ratio > 0 && config->leaf_min_split_ratio < 1);
    config->leaf_min_split_size = (ID_TYPE) (config->leaf_size * config->leaf_min_split_ratio);

    _mkdir(config->log_filepath);

    assert(!(config->load_index && config->with_id));

    if (config->on_disk) {
        assert(config->dump_index || config->load_index);
    }

    if (config->dump_index) {
        if (config->index_dump_folderpath == NULL) {
            char log_filepath[PATH_MAX];
            memcpy(log_filepath, config->log_filepath, sizeof(char) * strlen(config->log_filepath));

            config->index_dump_folderpath = dirname(log_filepath);
        }

        if (config->index_dump_folderpath[strlen(config->index_dump_folderpath) - 1] != '/') {
            config->index_dump_folderpath = concat(2, config->index_dump_folderpath, "/");
        }

        if (config->nodes_dump_folderpath == NULL) {
            config->nodes_dump_folderpath = concat(2, config->index_dump_folderpath, "node/");
        } else if (config->nodes_dump_folderpath[strlen(config->nodes_dump_folderpath) - 1] != '/') {
            config->nodes_dump_folderpath = concat(2, config->nodes_dump_folderpath, "/");
        }

        if (config->data_dump_folderpath == NULL) {
            config->data_dump_folderpath = concat(2, config->index_dump_folderpath, "data/");
        } else if (config->data_dump_folderpath[strlen(config->data_dump_folderpath) - 1] != '/') {
            config->data_dump_folderpath = concat(2, config->data_dump_folderpath, "/");
        }

        if (config->require_neural_filter) {
            if (config->filters_dump_folderpath == NULL) {
                config->filters_dump_folderpath = concat(2, config->index_dump_folderpath, "filter/");
            } else if (config->filters_dump_folderpath[strlen(config->filters_dump_folderpath) - 1] != '/') {
                config->filters_dump_folderpath = concat(2, config->filters_dump_folderpath, "/");
            }
        }

        _mkdir(config->nodes_dump_folderpath);
        _mkdir(config->data_dump_folderpath);
        _mkdir(config->filters_dump_folderpath);

        if (config->use_adhoc_breakpoints) {
            config->breakpoints_dump_filepath = concat(3, config->index_dump_folderpath,
                                                       "breakpoints", config->dump_filename_postfix);
        }
    } else if (config->dump_filters) {
        assert(config->filters_dump_folderpath != NULL);

        if (config->filters_dump_folderpath[strlen(config->filters_dump_folderpath) - 1] != '/') {
            config->filters_dump_folderpath = concat(2, config->filters_dump_folderpath, "/");
        }

        _mkdir(config->filters_dump_folderpath);
    }

    if (config->load_index) {
        if (config->index_load_folderpath == NULL) {
            assert(config->nodes_load_folderpath != NULL);
            assert(config->data_load_folderpath != NULL);

            if (config->require_neural_filter && config->load_filters) {
                assert(config->filters_load_folderpath != NULL);
            }
        } else {
            if (config->index_load_folderpath[strlen(config->index_load_folderpath) - 1] != '/') {
                config->index_load_folderpath = concat(2, config->index_load_folderpath, "/");
            }

            if (config->nodes_load_folderpath == NULL) {
                config->nodes_load_folderpath = concat(2, config->index_load_folderpath, "node/");
            }

            if (config->data_load_folderpath == NULL) {
                config->data_load_folderpath = concat(2, config->index_load_folderpath, "data/");
            }

            if (config->require_neural_filter && config->load_filters) {
                if (config->filters_load_folderpath == NULL) {
                    config->filters_load_folderpath = concat(2, config->index_load_folderpath, "filter/");
                }
            }
        }

        if (config->nodes_load_folderpath[strlen(config->nodes_load_folderpath) - 1] != '/') {
            config->nodes_load_folderpath = concat(2, config->nodes_load_folderpath, "/");
        }
        if (config->data_load_folderpath[strlen(config->data_load_folderpath) - 1] != '/') {
            config->data_load_folderpath = concat(2, config->data_load_folderpath, "/");
        }
        if (config->require_neural_filter && config->load_filters) {
            if (config->filters_load_folderpath[strlen(config->filters_load_folderpath) - 1] != '/') {
                config->filters_load_folderpath = concat(2, config->filters_load_folderpath, "/");
            }
        }

        if (config->use_adhoc_breakpoints) {
            config->breakpoints_load_filepath = concat(3, config->index_load_folderpath,
                                                       "breakpoints", config->dump_filename_postfix);
        }
    }

    VALUE_TYPE scale_factor = (VALUE_TYPE) config->series_length / (VALUE_TYPE) config->sax_length;

    if (VALUE_EQ(config->scale_factor, -1)) {
        config->scale_factor = scale_factor;
    } else {
        if (config->database_summarization_filepath == NULL) {
            if (VALUE_EQ(config->scale_factor, scale_factor)) {
                printf("config - unmatched scale factors: %f vs. %f(given)\n", scale_factor, config->scale_factor);
                exit(-1);
            }
        } else {
            assert(config->scale_factor > 0);
        }
    }

    if (config->require_neural_filter) {
        if (config->to_profile) {
            printf("config - profiling filters is not supported yet\n");
            exit(-1);
        }

        if (config->dim_latent < 1) {
            config->dim_latent = config->series_length;
        }

        if (config->filter_query_load_size < 1) {
            assert(config->filter_query_load_filepath == NULL);
            assert(config->filter_num_synthetic_query_global > 0);

            if (config->filter_num_synthetic_query_local <= 0) {
                if (config->filter_num_synthetic_query_local_min <= 0) {
                    config->filter_num_synthetic_query_local_min =
                            (ID_TYPE) config->filter_num_synthetic_query_global / 10;
                }

                if (config->filter_num_synthetic_query_local_max <= 0) {
                    config->filter_num_synthetic_query_local_max = config->filter_num_synthetic_query_global;
                }
            }

            if (config->filter_synthetic_query_in_leaf_patience < 0) {
                config->filter_synthetic_query_in_leaf_patience = config->filter_num_synthetic_query_local_max;
            }

            assert(config->filter_synthetic_query_min_noise_level > 0);
            assert(config->filter_synthetic_query_max_noise_level >
                   config->filter_synthetic_query_min_noise_level);

            assert(config->filter_excluding_quantile > 0 && config->filter_excluding_quantile < 0.5);
        }

        if (!config->load_filters && config->filter_query_load_filepath == NULL) {
            assert(config->filter_query_load_size < 1);
            char log_filepath[PATH_MAX];
            memcpy(log_filepath, config->log_filepath, sizeof(char) * strlen(config->log_filepath));

            char *log_folderpath = dirname(log_filepath);
            config->filter_local_queries_folderpath = concat(
                    2, log_folderpath, log_folderpath[strlen(log_folderpath) - 1] == '/' ? "query/" : "/query/");

            _mkdir(config->filter_local_queries_folderpath);

            config->filter_global_queries_filepath = concat(2, config->filter_local_queries_folderpath, "global.bin");
        } else {
            config->filter_local_queries_folderpath = NULL;
            config->filter_global_queries_filepath = NULL;
        }

        if (config->filter_min_leaf_size < 1) {
            config->filter_min_leaf_size = config->filter_min_leaf_size_default;
        }

        assert(config->filter_run_time_multiplier > 0);

        if (config->filter_min_leaf_size_fixed > 0) {
            assert(!config->filter_allocate_is_gain);
        }
    }

    assert(config->device_id == 0 || config->device_id == 1); // thalia

    if (config->max_threads_index == 0) {
        config->max_threads_index = config->cpu_cores;
    } else if (config->cpu_cores < config->max_threads_index) {
        config->cpu_cores = config->max_threads_index;
    }
    if (config->max_threads_query == 0) {
        config->max_threads_query = config->cpu_cores;
    } else if (config->cpu_cores < config->max_threads_query) {
        config->cpu_cores = config->max_threads_query;
    }

    if (config->max_threads_train == 0) {
        config->max_threads_train = config->cpu_cores;
    } else if (config->cpu_cores < config->max_threads_train) {
        config->cpu_cores = config->max_threads_train;
    }

    // for thalia; cpu_set_t is platform-dependent
    assert(config->cpu_cores > 0 && config->numa_cores > 0 &&
           (config->numa_id == 0 || config->numa_id == 1) &&
           ((config->numa_cores == 2 && config->skipped_cores + config->cpu_cores <= 80) ||
            (config->numa_cores == 1 && config->skipped_cores + config->cpu_cores <= 40)));

    initializeThreads(config, config->cpu_cores, config->numa_cores, config->skipped_cores, config->numa_id);

    return config;
}


void logConfig(Config const *config) {
    clog_info(CLOG(CLOGGER_ID), "config - database_filepath = %s", config->database_filepath);
    clog_info(CLOG(CLOGGER_ID), "config - database_summarization_filepath = %s",
              config->database_summarization_filepath);
    clog_info(CLOG(CLOGGER_ID), "config - query_filepath = %s", config->query_filepath);
    clog_info(CLOG(CLOGGER_ID), "config - query_summarization_filepath = %s", config->query_summarization_filepath);
    clog_info(CLOG(CLOGGER_ID), "config - query_bsf_distance_filepath = %s", config->query_bsf_distance_filepath);
    clog_info(CLOG(CLOGGER_ID), "config - log_filepath = %s", config->log_filepath);

    clog_info(CLOG(CLOGGER_ID), "config - series_length = %u", config->series_length);
    clog_info(CLOG(CLOGGER_ID), "config - database_size = %lu", config->database_size);
    clog_info(CLOG(CLOGGER_ID), "config - query_size = %u", config->query_size);
    clog_info(CLOG(CLOGGER_ID), "config - sax_length = %u", config->sax_length);
    clog_info(CLOG(CLOGGER_ID), "config - sax_cardinality = %d", config->sax_cardinality);
    clog_info(CLOG(CLOGGER_ID), "config - adhoc_breakpoints = %d", config->use_adhoc_breakpoints);
    clog_info(CLOG(CLOGGER_ID), "config - share_breakpoints = %d", config->share_breakpoints);

    clog_info(CLOG(CLOGGER_ID), "config - exact_search = %d", config->exact_search);
    clog_info(CLOG(CLOGGER_ID), "config - k = %d", config->k);

    clog_info(CLOG(CLOGGER_ID), "config - leaf_size = %u", config->leaf_size);
    clog_info(CLOG(CLOGGER_ID), "config - leaf_min_split_ratio = %f", config->leaf_min_split_ratio);
    clog_info(CLOG(CLOGGER_ID), "config - leaf_min_split_size = %d", config->leaf_min_split_size);

    clog_info(CLOG(CLOGGER_ID), "config - initial_leaf_size = %u", config->initial_leaf_size);
    clog_info(CLOG(CLOGGER_ID), "config - sort_leaves = %d", config->sort_leaves);
    clog_info(CLOG(CLOGGER_ID), "config - split_by_summarizations = %d", config->split_by_summarizations);

    clog_info(CLOG(CLOGGER_ID), "config - cpu_cores = %d", config->cpu_cores);
    clog_info(CLOG(CLOGGER_ID), "config - numa_cores = %d", config->numa_cores);
    clog_info(CLOG(CLOGGER_ID), "config - skipped_cores = %d", config->skipped_cores);
    clog_info(CLOG(CLOGGER_ID), "config - numa_id = %d", config->numa_id);
    clog_info(CLOG(CLOGGER_ID), "config - index_block_size = %u", config->index_block_size);

    clog_info(CLOG(CLOGGER_ID), "config - require_neural_filter = %d", config->require_neural_filter);
    clog_info(CLOG(CLOGGER_ID), "config - train_query_size = %d", config->filter_query_load_size);
    clog_info(CLOG(CLOGGER_ID), "config - filter_train_query_filepath = %s", config->filter_query_load_filepath);
    clog_info(CLOG(CLOGGER_ID), "config - filter_train_query_filepath_destination = %s",
              config->filter_global_queries_filepath);
    clog_info(CLOG(CLOGGER_ID), "config - filter_min_leaf_size = %d", config->filter_min_leaf_size);
    clog_info(CLOG(CLOGGER_ID), "config - filter_min_leaf_size_default = %d", config->filter_min_leaf_size_default);
    clog_info(CLOG(CLOGGER_ID), "config - filter_min_leaf_size_fixed = %d", config->filter_min_leaf_size_fixed);
    clog_info(CLOG(CLOGGER_ID), "config - filter_run_time_multiplier = %.3f", config->filter_run_time_multiplier);

    clog_info(CLOG(CLOGGER_ID), "config - filter_global_queries_filepath = %s",
              config->filter_global_queries_filepath);
    clog_info(CLOG(CLOGGER_ID), "config - filter_local_queries_folderpath = %s",
              config->filter_local_queries_folderpath);

    clog_info(CLOG(CLOGGER_ID), "config - filter_train_synthetic_query_min_noise_level = %.2f",
              config->filter_synthetic_query_min_noise_level);
    clog_info(CLOG(CLOGGER_ID), "config - filter_train_synthetic_query_max_noise_level = %.2f",
              config->filter_synthetic_query_max_noise_level);

    clog_info(CLOG(CLOGGER_ID), "config - filter_num_synthetic_query_global = %d",
              config->filter_num_synthetic_query_global);
    clog_info(CLOG(CLOGGER_ID), "config - filter_num_synthetic_query_local = %d",
              config->filter_num_synthetic_query_local);
    clog_info(CLOG(CLOGGER_ID), "config - filter_num_synthetic_query_local_min = %d",
              config->filter_num_synthetic_query_local_min);
    clog_info(CLOG(CLOGGER_ID), "config - filter_num_synthetic_query_local_max = %d",
              config->filter_num_synthetic_query_local_max);
    clog_info(CLOG(CLOGGER_ID), "config - filter_excluding_quantile = %.4f",
              config->filter_excluding_quantile);
    clog_info(CLOG(CLOGGER_ID), "config - filter_density_comparable_margin = %.4f",
              config->filter_density_comparable_margin);
    clog_info(CLOG(CLOGGER_ID), "config - filter_synthetic_query_in_leaf_patience = %d",
              config->filter_synthetic_query_in_leaf_patience);

    clog_info(CLOG(CLOGGER_ID), "config - batch_size = %d", config->batch_size);
    clog_info(CLOG(CLOGGER_ID), "config - learning_rate = %f", config->learning_rate);
    clog_info(CLOG(CLOGGER_ID), "config - lr_min = %f", config->lr_min);
    clog_info(CLOG(CLOGGER_ID), "config - max_epochs = %d", config->max_epochs);

    clog_info(CLOG(CLOGGER_ID), "config - log_leaf_visits = %d", config->log_leaf_visits);

    clog_info(CLOG(CLOGGER_ID), "config - max_threads_index = %d", config->max_threads_index);
    clog_info(CLOG(CLOGGER_ID), "config - max_threads_train = %d", config->max_threads_train);
    clog_info(CLOG(CLOGGER_ID), "config - max_threads_query = %d", config->max_threads_query);

    clog_info(CLOG(CLOGGER_ID), "config - device_id = %d", config->device_id);

    clog_info(CLOG(CLOGGER_ID), "config - on_disk = %d", config->on_disk);

    clog_info(CLOG(CLOGGER_ID), "config - max_norm = %f", config->max_norm);
    clog_info(CLOG(CLOGGER_ID), "config - norm_type = %f", config->norm_type);

    clog_info(CLOG(CLOGGER_ID), "config - dump_index = %d", config->dump_index);
    clog_info(CLOG(CLOGGER_ID), "config - index_dump_folderpath = %s", config->index_dump_folderpath);
    clog_info(CLOG(CLOGGER_ID), "config - nodes_dump_folderpath = %s", config->nodes_dump_folderpath);
    clog_info(CLOG(CLOGGER_ID), "config - data_dump_folderpath = %s", config->data_dump_folderpath);
    clog_info(CLOG(CLOGGER_ID), "config - filters_dump_folderpath = %s", config->filters_dump_folderpath);

    clog_info(CLOG(CLOGGER_ID), "config - index_load_folderpath = %s", config->index_load_folderpath);
    clog_info(CLOG(CLOGGER_ID), "config - nodes_load_folderpath = %s", config->nodes_load_folderpath);
    clog_info(CLOG(CLOGGER_ID), "config - data_load_folderpath = %s", config->data_load_folderpath);
    clog_info(CLOG(CLOGGER_ID), "config - filters_load_folderpath = %s", config->filters_load_folderpath);
    clog_info(CLOG(CLOGGER_ID), "config - load_filters = %d", config->load_filters);

    clog_info(CLOG(CLOGGER_ID), "config - filter_train_val_split = %f", config->filter_train_val_split);
    clog_info(CLOG(CLOGGER_ID), "config - is_filter_conformal = %d", config->is_filter_conformal);
    clog_info(CLOG(CLOGGER_ID), "config - filter_conformal_core_type = %s", config->filter_conformal_core_type);
    clog_info(CLOG(CLOGGER_ID), "config - filter_conformal_recall = %f", config->filter_conformal_recall);

    clog_info(CLOG(CLOGGER_ID), "config - filter_max_gpu_mem_mb = %f", config->filter_max_gpu_mem_mb);
    clog_info(CLOG(CLOGGER_ID), "config - filter_allocate_is_gain = %d", config->filter_allocate_is_gain);
    clog_info(CLOG(CLOGGER_ID), "config - filter_conformal_is_smoothen = %d", config->filter_conformal_is_smoothen);
    clog_info(CLOG(CLOGGER_ID), "config - filter_conformal_smoothen_core = %s", config->filter_conformal_smoothen_core);
    clog_info(CLOG(CLOGGER_ID), "config - dump_model_filename_postfix = %s", config->dump_model_filename_postfix);

    clog_info(CLOG(CLOGGER_ID), "config - filter_trial_confidence_level = %f", config->filter_trial_confidence_level);

    clog_info(CLOG(CLOGGER_ID), "config - filter_trial_confidence_level = %f", config->filter_trial_confidence_level);
    clog_info(CLOG(CLOGGER_ID), "config - filter_trial_iterations = %d", config->filter_trial_iterations);
    clog_info(CLOG(CLOGGER_ID), "config - filter_trial_nnode = %d", config->filter_trial_nnode);
    clog_info(CLOG(CLOGGER_ID), "config - filter_trial_filter_preselection_size_threshold = %d",
              config->filter_trial_filter_preselection_size_threshold);
    clog_info(CLOG(CLOGGER_ID), "config - allocator_cpu_trial_iterations = %d", config->allocator_cpu_trial_iterations);

    clog_info(CLOG(CLOGGER_ID), "config - to_profile = %d", config->to_profile);
    clog_info(CLOG(CLOGGER_ID), "config - is_profile_exhaustive = %d", config->is_profile_exhaustive);
}


void freeConfig(Config *config) {
    if (config->filter_global_queries_filepath != NULL) {
        free(config->filter_global_queries_filepath);
        config->filter_global_queries_filepath = NULL;
    }
}