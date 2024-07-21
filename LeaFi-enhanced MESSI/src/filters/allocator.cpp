/******************************************************************************
 * Copyright (c) 2024, the LeaFi author(s).
 ******************************************************************************/

#include "allocator.h"

#include <vector>
#include <random>
#include <string>
#include <fstream>
#include <immintrin.h>

#include <cuda.h>
#include <ATen/ATen.h>
#include <torch/cuda.h>
#include <torch/types.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>

extern "C"
{
#include "distance.h"
#include "globals.h"
}

#include "clog.h"
#include "static_variables.h"
#include "conformal_adjustor.h"


template<class T>
std::vector<T> make_reserved(const ID_TYPE n) {
    std::vector<T> vec;
    vec.reserve(n);
    return vec;
}


void measure_cpu(FilterAllocator *allocator, Config const *config) {
    auto batch_nbytes = static_cast<ID_TYPE>(sizeof(VALUE_TYPE)) * config->series_length * config->leaf_size;
    auto trial_batch = static_cast<VALUE_TYPE *>(aligned_alloc(sizeof(__m256), batch_nbytes));

    auto distances = make_reserved<VALUE_TYPE>(config->leaf_size);

    if (config->on_disk) {
        // credit to https://stackoverflow.com/a/19728404
        std::random_device rd;
        std::mt19937 rng(rd());
        std::uniform_int_distribution<ID_TYPE> uni_i_d(0, config->database_size - config->leaf_size);

        auto start = std::chrono::high_resolution_clock::now();

        for (ID_TYPE trial_i = 0; trial_i < config->allocator_cpu_trial_iterations; ++trial_i) {
            auto m256_fetched_cache = static_cast<VALUE_TYPE *>(aligned_alloc(256, sizeof(VALUE_TYPE) * 8));

            std::ifstream db_fin;
            db_fin.open(config->database_filepath, std::ios::in | std::ios::binary);
            db_fin.seekg(static_cast<ID_TYPE>(sizeof(VALUE_TYPE)) * config->series_length * uni_i_d(rng));
            db_fin.read(reinterpret_cast<char *>(trial_batch), batch_nbytes);

            for (ID_TYPE series_i = 0; series_i < config->leaf_size; ++series_i) {
                VALUE_TYPE distance = l2SquareSIMD(config->series_length,
                                                   trial_batch, trial_batch + series_i * config->series_length,
                                                   m256_fetched_cache);
                distances.push_back(distance);
            }

            distances.clear();
            free(m256_fetched_cache);
            db_fin.close();
        }

        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

        allocator->cpu_ms_per_series = duration.count() / static_cast<double>(
                config->allocator_cpu_trial_iterations * config->leaf_size);
    } else {
        std::ifstream db_fin;
        db_fin.open(config->database_filepath, std::ios::in | std::ios::binary);
        db_fin.read(reinterpret_cast<char *>(trial_batch), batch_nbytes);

        auto start = std::chrono::high_resolution_clock::now();

        for (ID_TYPE trial_i = 0; trial_i < config->allocator_cpu_trial_iterations; ++trial_i) {
            auto m256_fetched_cache = static_cast<VALUE_TYPE *>(aligned_alloc(256, sizeof(VALUE_TYPE) * 8));

            for (ID_TYPE series_i = 0; series_i < config->leaf_size; ++series_i) {
                VALUE_TYPE distance = l2SquareSIMD(config->series_length,
                                                   trial_batch, trial_batch + series_i * config->series_length,
                                                   m256_fetched_cache);
                distances.push_back(distance);
            }

            distances.clear();
            free(m256_fetched_cache);
        }

        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

        allocator->cpu_ms_per_series = duration.count() / static_cast<double>(
                config->allocator_cpu_trial_iterations * config->leaf_size);

        db_fin.close();
    }

    clog_info(CLOG(CLOGGER_ID), "allocator trial cpu time = %fmus", allocator->cpu_ms_per_series);
    free(trial_batch);
}


void measure_gpu(FilterAllocator *allocator, Config const *config) {
    if (torch::cuda::is_available()) {
        profileFilter(config, &allocator->gpu_ms, &allocator->gpu_mem_mb);
    }
}


void measureTimeNMem(FilterAllocator *allocator, Config *config) {
    size_t gpu_free_bytes, gpu_total_bytes;
    cudaSetDevice(config->device_id);
    cuMemGetInfo(&gpu_free_bytes, &gpu_total_bytes);
    VALUE_TYPE gpu_free_mb = static_cast<VALUE_TYPE>(gpu_free_bytes) / 1024 / 1024;

    if (gpu_free_mb < allocator->available_gpu_mem_mb) {
        clog_warn(CLOG(CLOGGER_ID), "allocator required %fmb is not available; down to all free %fmb",
                  allocator->available_gpu_mem_mb, gpu_free_mb);
        allocator->available_gpu_mem_mb = gpu_free_mb;
    } else {
        clog_info(CLOG(CLOGGER_ID), "allocator requested %fmb; %fmb available",
                  allocator->available_gpu_mem_mb, gpu_free_mb);
    }

    if (config->filter_min_leaf_size_fixed > 0) {
        config->filter_min_leaf_size = config->filter_min_leaf_size_fixed;

        clog_info(CLOG(CLOGGER_ID), "allocator node_size_threshold fixed at %d; default %d",
                  config->filter_min_leaf_size, config->filter_min_leaf_size_default);
    } else {
        measure_cpu(allocator, config);
        measure_gpu(allocator, config);

        int node_size_threshold = static_cast<ID_TYPE>(
                config->filter_run_time_multiplier * allocator->gpu_ms / allocator->cpu_ms_per_series);
        clog_info(CLOG(CLOGGER_ID), "allocator measured node_size_threshold * %.3f = %d",
                  config->filter_run_time_multiplier, node_size_threshold);

        if (config->filter_min_leaf_size < 1) {
            config->filter_min_leaf_size = node_size_threshold;
        } else if (config->filter_min_leaf_size < node_size_threshold) {
            clog_warn(CLOG(CLOGGER_ID),
                      "allocator rectify filter_min_leaf_size %d to measured %d",
                      config->filter_min_leaf_size, node_size_threshold);

            config->filter_min_leaf_size = node_size_threshold;
        }

//    if (config->filter_min_leaf_size < node_size_threshold) {
//        clog_warn(CLOG(CLOGGER_ID),
//                  "allocator measured node_size_threshold (%d) exceeds config->min_leafsize4filter (%d) ",
//                  node_size_threshold, config->filter_min_leaf_size);
//    }
    }
}


FilterAllocator *initAllocator(Config const *config, unsigned int num_filters) {
    auto allocator = static_cast<FilterAllocator *>(malloc(sizeof(FilterAllocator)));

    allocator->filters = static_cast<NeuralFilter **>(malloc(sizeof(NeuralFilter *) * num_filters));
    allocator->num_filters = 0;

    allocator->cpu_ms_per_series = -1;
    allocator->gpu_ms = -1;
    allocator->gpu_mem_mb = -1;
    allocator->estimated_filter_pruning_ratio = -1;

    allocator->available_gpu_mem_mb = config->filter_max_gpu_mem_mb;

    if (config->filter_query_load_size > 0) {
        allocator->num_global_examples = config->filter_query_load_size;
    } else {
        allocator->num_global_examples = config->filter_num_synthetic_query_global;
    }
    allocator->num_global_train_examples = allocator->num_global_examples * config->filter_train_val_split;

    allocator->validation_recalls = nullptr;

    measureTimeNMem(allocator, (Config *) config);

    return allocator;
}


void pushFilter(FilterAllocator *allocator, NeuralFilter *filter) {
    allocator->filters[allocator->num_filters] = filter;
    allocator->num_filters += 1;
}


typedef struct TrialCache {
    Config const *config;
    NeuralFilter **filters;

    ID_TYPE num_samples;
    std::vector<ID_TYPE> *sampled_pos_vec;
    std::vector<VALUE_TYPE> *filter_pruning_ratios;

    ID_TYPE *shared_filter_pos;
    ID_TYPE filter_block_size;

    VALUE_TYPE confidence_level;
    ID_TYPE num_all_examples;
    ID_TYPE num_train_examples;

    unsigned int stream_id;
} TrialCache;


void *trialThread(void *cache) {
    std::string sax_str = "trial";
    auto sax_cstr = const_cast<char *>(sax_str.c_str());

    auto *trial_cache = (TrialCache *) cache;
    Config const *config = trial_cache->config;
    NeuralFilter **filters = trial_cache->filters;

    unsigned int num_samples = trial_cache->num_samples;
    std::vector<ID_TYPE> *sampled_pos_vec = trial_cache->sampled_pos_vec;
    std::vector<VALUE_TYPE> *filter_pruning_ratios = trial_cache->filter_pruning_ratios;

    ID_TYPE *shared_filter_pos = trial_cache->shared_filter_pos;
    unsigned int block_size = trial_cache->filter_block_size;

    VALUE_TYPE confidence_level = trial_cache->confidence_level;
    ID_TYPE num_all_examples = trial_cache->num_all_examples;
    ID_TYPE num_train_examples = trial_cache->num_train_examples;

    unsigned int stream_id = trial_cache->stream_id;

    ID_TYPE filter_pos, stop_filter_pos;
    while ((filter_pos = __sync_fetch_and_add(shared_filter_pos, block_size)) < num_samples) {
        stop_filter_pos = filter_pos + block_size;
        if (stop_filter_pos > num_samples) {
            stop_filter_pos = num_samples;
        }

        for (unsigned int i = filter_pos; i < stop_filter_pos; ++i) {
            NeuralFilter *filter = filters[(*sampled_pos_vec)[i]];
            bool failed_to_train = false;

            if (!filter->is_trained) {
                if (trainNeuralFilter(config, filter, stream_id, sax_cstr)) {
                    clog_info(CLOG(CLOGGER_ID), "allocator trial - trained filter %d", filter->id);
                } else {
                    clog_error(CLOG(CLOGGER_ID), "allocator trial - failed to train filter %d", filter->id);

                    failed_to_train = true;
                }
            } else {
                clog_info(CLOG(CLOGGER_ID), "allocator trial - filter %d already trained", filter->id);
            }

            if (failed_to_train) {
                (*filter_pruning_ratios)[i] = -1;
            } else {
                ID_TYPE cnt_prune = 0;

                void *adjustor_ptr = filter->conformal_adjustor;
                if (filter->conformal_adjustor != nullptr) {
                    auto adjustor = (ConformalRegressor *) adjustor_ptr;

                    auto abs_error_i = static_cast<ID_TYPE>(static_cast<VALUE_TYPE>(adjustor->alphas_.size()) *
                                                            confidence_level);
                    adjustor->set_alpha(static_cast<VALUE_TYPE>(adjustor->alphas_[abs_error_i]), true, false);

                    for (unsigned int j = num_train_examples; j < num_all_examples; ++j) {
                        if (filter->global_bsf_distances[j] < adjustor->predict(filter->global_pred_distances[j])) {
                            cnt_prune += 1;
                        }
                    }
                } else {
                    for (unsigned int j = num_train_examples; j < num_all_examples; ++j) {
                        if (filter->global_bsf_distances[j] < filter->global_pred_distances[j]) {
                            cnt_prune += 1;
                        }
                    }
                }

                (*filter_pruning_ratios)[i] =
                        static_cast<VALUE_TYPE>(cnt_prune) / (num_all_examples - num_train_examples);

                clog_info(CLOG(CLOGGER_ID), "allocator trial filter %d - pruning ratio %f",
                          filter->id, (*filter_pruning_ratios)[i]);
            }
        }
    }

    return nullptr;
}


static bool compDecreFilterNSeries(NeuralFilter *filter_1, NeuralFilter *filter_2) {
    return filter_1->node_size > filter_2->node_size;
}


void estimatePruningRatios(FilterAllocator *allocator, Config const *config) {
    std::sort(allocator->filters, allocator->filters + allocator->num_filters, compDecreFilterNSeries);

    ID_TYPE end_i_exclusive = allocator->num_filters;
    while (allocator->filters[end_i_exclusive - 1]->node_size < config->filter_min_leaf_size &&
           end_i_exclusive > 1) {
        end_i_exclusive -= 1;
    }

    ID_TYPE offset = 0;
    ID_TYPE step = end_i_exclusive / config->filter_trial_nnode;

    auto sampled_filter_idx = make_reserved<ID_TYPE>(config->filter_trial_nnode);
    for (ID_TYPE sample_i = 0; sample_i < config->filter_trial_nnode; ++sample_i) {
        sampled_filter_idx.push_back(offset + sample_i * step);
    }

    auto filter_pruning_ratios = make_reserved<VALUE_TYPE>(config->filter_trial_nnode);
    for (ID_TYPE i = 0; i < config->filter_trial_nnode; ++i) {
        filter_pruning_ratios.push_back(0);
    }

#ifdef FINE_TIMING
    struct timespec start_timestamp, stop_timestamp;
    TimeDiff time_diff;
    clock_code = clock_gettime(CLK_ID, &start_timestamp);
#endif
    unsigned int max_threads_train = config->max_threads_train;
    TrialCache trial_caches[max_threads_train];
    pthread_t trial_threads[max_threads_train];

    ID_TYPE shared_filter_pos = 0;
    unsigned int filter_block_size = (allocator->num_filters + max_threads_train - 1) / max_threads_train;

    for (unsigned int j = 0; j < max_threads_train; ++j) {
        trial_caches[j].config = config;
        trial_caches[j].filters = allocator->filters;

        trial_caches[j].num_samples = sampled_filter_idx.size();
        trial_caches[j].sampled_pos_vec = &sampled_filter_idx;
        trial_caches[j].filter_pruning_ratios = &filter_pruning_ratios;

        trial_caches[j].filter_block_size = filter_block_size;
        trial_caches[j].shared_filter_pos = &shared_filter_pos;

        trial_caches[j].confidence_level = config->filter_trial_confidence_level;
        trial_caches[j].num_all_examples = allocator->num_global_examples;
        trial_caches[j].num_train_examples = allocator->num_global_train_examples;

        trial_caches[j].stream_id = j;

        pthread_create(&trial_threads[j], nullptr, trialThread, (void *) &trial_caches[j]);
    }

    for (unsigned int j = 0; j < max_threads_train; ++j) {
        pthread_join(trial_threads[j], nullptr);
    }
#ifdef FINE_TIMING
    clock_code = clock_gettime(CLK_ID, &stop_timestamp);
    getTimeDiff(&time_diff, start_timestamp, stop_timestamp);
    clog_info(CLOG(CLOGGER_ID), "trial - trained %d filters = %ld.%lds",
              sampled_filter_idx.size(), time_diff.tv_sec, time_diff.tv_nsec);
#endif
    ID_TYPE num_trials = config->filter_trial_nnode;
    assert(num_trials == filter_pruning_ratios.size());

    for (float filter_pruning_ratio: filter_pruning_ratios) {
        if (filter_pruning_ratio >= 0) {
            allocator->estimated_filter_pruning_ratio += filter_pruning_ratio;
        } else {
            num_trials -= 1;
        }
    }
    if (num_trials > 0) {
        allocator->estimated_filter_pruning_ratio /= filter_pruning_ratios.size();

        clog_info(CLOG(CLOGGER_ID), "allocator trial - pruning_ratio estimated %f from %d / %d filters",
                  allocator->estimated_filter_pruning_ratio, num_trials, config->filter_trial_nnode);
    } else {
        allocator->estimated_filter_pruning_ratio = 0;

        clog_error(CLOG(CLOGGER_ID), "allocator trial failed; no successful training among %d filters",
                   config->filter_trial_nnode);
    }
}


void estimateTimeImprovements(FilterAllocator *allocator, Config const *config) {
    for (ID_TYPE filter_i = 0; filter_i < allocator->num_filters; ++filter_i) {
        NeuralFilter *filter = allocator->filters[filter_i];

        double amortized_gpu_msps = allocator->gpu_ms / static_cast<double>(filter->node_size);

        if (amortized_gpu_msps > allocator->cpu_ms_per_series) {
            filter->estimated_gain = 0;
            continue;
        }

        ID_TYPE cnt_sax_prune = 0;
        for (ID_TYPE example_i = 0; example_i < allocator->num_global_examples; ++example_i) {
            if (filter->global_bsf_distances[example_i] < filter->global_node_distances[example_i]) {
                cnt_sax_prune += 1;
            }
        }
        filter->sax_prune_ratio = static_cast<VALUE_TYPE>(cnt_sax_prune) / allocator->num_global_examples;

        filter->estimated_gain = static_cast<VALUE_TYPE>(
                static_cast<double>(filter->node_size)
                * static_cast<double>((1 - filter->sax_prune_ratio) * allocator->estimated_filter_pruning_ratio)
                * (allocator->cpu_ms_per_series - amortized_gpu_msps));

        if (filter->estimated_gain < 0) {
            filter->estimated_gain = 0;
        }
    }
}


static bool compDecreFilterGain(NeuralFilter *filter_1, NeuralFilter *filter_2) {
    return filter_1->estimated_gain > filter_2->estimated_gain;
}


ID_TYPE selectNActivateFilters(FilterAllocator *allocator, Config const *config, bool is_trial) {
    VALUE_TYPE mt_data_gpu_memory_mb =
            ((VALUE_TYPE) (sizeof(VALUE_TYPE)) * config->series_length * config->filter_query_load_size
             * 2 // latent features, gradients
             * config->max_threads_train) / 1024 / 1024;
    clog_info(CLOG(CLOGGER_ID), "allocator preserved %.3fmb gpu for training", mt_data_gpu_memory_mb);

    VALUE_TYPE allocated_gpu_memory_mb = mt_data_gpu_memory_mb;
    ID_TYPE cnt_allocated_filters = 0;

    bool gain_based_allocation_failed = false;

    ID_TYPE filter_min_leaf_size = config->filter_min_leaf_size;
    if (is_trial) {
        if (filter_min_leaf_size > config->filter_min_leaf_size_default) {
            filter_min_leaf_size = config->filter_min_leaf_size_default;
        }
    }

    ID_TYPE num_deactivated = 0, num_filters = 0, num_trained = 0;
    for (ID_TYPE filter_i = 0; filter_i < allocator->num_filters; ++filter_i) {
        if (allocator->filters[filter_i]->is_activated) {
            num_deactivated += 1;
        }

        if (allocator->filters[filter_i]->is_trained) {
            num_trained += 1;
        }

        num_filters += 1;
        allocator->filters[filter_i]->is_activated = false;
    }
    clog_debug(CLOG(CLOGGER_ID), "allocator deactivated %d of %d filters; %d trained",
               num_deactivated, num_filters, num_trained);

    if (config->filter_allocate_is_gain && !is_trial) {
        estimatePruningRatios(allocator, config);
        estimateTimeImprovements(allocator, config);

        if (allocator->estimated_filter_pruning_ratio > 0) {
            std::sort(allocator->filters, allocator->filters + allocator->num_filters, compDecreFilterGain);

            if (allocator->filters[0]->estimated_gain < 0.1) {
                gain_based_allocation_failed = true;
                clog_error(CLOG(CLOGGER_ID), "allocator request gain-based assignment but did not observe any gains");
            } else {
                for (ID_TYPE filter_i = 0; filter_i < allocator->num_filters; ++filter_i) {
                    if (allocated_gpu_memory_mb + allocator->gpu_mem_mb <= allocator->available_gpu_mem_mb
                        && allocator->filters[filter_i]->estimated_gain > 0) {
                        if (allocator->filters[filter_i]->node_size > filter_min_leaf_size) {
                            if (allocator->filters[filter_i]->is_trained) {
                                allocator->filters[filter_i]->is_activated = true;

                                allocated_gpu_memory_mb += allocator->gpu_mem_mb;
                                cnt_allocated_filters += 1;

                                clog_debug(CLOG(CLOGGER_ID), "allocator activate %d filter %d gain %.3f (size %d > %d)",
                                           cnt_allocated_filters, allocator->filters[filter_i]->id,
                                           allocator->filters[filter_i]->estimated_gain,
                                           allocator->filters[filter_i]->node_size, filter_min_leaf_size);
                            } else {
                                clog_error(CLOG(CLOGGER_ID),
                                           "allocator failed to activate filter %d gain %.3f (size %d > %d), untrained",
                                           allocator->filters[filter_i]->id,
                                           allocator->filters[filter_i]->estimated_gain,
                                           allocator->filters[filter_i]->node_size, filter_min_leaf_size);
                            }
                        } else {
                            clog_error(CLOG(CLOGGER_ID), "allocator filter %d gain %.3f but size %d <= %d",
                                       allocator->filters[filter_i]->id, allocator->filters[filter_i]->estimated_gain,
                                       allocator->filters[filter_i]->node_size, filter_min_leaf_size);
                        }
                    } else {
                        break;
                    }
                }
            }

            clog_info(CLOG(CLOGGER_ID), "allocator assigned (upon estimated gains) %d filters using %fmb gpu",
                      cnt_allocated_filters, allocated_gpu_memory_mb);
        } else {
            gain_based_allocation_failed = true;
            clog_warn(CLOG(CLOGGER_ID), "allocator request gain-based assignment but estimations not available");
        }
    }

    if (!config->filter_allocate_is_gain || gain_based_allocation_failed || is_trial) {
        std::sort(allocator->filters, allocator->filters + allocator->num_filters, compDecreFilterNSeries);

        for (ID_TYPE filter_i = 0; filter_i < allocator->num_filters; ++filter_i) {
//            if (!is_trial) {
//                clog_debug(CLOG(CLOGGER_ID),
//                           "allocator check filter %d size %d threshold %d allocated %.3f of %.3f mem",
//                           allocator->filters[filter_i]->id, allocator->filters[filter_i]->node_size,
//                           filter_min_leaf_size, allocated_gpu_memory_mb, allocator->gpu_mem_mb);
//            }

            if (allocated_gpu_memory_mb + allocator->gpu_mem_mb <= allocator->available_gpu_mem_mb
                && allocator->filters[filter_i]->node_size >= filter_min_leaf_size) {
                if (is_trial) {
                    if (allocator->filters[filter_i]->is_trained) {
                        clog_error(CLOG(CLOGGER_ID), "allocator trial filter %d found trained",
                                   allocator->filters[filter_i]->id);
                    } else {
                        allocator->filters[filter_i]->is_activated = true;

                        allocated_gpu_memory_mb += allocator->gpu_mem_mb;
                        cnt_allocated_filters += 1;

                        clog_debug(CLOG(CLOGGER_ID), "allocator activate %d trial filter %d size %d > %d",
                                   cnt_allocated_filters, allocator->filters[filter_i]->id,
                                   allocator->filters[filter_i]->node_size, filter_min_leaf_size);
                    }
                } else { // train filters to deploy
                    if (config->load_filters) {
                        if (allocator->filters[filter_i]->is_trained) {
                            allocator->filters[filter_i]->is_activated = true;

                            allocated_gpu_memory_mb += allocator->gpu_mem_mb;
                            cnt_allocated_filters += 1;

                            clog_debug(CLOG(CLOGGER_ID), "allocator activate %d loaded filter %d, size %d >= %d",
                                       cnt_allocated_filters, allocator->filters[filter_i]->id,
                                       allocator->filters[filter_i]->node_size, filter_min_leaf_size);
                        } else {
                            assert(!allocator->filters[filter_i]->is_activated);

                            clog_debug(CLOG(CLOGGER_ID),
                                       "allocator failed to activate loaded but untrained filter %d, size %d >= %d",
                                       allocator->filters[filter_i]->id,
                                       allocator->filters[filter_i]->node_size, filter_min_leaf_size);
                        }
                    } else {
                        allocator->filters[filter_i]->is_activated = true;

                        allocated_gpu_memory_mb += allocator->gpu_mem_mb;
                        cnt_allocated_filters += 1;

                        // could be trained in trial runs
                        clog_debug(CLOG(CLOGGER_ID), "allocator activate %d%s filter %d, size %d > %d",
                                   cnt_allocated_filters, allocator->filters[filter_i]->is_trained ? " trained" : "",
                                   allocator->filters[filter_i]->id,
                                   allocator->filters[filter_i]->node_size, filter_min_leaf_size);
                    }
                }
            } else {
                if (config->load_filters) {
                    if (allocator->filters[filter_i]->is_activated) {
                        clog_info(CLOG(CLOGGER_ID), "allocator deactivate loaded filter %d, size %d < %d",
                                  allocator->filters[filter_i]->id,
                                  allocator->filters[filter_i]->node_size, filter_min_leaf_size);
                    }
                } else {
                    if (allocator->filters[filter_i]->is_activated) {
                        clog_info(CLOG(CLOGGER_ID), "allocator deactivate trial filter %d, size %d < %d",
                                  allocator->filters[filter_i]->id,
                                  allocator->filters[filter_i]->node_size, filter_min_leaf_size);
                    }
                }

                allocator->filters[filter_i]->is_activated = false;
            }
        }

        clog_info(CLOG(CLOGGER_ID), "allocator assigned (upon node sizes) %d%s filters using %fmb gpu",
                  cnt_allocated_filters, is_trial ? " trial" : "", allocated_gpu_memory_mb);
    }

    return cnt_allocated_filters;
}


ID_TYPE countActiveFilters(FilterAllocator *allocator) {
    ID_TYPE cnt_allocated_filters = 0;

    for (ID_TYPE filter_i = 0; filter_i < allocator->num_filters; ++filter_i) {
        if (allocator->filters[filter_i]->is_activated) {
            cnt_allocated_filters += 1;
        }
    }

    return cnt_allocated_filters;
}


template<class T>
static std::string array2str(T *values, ID_TYPE length) {
    return std::accumulate(values + 1,
                           values + length,
                           std::to_string(values[0]),
                           [](const std::string &a, T b) {
                               return a + " " + std::to_string(b);
                           });
}


void adjustErr4Recall(FilterAllocator *allocator, Config const *config) {
    ID_TYPE num_examples = config->filter_query_load_size;
    if (num_examples < 1) {
        num_examples = config->filter_num_synthetic_query_global;
    }

    ID_TYPE num_train_examples = num_examples * config->filter_train_val_split;
    ID_TYPE num_conformal_examples = num_examples - num_train_examples;

    auto nn_distances = make_reserved<VALUE_TYPE>(num_conformal_examples);
    auto nn_filter_ids = make_reserved<ID_TYPE>(num_conformal_examples);

    VALUE_TYPE max_nn = VALUE_MIN; // to filter failed conformal regressors

    for (ID_TYPE query_i = 0; query_i < num_conformal_examples; ++query_i) {
        nn_distances.push_back(VALUE_MAX);
        nn_filter_ids.push_back(-1);

        for (ID_TYPE filter_i = 0; filter_i < allocator->num_filters; ++filter_i) {
            VALUE_TYPE local_nn = allocator->filters[filter_i]->global_nn_distances[num_train_examples + query_i];
            if (allocator->filters[filter_i]->is_distance_squared) {
                local_nn = sqrtf(local_nn);
            }

            if (local_nn < nn_distances[query_i]) {
                nn_distances[query_i] = local_nn;
                nn_filter_ids[query_i] = filter_i;
            } else if (local_nn > max_nn) {
                max_nn = local_nn;
            }
        }
    }

    clog_info(CLOG(CLOGGER_ID), "allocator nn_distances = %s",
              array2str(nn_distances.data(), num_conformal_examples).c_str());
    clog_info(CLOG(CLOGGER_ID), "allocator nn_filter_ids = %s",
              array2str(nn_filter_ids.data(), num_conformal_examples).c_str());

    // two sentry points: recall at small error (recall_at_0_error, 0_error), recall at large error (0.999999, 42)
    // corresponding errors were already added
    auto *recalls = static_cast<VALUE_TYPE *>(malloc(sizeof(VALUE_TYPE) * num_conformal_examples + 2));
    allocator->validation_recalls = recalls;

    // recall at validation error intervals
    for (ID_TYPE sorted_error_i = 0; sorted_error_i < num_conformal_examples + 2; ++sorted_error_i) {
        ID_TYPE cnt_hit = 0;

        for (ID_TYPE query_i = 0; query_i < num_conformal_examples; ++query_i) {
            NeuralFilter *filter = allocator->filters[nn_filter_ids[query_i]];

            if (filter->is_activated) {
                if (filter->is_trained) {
                    auto abs_error_interval = static_cast<VALUE_TYPE>(
                            ((ConformalRegressor *) filter->conformal_adjustor)->alphas_[sorted_error_i]);

                    VALUE_TYPE bsf_distance = filter->global_bsf_distances[num_train_examples + query_i];
                    VALUE_TYPE pred_distance = filter->global_pred_distances[num_train_examples + query_i];

                    if (pred_distance - abs_error_interval <= bsf_distance) {
                        cnt_hit += 1;
//#ifdef FINE_PROFILING
//                        clog_debug(CLOG(CLOGGER_ID),
//                                   "allocator error %d query %d filter %d, pred %.4f - %.4f <= bsf %.4f, hit",
//                                   sorted_error_i, query_i, filter->id,
//                                   pred_distance, abs_error_interval, bsf_distance);
//#endif
                    } else {
//#ifdef FINE_PROFILING
//                        clog_debug(CLOG(CLOGGER_ID),
//                                   "allocator error %d query %d filter %d, pred %.4f - %.4f > bsf %.4f, miss",
//                                   sorted_error_i, query_i, filter->id,
//                                   pred_distance, abs_error_interval, bsf_distance);
//#endif
                    }
                } else {
                    filter->is_activated = false;
                    cnt_hit += 1;

                    clog_error(CLOG(CLOGGER_ID),
                               "allocator filter %d (contains nn %d) activated but not trained; deactivate",
                               filter->id, query_i);
                }
            } else {
                cnt_hit += 1;
            }
        }

        recalls[sorted_error_i] = static_cast<ERROR_TYPE>(cnt_hit) / num_conformal_examples;
    }

    if (recalls[num_conformal_examples + 1] < 0.99) {
        clog_error(CLOG(CLOGGER_ID), "allocator reached %.3f recall with max sentry; expected ~1",
                   recalls[num_conformal_examples + 1]);
    }
    recalls[num_conformal_examples + 1] = 1 - EPSILON_GAP;

    clog_info(CLOG(CLOGGER_ID), "allocator recalls = %s",
              array2str(recalls, num_conformal_examples + 2).c_str());

    // to adjust the recalls to be strictly increasing
    for (int backtrace_i = num_conformal_examples; backtrace_i >= 0; --backtrace_i) {
        if (recalls[backtrace_i] > recalls[backtrace_i + 1] - EPSILON_GAP) {
            recalls[backtrace_i] = recalls[backtrace_i + 1] - EPSILON_GAP;
        }
    }

    clog_info(CLOG(CLOGGER_ID), "allocator adjusted recalls = %s",
              array2str(recalls, num_conformal_examples + 2).c_str());

    std::vector<ERROR_TYPE> recalls_vec;
    for (ID_TYPE recall_i = 0; recall_i < num_conformal_examples + 2; ++recall_i) {
        recalls_vec.push_back(recalls[recall_i]);
    }

    if (config->filter_conformal_is_smoothen) {
        for (ID_TYPE filter_i = 0; filter_i < allocator->num_filters; ++filter_i) {
            if (allocator->filters[filter_i]->is_activated) {
                if (allocator->filters[filter_i]->is_trained) {
                    ((ConformalRegressor *) allocator->filters[filter_i]->conformal_adjustor)->fit_spline(
                            config->filter_conformal_smoothen_core, recalls_vec);
                } else {
                    allocator->filters[filter_i]->is_activated = false;
                    clog_error(CLOG(CLOGGER_ID), "allocator filter %d activated but not trained; deactivate",
                               allocator->filters[filter_i]->id);
                }
            }
        }

        VALUE_TYPE overflowed_err = -1;
        if (config->filter_conformal_recall <= recalls_vec[0]) {
            // sentry: recalls_vec[0] -> abs_error = 0
            overflowed_err = 0;
        } else if (config->filter_conformal_recall >= recalls_vec[num_conformal_examples + 1]) {
            // sentry: recalls_vec[num_conformal_instances + 1] -> abs_error = MAX_PRED
            overflowed_err = max_nn;
        }

        if (overflowed_err > -1) {
            clog_error(CLOG(CLOGGER_ID),
                       "allocator requested recall %.4f (out of %.4f to %.4f) unavailable; rectify with range %.4f",
                       config->filter_conformal_recall, recalls_vec[0], recalls_vec[num_conformal_examples + 1],
                       overflowed_err);
        }

        ConformalRegressor *conf_regressor;
        VALUE_TYPE abs_error;
        for (ID_TYPE filter_i = 0; filter_i < allocator->num_filters; ++filter_i) {
            if (allocator->filters[filter_i]->is_activated) {
                conf_regressor = (ConformalRegressor *) allocator->filters[filter_i]->conformal_adjustor;

                if (overflowed_err > -1) {
                    conf_regressor->set_alpha(overflowed_err, false, true);

                    conf_regressor->is_trial_ = false;
                    conf_regressor->is_fitted_ = true;
                } else {
                    conf_regressor->set_alpha_by_recall(config->filter_conformal_recall);
                    abs_error = conf_regressor->get_alpha();

                    if (VALUE_LEQ(abs_error, VALUE_EPSILON) || VALUE_G(abs_error, max_nn * 1.1)) {
                        allocator->filters[filter_i]->is_activated = false;

                        clog_error(CLOG(CLOGGER_ID), "allocator node %d suspicious abs_error %f at %f, filter disabled",
                                   allocator->filters[filter_i]->id,
                                   conf_regressor->get_alpha(), config->filter_conformal_recall);
                    } else {
                        clog_info(CLOG(CLOGGER_ID), "allocator filter %d set range %.3f for %.4f (min recall %.4f)",
                                  allocator->filters[filter_i]->id, conf_regressor->get_alpha(),
                                  config->filter_conformal_recall, recalls_vec[0]);
                    }
                }
            }
        }
    } else {
        ID_TYPE last_recall_i = num_conformal_examples + 1;
        for (ID_TYPE recall_i = num_conformal_examples; recall_i >= 0; --recall_i) {
            if (recalls[recall_i] < config->filter_conformal_recall) {
                last_recall_i = recall_i + 1;

                clog_info(CLOG(CLOGGER_ID), "allocator reached recall %f with error_i %f (%d/%d, 2 sentries included)",
                          recalls[last_recall_i],
                          static_cast<VALUE_TYPE>(last_recall_i) / (num_conformal_examples + 2),
                          last_recall_i,
                          num_conformal_examples + 2);
                break;
            }
        }

        for (ID_TYPE filter_i = 0; filter_i < allocator->num_filters; ++filter_i) {
            if (allocator->filters[filter_i]->is_activated) {
                if (allocator->filters[filter_i]->is_trained) {
                    ((ConformalRegressor *) allocator->filters[filter_i]->conformal_adjustor)->set_alpha_by_pos(
                            last_recall_i);

                    clog_info(CLOG(CLOGGER_ID), "allocator filter %d set range %.3f for %.4f (min %.4f)",
                              allocator->filters[filter_i]->id,
                              ((ConformalRegressor *) allocator->filters[filter_i]->conformal_adjustor)->get_alpha(),
                              config->filter_conformal_recall, recalls_vec[0]);
                } else {
                    allocator->filters[filter_i]->is_activated = false;
                    clog_error(CLOG(CLOGGER_ID), "allocator filter %d activated but not trained; deactivate",
                               allocator->filters[filter_i]->id);
                }
            }
        }
    }
}


void freeAllocator(FilterAllocator *allocator) {
    if (allocator->validation_recalls != nullptr) {
        free(allocator->validation_recalls);
        allocator->validation_recalls = nullptr;
    }
}
