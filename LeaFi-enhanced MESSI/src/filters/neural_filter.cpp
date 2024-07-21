/******************************************************************************
 * Copyright (c) 2024, the LeaFi author(s).
 ******************************************************************************/

#include "neural_filter.h"

#include <cmath>
#include <random>
#include <memory>
#include <vector>
#include <numeric>
#include <iostream>
#include <cstdlib>

#include <torch/torch.h>
#include <torch/cuda.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/core/Device.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAMacros.h>
#include <cuda_runtime_api.h>

extern "C" {
#include "str.h"
}

#include "lr_scheduler.h"
#include "series_dataset.h"
#include "static_variables.h"
#include "conformal_adjustor.h"

torch::Tensor predictions_device;
torch::Tensor global_tsr;


constexpr auto TORCH_VALUE_TYPE = torch::kFloat32;

template<class T>
std::vector<T> make_reserved(const ID_TYPE n) {
    std::vector<T> vec;
    vec.reserve(n);
    return vec;
}


struct MLP1 : torch::nn::Module {
    MLP1(unsigned int dim_input, unsigned int dim_latent, VALUE_TYPE dropout_p, VALUE_TYPE negative_slope)
            : dropout_p(dropout_p), negative_slope(negative_slope) {
        fc1 = register_module("fc1", torch::nn::Linear(dim_input, dim_latent));
        fc2 = register_module("fc2", torch::nn::Linear(dim_latent, 1));
    }

    torch::Tensor forward(const torch::Tensor &x) {
        torch::Tensor z1 = torch::leaky_relu(fc1->forward(x), negative_slope);
        return at::squeeze(fc2->forward(z1));
    }

    torch::Tensor infer(const torch::Tensor &x) {
        torch::Tensor z1 = torch::leaky_relu(fc1->forward(x), negative_slope);
        return at::squeeze(fc2->forward(z1));
    }

    VALUE_TYPE dropout_p, negative_slope;

    torch::nn::Linear fc1{nullptr}, fc2{nullptr};
};


void initGlobalVariables(Config const *config, unsigned int num_filters) {
    int device_id = config->device_id;
    unsigned int max_threads_train = config->max_threads_train;

    if (torch::cuda::is_available()) {
        auto *streams = new std::vector<at::cuda::CUDAStream>();

        for (unsigned int i = 0; i < max_threads_train; ++i) {
            at::cuda::CUDAStream new_stream = at::cuda::getStreamFromPool(false, device_id);

            auto new_id = new_stream.id();
            auto new_query = new_stream.query();
            auto new_priority = new_stream.priority();

            clog_debug(CLOG(CLOGGER_ID), "cuda - stream %d: id = %d, query = %d, priority = %d",
                       i, new_id, new_query, new_priority);

            streams->emplace_back(new_stream);
        }

        streams_global = streams;
        device_global = new torch::Device(torch::kCUDA, device_id);

        predictions_device = torch::full(num_filters, VALUE_MAX, torch::TensorOptions().dtype(torch::kFloat32)).to(
                *device_global, true);
    } else {
        streams_global = nullptr;
        device_global = new torch::Device(torch::kCPU);

        predictions_device = torch::full(num_filters, VALUE_MAX, torch::TensorOptions().dtype(torch::kFloat32)).to(
                *device_global, true);

        clog_error(CLOG(CLOGGER_ID), "GPU not available");
        exit(-1);
    }
}


void initInferInputs(Config const *config, VALUE_TYPE *query_series2filter) {
    if (torch::cuda::is_available()) {
        global_tsr = torch::from_blob(query_series2filter, {1, config->series_length},
                                      torch::TensorOptions().dtype(torch::kFloat32)).to(*device_global);

        c10::cuda::CUDACachingAllocator::emptyCache();
    } else {
        clog_error(CLOG(CLOGGER_ID), "train = preload cpu not implemented");
        exit(-1);
    }
}


NeuralFilter *initFilterInfo(const Config *config, ID_TYPE node_size, int filter_id) {
    auto *neural_filter = static_cast<NeuralFilter *>(malloc(sizeof(NeuralFilter)));

    neural_filter->id = filter_id;
    neural_filter->is_activated = false;

    neural_filter->series_length = config->series_length;
    neural_filter->dim_instance = config->series_length;

    neural_filter->net = nullptr;
    neural_filter->conformal_adjustor = nullptr;
    neural_filter->is_trained = false;

    neural_filter->num_global_query = config->filter_num_synthetic_query_global;
    neural_filter->global_queries_shared = nullptr;
    neural_filter->global_bsf_distances = nullptr;
    neural_filter->global_node_distances = nullptr;
    neural_filter->global_nn_distances = nullptr;
    neural_filter->global_pred_distances = nullptr;

    neural_filter->num_local_query = 0; // only for larger nodes
    neural_filter->local_queries = nullptr;
    neural_filter->local_nn_distances = nullptr;

    neural_filter->is_distance_squared = false;

    neural_filter->node_size = node_size;
    neural_filter->sax_prune_ratio = -1;
    neural_filter->estimated_gain = -1;

    return neural_filter;
}


void addFilterTrainQuery(NeuralFilter *neural_filter, VALUE_TYPE const *filter_train_queries,
                         unsigned int global_query_size, unsigned int local_query_size) {
    if (neural_filter->num_global_query > 0) {
        assert(neural_filter->num_global_query == global_query_size);
    } else {
        neural_filter->num_global_query = global_query_size;
    }
    assert(neural_filter->global_queries_shared == nullptr);
    neural_filter->global_queries_shared = (VALUE_TYPE *) filter_train_queries;

    assert(neural_filter->global_node_distances == nullptr);
    neural_filter->global_node_distances = static_cast<VALUE_TYPE *>(malloc(sizeof(VALUE_TYPE) * global_query_size));
    for (ID_TYPE i = 0; i < global_query_size; ++i) {
        neural_filter->global_node_distances[i] = -1;
    }

    assert(neural_filter->global_bsf_distances == nullptr);
    neural_filter->global_bsf_distances = static_cast<VALUE_TYPE *>(malloc(sizeof(VALUE_TYPE) * global_query_size));
    for (ID_TYPE i = 0; i < global_query_size; ++i) {
        neural_filter->global_bsf_distances[i] = -1;
    }

    assert(neural_filter->global_nn_distances == nullptr);
    neural_filter->global_nn_distances = static_cast<VALUE_TYPE *>(malloc(sizeof(VALUE_TYPE) * global_query_size));
    for (ID_TYPE i = 0; i < global_query_size; ++i) {
        neural_filter->global_nn_distances[i] = -1;
    }

    if (neural_filter->is_activated) {
        assert(neural_filter->local_queries == nullptr);
        neural_filter->local_queries = static_cast<VALUE_TYPE *>(aligned_alloc(
                256, sizeof(VALUE_TYPE) * neural_filter->series_length * local_query_size));

        assert(neural_filter->local_nn_distances == nullptr);
        neural_filter->local_nn_distances = static_cast<VALUE_TYPE *>(malloc(sizeof(VALUE_TYPE) * local_query_size));
        for (ID_TYPE i = 0; i < local_query_size; ++i) {
            neural_filter->local_nn_distances[i] = -1;
        }
    } else {
        assert(neural_filter->local_queries == nullptr);
        assert(neural_filter->local_nn_distances == nullptr);
        neural_filter->num_local_query = 0;
    }

    neural_filter->is_distance_squared = true;
}


void profileFilter(const Config *config, double *gpu_ms, VALUE_TYPE *gpu_mem_mb) {
    auto *neural_filter = static_cast<NeuralFilter *>(malloc(sizeof(NeuralFilter)));
    neural_filter->is_activated = true;

    c10::InferenceMode guard;
    MLP1 *net = new MLP1((int) config->series_length, config->dim_latent, config->dropout_p,
                         config->leaky_relu_negative_slope);
    net->to(*device_global);
    net->eval();
    neural_filter->net = (void *) net;

    if (config->is_filter_conformal) {
        auto conformal_adjustor = new ConformalRegressor(config->filter_conformal_core_type);
        conformal_adjustor->set_alpha(0, true);
        conformal_adjustor->is_trial_ = true;
        neural_filter->conformal_adjustor = (void *) conformal_adjustor;
    } else {
        neural_filter->conformal_adjustor = nullptr;
    }
    neural_filter->is_trained = true;
    size_t memory_size = 0;

    for (const auto &parameter: net->parameters()) {
        memory_size += parameter.nbytes() * 3; // parameters, 1x features, 1x gradients
    }

    for (const auto &buffer: net->buffers()) {
        memory_size += buffer.nbytes(); // 1x features
    }

    *gpu_mem_mb = static_cast<VALUE_TYPE>(memory_size) / (1024 * 1024);

    std::random_device rd;
    std::mt19937 e2(rd());
    std::uniform_real_distribution<> dist(0, 1);

    auto random_input = static_cast<VALUE_TYPE *>(aligned_alloc(sizeof(__m256),
                                                                sizeof(VALUE_TYPE) * config->series_length));
    for (ID_TYPE i = 0; i < config->series_length; ++i) {
        random_input[i] = dist(e2);
    }

    initInferInputs(config, random_input);

    auto trial_predictions = make_reserved<VALUE_TYPE>(config->filter_trial_iterations);

    auto start = std::chrono::high_resolution_clock::now();

    for (ID_TYPE trial_i = 0; trial_i < config->filter_trial_iterations; ++trial_i) {
        trial_predictions.push_back(inferFilter(neural_filter));
    }

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    *gpu_ms = static_cast<double>(duration.count()) / static_cast<double_t>(config->filter_trial_iterations);

    clog_info(CLOG(CLOGGER_ID), "filter gpu mem = %fMB, time = %fmus", *gpu_mem_mb, *gpu_ms);

    free(random_input);
}


int validateDataset(VALUE_TYPE const *values, unsigned int len) {
    for (unsigned int i = 0; i < len; ++i) {
        if (std::isnan(values[i])) {
            return -1;
        } else if (VALUE_LEQ(values[i], 0) || VALUE_GEQ(values[i], VALUE_MAX)) {
            return -2;
        }
    }

    return 1;
}


int trainConformalAdjustor(ConformalRegressor *conformal_adjuster, ID_TYPE num_conformal_examples,
                           VALUE_TYPE *predictions_ptr, VALUE_TYPE *targets_ptr, ID_TYPE filter_id) {
    int return_code = 1;

    std::vector<ERROR_TYPE> residuals;
    residuals.reserve(num_conformal_examples + 2);

    ERROR_TYPE abs_error_mean = 0, abs_error_std = 0, max_pred = -1;
    ID_TYPE num_errors = 0;

    for (ID_TYPE query_i = 0; query_i < num_conformal_examples; ++query_i) {
        if (predictions_ptr[query_i] > VALUE_MIN && predictions_ptr[query_i] < VALUE_MAX &&
            VALUE_NEQ(predictions_ptr[query_i], 0)) {
            ERROR_TYPE abs_error = fabs(predictions_ptr[query_i] - targets_ptr[query_i]);

            residuals.push_back(abs_error);

            abs_error_mean += abs_error;
            num_errors += 1;

            if (predictions_ptr[query_i] > max_pred) {
                max_pred = predictions_ptr[query_i];
            } else if (targets_ptr[query_i] > max_pred) {
                max_pred = targets_ptr[query_i];
            }
        } else {
            clog_error(CLOG(CLOGGER_ID), "train filter %d - outlier pred %d = %f",
                       filter_id, query_i, predictions_ptr[query_i]);
            return_code = -1;
        }
    }

    abs_error_mean /= num_errors;

    for (ID_TYPE residual_i = 0; residual_i < num_errors; ++residual_i) {
        ERROR_TYPE deviation = residuals[residual_i] - abs_error_mean;
        abs_error_std += deviation * deviation;
    }

    abs_error_std = sqrt(abs_error_std / num_errors);
    ERROR_TYPE error_upper_bound = abs_error_mean + abs_error_std * 3 + VALUE_MARGIN;

    if (error_upper_bound > max_pred) {
        error_upper_bound = max_pred;
    }

    residuals.push_back(0);
    residuals.push_back(error_upper_bound);

    conformal_adjuster->fit(residuals);

    return return_code;
}


int trainNeuralFilter(Config const *config, NeuralFilter *filter, unsigned int stream_id, char *sax_str) {
    if (filter->is_trained) {
        return 1;
    } else {
        assert(filter->is_activated);
    }

    int dim_instances = (int) config->series_length;

    ID_TYPE num_global_query = filter->num_global_query;
    ID_TYPE num_global_train = num_global_query * config->filter_train_val_split;
    ID_TYPE num_global_valid = num_global_query - num_global_train;

    ID_TYPE num_local_query = filter->num_local_query;
    ID_TYPE num_local_train = num_local_query * config->filter_train_val_split;
    ID_TYPE num_local_valid = num_local_query - num_local_train;

    if (num_local_query < 0) { num_local_query = 0; }
    if (num_local_train < 0) { num_local_train = 0; }
    if (num_local_valid < 0) { num_local_valid = 0; }

    ID_TYPE num_train_examples = num_global_train + num_local_train;
    ID_TYPE num_valid_examples = num_global_valid + num_local_valid;
    ID_TYPE num_examples = num_train_examples + num_valid_examples;

    ID_TYPE batch_size = config->batch_size;
    if (batch_size > num_train_examples || batch_size <= 0) {
        batch_size = num_train_examples;
    }

    if (filter->is_distance_squared) {
        for (ID_TYPE query_i = 0; query_i < num_global_query; query_i += 1) {
            filter->global_node_distances[query_i] = sqrtf(filter->global_node_distances[query_i]);
            filter->global_bsf_distances[query_i] = sqrtf(filter->global_bsf_distances[query_i]);
            filter->global_nn_distances[query_i] = sqrtf(filter->global_nn_distances[query_i]);
        }

        if (num_local_query > 0) {
            for (ID_TYPE query_i = 0; query_i < num_local_query; query_i += 1) {
                filter->local_nn_distances[query_i] = sqrtf(filter->local_nn_distances[query_i]);
            }
        }

        filter->is_distance_squared = false;
    }
#ifdef FINE_PROFILING
    std::string node_str = std::accumulate(filter->global_node_distances + 1,
                                           filter->global_node_distances + num_global_query,
                                           std::to_string(filter->global_node_distances[0]),
                                           [](const std::string &a, float b) {
                                               return a + " " + std::to_string(b);
                                           });
    clog_info(CLOG(CLOGGER_ID), "train stream %d node %s filter %d - node = %s",
              stream_id, sax_str, filter->id, node_str.data());

    std::string bsf_str = std::accumulate(filter->global_bsf_distances + 1,
                                          filter->global_bsf_distances + num_global_query,
                                          std::to_string(filter->global_bsf_distances[0]),
                                          [](const std::string &a, float b) {
                                              return a + " " + std::to_string(b);
                                          });
    clog_info(CLOG(CLOGGER_ID), "train stream %d node %s filter %d - bsf = %s",
              stream_id, sax_str, filter->id, bsf_str.data());

    std::string global_nn_str = std::accumulate(filter->global_nn_distances + 1,
                                                filter->global_nn_distances + num_global_query,
                                                std::to_string(filter->global_nn_distances[0]),
                                                [](const std::string &a, float b) {
                                                    return a + " " + std::to_string(b);
                                                });
    clog_info(CLOG(CLOGGER_ID), "train stream %d node %s filter %d - global nn = %s",
              stream_id, sax_str, filter->id, global_nn_str.data());

    if (num_local_query > 0) {
        std::string local_nn_str = std::accumulate(filter->local_nn_distances + 1,
                                                   filter->local_nn_distances + num_local_query,
                                                   std::to_string(filter->local_nn_distances[0]),
                                                   [](const std::string &a, float b) {
                                                       return a + " " + std::to_string(b);
                                                   });
        clog_info(CLOG(CLOGGER_ID), "train stream %d node %s filter %d - local nn = %s",
                  stream_id, sax_str, filter->id, local_nn_str.data());
    }

#endif
    if (validateDataset(filter->global_nn_distances, num_global_query) < 0) {
        clog_error(CLOG(CLOGGER_ID), "train stream %d node %s filter %d - invalid global nn",
                   stream_id, sax_str, filter->id);
        return false;
    }
    if (num_local_query > 0 && validateDataset(filter->local_nn_distances, num_local_query) < 0) {
        clog_error(CLOG(CLOGGER_ID), "train stream %d node %s filter %d - invalid local nn",
                   stream_id, sax_str, filter->id);
        return false;
    }

    auto all_data = static_cast<VALUE_TYPE *>(aligned_alloc(
            256, sizeof(VALUE_TYPE) * config->series_length * num_examples));
    VALUE_TYPE *train_data = all_data;
    VALUE_TYPE *validation_data = all_data + config->series_length * num_train_examples;

    ID_TYPE memcpy_num_values = config->series_length * num_global_train;
    memcpy(all_data,
           filter->global_queries_shared,
           sizeof(VALUE_TYPE) * memcpy_num_values);
    ID_TYPE memcpy_dest_ptr_offset = memcpy_num_values;

    if (num_local_query > 0) {
        memcpy_num_values = config->series_length * num_local_train;
        memcpy(all_data + memcpy_dest_ptr_offset,
               filter->local_queries,
               sizeof(VALUE_TYPE) * memcpy_num_values);
        memcpy_dest_ptr_offset += memcpy_num_values;
    }

    memcpy_num_values = config->series_length * num_global_valid;
    memcpy(all_data + memcpy_dest_ptr_offset,
           filter->global_queries_shared + config->series_length * num_global_train,
           sizeof(VALUE_TYPE) * memcpy_num_values);
    memcpy_dest_ptr_offset += memcpy_num_values;

    if (num_local_query > 0) {
        memcpy_num_values = config->series_length * num_local_valid;
        memcpy(all_data + memcpy_dest_ptr_offset,
               filter->local_queries + config->series_length * num_local_train,
               sizeof(VALUE_TYPE) * memcpy_num_values);
    }

    auto all_targets = static_cast<VALUE_TYPE *>(malloc(sizeof(VALUE_TYPE) * num_examples));
    VALUE_TYPE *train_targets = all_targets;
    VALUE_TYPE *validation_targets = all_targets + num_train_examples;
    VALUE_TYPE *conformal_targets = all_targets + num_train_examples + num_valid_examples;

    memcpy(all_targets,
           filter->global_nn_distances,
           sizeof(VALUE_TYPE) * num_global_train);
    memcpy_dest_ptr_offset = num_global_train;

    if (num_local_query > 0) {
        memcpy(all_targets + memcpy_dest_ptr_offset,
               filter->local_nn_distances,
               sizeof(VALUE_TYPE) * num_local_train);
        memcpy_dest_ptr_offset += num_local_train;
    }

    memcpy(all_targets + memcpy_dest_ptr_offset,
           filter->global_nn_distances + num_global_train,
           sizeof(VALUE_TYPE) * num_global_valid);
    memcpy_dest_ptr_offset += num_global_valid;

    if (num_local_query > 0) {
        memcpy(all_targets + memcpy_dest_ptr_offset,
               filter->local_nn_distances + num_local_train,
               sizeof(VALUE_TYPE) * num_local_valid);
    }

    {
        ID_TYPE device_id = config->device_id;
        VALUE_TYPE max_norm = config->max_norm, norm_type = config->norm_type;

        at::cuda::CUDAStream local_stream = (*streams_global)[stream_id];
        at::cuda::setCurrentCUDAStream(local_stream);
        at::cuda::CUDAStreamGuard guard(local_stream);

        auto train_dataset = SeriesDataset(train_data, train_targets,
                                           num_train_examples, dim_instances,
                                           *device_global);
        auto train_dataloader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
                train_dataset.map(torch::data::transforms::Stack<>()), batch_size);

        auto validation_series_tensor = torch::from_blob(
                validation_data, {num_valid_examples, dim_instances},
                torch::TensorOptions().dtype(torch::kFloat32)).to(*device_global);
        auto validation_targets_tensor = torch::from_blob(
                validation_targets, num_valid_examples,
                torch::TensorOptions().dtype(torch::kFloat32)).to(*device_global);

        MLP1 *net = new MLP1(dim_instances, config->dim_latent, config->dropout_p, config->leaky_relu_negative_slope);
        net->to(*device_global);

        torch::optim::SGD optimizer(net->parameters(), config->learning_rate);

        size_t epoch_start_early_stop_check = config->max_epochs / 2;
        size_t initial_cooldown_epochs = epoch_start_early_stop_check;
        ReduceLROnPlateau lr_scheduler = ReduceLROnPlateau(optimizer, initial_cooldown_epochs);

        std::unordered_map<std::string, torch::Tensor> best_model_state;
        VALUE_TYPE best_validation_loss = VALUE_MAX;
        ID_TYPE best_validation_epoch = -1;

#ifdef FINE_PROFILING
        std::vector<VALUE_TYPE> global_train_losses, local_train_losses, global_validation_losses;
        global_train_losses.reserve(config->max_epochs);
        global_validation_losses.reserve(config->max_epochs);
        local_train_losses.reserve(num_train_examples / batch_size + 1);
#endif
        size_t epoch_stop = config->max_epochs;

        for (size_t epoch = 0; epoch < config->max_epochs; ++epoch) {
            for (auto &batch: *train_dataloader) {
                auto batch_data = batch.data, batch_target = batch.target;

                optimizer.zero_grad();

                torch::Tensor prediction = net->forward(batch_data);
                torch::Tensor loss = torch::mse_loss(prediction, batch_target);

                loss.backward();
                auto norm = torch::nn::utils::clip_grad_norm_(net->parameters(), max_norm, norm_type);
                optimizer.step();
#ifdef FINE_PROFILING
                local_train_losses.emplace_back(loss.detach().item<float>());
#endif
            }
#ifdef FINE_PROFILING
            global_train_losses.emplace_back(std::accumulate(local_train_losses.begin(), local_train_losses.end(), 0.0)
                                             / (VALUE_TYPE) local_train_losses.size());
            local_train_losses.clear();
#endif
            {
                torch::NoGradGuard no_grad;

                auto prediction = net->forward(validation_series_tensor);
                auto loss = torch::mse_loss(prediction, validation_targets_tensor);

                VALUE_TYPE validation_loss = loss.detach().item<VALUE_TYPE>();
#ifdef FINE_PROFILING
                global_validation_losses.emplace_back(validation_loss);
#endif
                if (epoch > epoch_start_early_stop_check) {
                    if (best_validation_loss > validation_loss) {
                        best_validation_loss = validation_loss;
                        best_validation_epoch = epoch;

                        for (const auto &pair: net->named_parameters()) {
                            best_model_state[pair.key()] = pair.value().clone();
                        }
                    }

                    LR_RETURN_CODE code = lr_scheduler.check_and_schedule(validation_loss, epoch);

                    if (code == EARLY_STOP) {
                        for (auto &pair: best_model_state) {
                            net->named_parameters()[pair.first].detach_();
                            net->named_parameters()[pair.first].copy_(pair.second);
                        }

                        clog_info(CLOG(CLOGGER_ID),
                                  "train stream %d node %s filter %d - early stop at %d, %.4f, best at %d, %.4f; model restored",
                                  stream_id, sax_str, filter->id,
                                  epoch, validation_loss, best_validation_epoch, best_validation_loss);

                        epoch_stop = epoch;
                        epoch = config->max_epochs + 1;
                    }
                }
            }
        }

#ifdef FINE_PROFILING
        std::string loss_str = std::accumulate(global_train_losses.begin() + 1,
                                               global_train_losses.begin() + epoch_stop,
                                               std::to_string(global_train_losses[0]),
                                               [](const std::string &a, float b) {
                                                   return a + " " + std::to_string(b);
                                               });
        clog_info(CLOG(CLOGGER_ID), "train stream %d node %s filter %d - train losses = %s",
                  stream_id, sax_str, filter->id, loss_str.data());

        loss_str = std::accumulate(global_validation_losses.begin() + 1,
                                   global_validation_losses.begin() + epoch_stop,
                                   std::to_string(global_validation_losses[0]),
                                   [](const std::string &a, float b) {
                                       return a + " " + std::to_string(b);
                                   });
        clog_info(CLOG(CLOGGER_ID), "train stream %d node %s filter %d - validation losses = %s",
                  stream_id, sax_str, filter->id, loss_str.data());
#endif
        optimizer.zero_grad();
        for (const auto &parameter: net->parameters()) {
            parameter.requires_grad_(false);
        }

        filter->net = (void *) net;
        net->eval();
        torch::NoGradGuard no_grad;

        auto all_data_tensor = torch::from_blob(all_data, {num_examples, dim_instances},
                                                torch::TensorOptions().dtype(torch::kFloat32)).to(*device_global);

        auto predictions = net->forward(all_data_tensor).detach().to(torch::Device(torch::kCPU));
        VALUE_TYPE *prediction_ptr = predictions.accessor<VALUE_TYPE, 1>().data();

#ifdef FINE_PROFILING
        std::string prediction_str = std::accumulate(prediction_ptr + 1,
                                                     prediction_ptr + num_examples,
                                                     std::to_string(prediction_ptr[0]),
                                                     [](const std::string &a, float b) {
                                                         return a + " " + std::to_string(b);
                                                     });
        clog_info(CLOG(CLOGGER_ID), "train stream %d node %s filter %d - pred = %s",
                  stream_id, sax_str, filter->id, prediction_str.data());
#endif

        filter->global_pred_distances = static_cast<VALUE_TYPE *>(malloc(sizeof(VALUE_TYPE) * num_global_query));

        memcpy(filter->global_pred_distances, prediction_ptr, sizeof(VALUE_TYPE) * num_global_train);
        memcpy(filter->global_pred_distances + num_global_train,
               prediction_ptr + num_global_train + num_local_train,
               sizeof(VALUE_TYPE) * num_global_valid);

        if (validateDataset(prediction_ptr, num_examples) < 0) {
            clog_error(CLOG(CLOGGER_ID), "train stream %d node %s filter %d - invalid pred",
                       stream_id, sax_str, filter->id);
        } else {
            if (config->is_filter_conformal) {
                ID_TYPE num_conformal_examples = num_global_valid;

                auto *conformal_adjuster = new ConformalRegressor(config->filter_conformal_core_type);

                trainConformalAdjustor(conformal_adjuster, num_conformal_examples,
                                       filter->global_pred_distances + num_global_train,
                                       filter->global_nn_distances + num_global_train, filter->id);

                filter->conformal_adjustor = (void *) conformal_adjuster;
            }

            filter->is_trained = true;
        }
    }

    c10::cuda::CUDACachingAllocator::emptyCache();

    free(all_data);
    free(all_targets);

    return filter->is_trained;
}


int logNeuralFilter(Config const *config, NeuralFilter *neural_filter, unsigned int stream_id, char *sax_str) {
    if (neural_filter->is_distance_squared) {
        for (ID_TYPE query_i = 0; query_i < neural_filter->num_global_query; ++query_i) {
            neural_filter->global_node_distances[query_i] = sqrtf(neural_filter->global_node_distances[query_i]);
            neural_filter->global_bsf_distances[query_i] = sqrtf(neural_filter->global_bsf_distances[query_i]);
            neural_filter->global_nn_distances[query_i] = sqrtf(
                    neural_filter->global_nn_distances[query_i]);
        }

        neural_filter->is_distance_squared = false;
    }

#ifdef FINE_PROFILING
    std::string node_str = std::accumulate(neural_filter->global_node_distances + 1,
                                           neural_filter->global_node_distances + neural_filter->num_global_query,
                                           std::to_string(neural_filter->global_node_distances[0]),
                                           [](const std::string &a, float b) {
                                               return a + ", " + std::to_string(b);
                                           });
    clog_info(CLOG(CLOGGER_ID), "train stream %d node %s filter %d - node = %s",
              stream_id, sax_str, neural_filter->id, node_str.data());

    std::string bsf_str = std::accumulate(neural_filter->global_bsf_distances + 1,
                                          neural_filter->global_bsf_distances + neural_filter->num_global_query,
                                          std::to_string(neural_filter->global_bsf_distances[0]),
                                          [](const std::string &a, float b) {
                                              return a + ", " + std::to_string(b);
                                          });
    clog_info(CLOG(CLOGGER_ID), "train stream %d node %s filter %d - bsf = %s",
              stream_id, sax_str, neural_filter->id, bsf_str.data());

    std::string ann_str = std::accumulate(neural_filter->global_nn_distances + 1,
                                          neural_filter->global_nn_distances + neural_filter->num_global_query,
                                          std::to_string(neural_filter->global_nn_distances[0]),
                                          [](const std::string &a, float b) {
                                              return a + ", " + std::to_string(b);
                                          });
    clog_info(CLOG(CLOGGER_ID), "train stream %d node %s filter %d - global nn = %s",
              stream_id, sax_str, neural_filter->id, ann_str.data());
#endif

    return neural_filter->is_trained;
}


bool isFilterActive(NeuralFilter *neural_filter) {
    return neural_filter != nullptr && neural_filter->is_activated && neural_filter->is_trained;
}


VALUE_TYPE inferFilter(NeuralFilter *neural_filter) {
    if (neural_filter->conformal_adjustor != nullptr) {
        auto pred = ((ConformalRegressor *) neural_filter->conformal_adjustor)->predict(
                ((MLP1 *) neural_filter->net)->infer(global_tsr).item<VALUE_TYPE>());
        return pred * pred;
    } else {
        auto pred = ((MLP1 *) neural_filter->net)->infer(global_tsr).item<VALUE_TYPE>();
        return pred * pred;
    }
}


VALUE_TYPE checkNInferFilter(NeuralFilter *neural_filter) {
    if (isFilterActive(neural_filter)) {
        return inferFilter(neural_filter);
    } else {
        return VALUE_MAX;
    }
}


typedef struct InferCache {
    unsigned int stream_id;

    unsigned int num_filters;
    ID_TYPE *shared_filter_id;
    unsigned int filter_block_size;

    NeuralFilter **filters;
} InferCache;


void *inferThread(void *cache) {
    InferCache *inferCache = (InferCache *) cache;

    unsigned int stream_id = inferCache->stream_id;
    unsigned int block_size = inferCache->filter_block_size;
    unsigned int num_filters = inferCache->num_filters;
    ID_TYPE *shared_filter_id = inferCache->shared_filter_id;
    NeuralFilter const *const *filters = (NeuralFilter const *const *) inferCache->filters;

    ID_TYPE start_filter_id, stop_filter_id;

    clog_debug(CLOG(CLOGGER_ID), "infer s%d - before", stream_id);

    torch::NoGradGuard no_grad;
    at::cuda::CUDAStream local_stream = (*streams_global)[stream_id];
    at::cuda::setCurrentCUDAStream(local_stream);
    at::cuda::CUDAStreamGuard guard(local_stream); // compiles with cuda

    while ((start_filter_id = __sync_fetch_and_add(shared_filter_id, block_size)) < num_filters) {
        stop_filter_id = start_filter_id + block_size;
        if (stop_filter_id > num_filters) {
            stop_filter_id = num_filters;
        }

        for (unsigned int i = start_filter_id; i < stop_filter_id; ++i) {
            if (filters[i]->is_trained) {
                predictions_device[i + 1] = ((MLP1 *) filters[i]->net)->infer(global_tsr);

                std::ostringstream stream;
                stream << ((MLP1 *) filters[i]->net)->infer(global_tsr);
                std::string tensor_string = stream.str();

                clog_debug(CLOG(CLOGGER_ID), "infer s%d - filter %d = %f, %s",
                           stream_id, i, predictions_device[i + 1].item<VALUE_TYPE>(), tensor_string.data());
            }
        }
    }

    clog_debug(CLOG(CLOGGER_ID), "infer s%d - after", inferCache->stream_id);

    return nullptr;
}


pthread_t *infer_threads;


void inferBatchConcurrent(NeuralFilter **filters, unsigned int num_filters, unsigned int num_streams) {
    InferCache inferCache[num_streams];
    unsigned int filter_block_size = 32 * 16;
    ID_TYPE shared_filter_id = 0;

    for (unsigned int i = 0; i < num_streams; ++i) {
        inferCache[i].stream_id = i;

        inferCache[i].num_filters = num_filters;
        inferCache[i].shared_filter_id = &shared_filter_id;
        inferCache[i].filter_block_size = filter_block_size;

        inferCache[i].filters = filters;
    }

    pthread_t *infer_threads_local = static_cast<pthread_t *>(malloc(sizeof(pthread_t) * num_streams));

    for (unsigned int i = 0; i < num_streams; ++i) {
        pthread_create(&infer_threads_local[i], nullptr, inferThread, (void *) &inferCache[i]);
    }

    infer_threads = infer_threads_local;
}


VALUE_TYPE *syncPredictionsConcurrent(unsigned int num_streams) {
    for (unsigned int i = 0; i < num_streams; ++i) {
        pthread_join(infer_threads[i], nullptr);
    }

    free(infer_threads);
    infer_threads = nullptr;

    return (VALUE_TYPE *) predictions_device.to(torch::TensorOptions().device(torch::kCPU),
            /*non_blocking*/ false, /*copy*/ true).data_ptr();
}


void inferBatchNonblocking(NeuralFilter **filters, unsigned int num_filters, unsigned int num_streams,
                           unsigned int unroll_degree) {
    unsigned int current_stream = 0, unroll_folds = num_filters / unroll_degree;
    torch::NoGradGuard no_grad;

    for (unsigned int i = 0; i < unroll_folds * unroll_degree; i += unroll_degree) {
        at::cuda::CUDAStream local_stream = (*streams_global)[current_stream];
        at::cuda::setCurrentCUDAStream(local_stream);
        at::cuda::CUDAStreamGuard guard(local_stream); // compiles with cuda

        for (unsigned int j = i; j < i + unroll_degree; ++j) {
            if (filters[j]->is_trained) {
                predictions_device[j + 1] = ((MLP1 *) filters[j]->net)->infer(global_tsr);
            }
        }

        current_stream = (current_stream + 1) % num_streams;
    }

    at::cuda::CUDAStream local_stream = (*streams_global)[current_stream];
    at::cuda::setCurrentCUDAStream(local_stream);
    at::cuda::CUDAStreamGuard guard(local_stream); // compiles with cuda

    for (unsigned int i = unroll_folds * unroll_degree; i < num_filters; ++i) {
        if (filters[i]->is_trained) {
            predictions_device[i + 1] = ((MLP1 *) filters[i]->net)->infer(global_tsr);
        }
    }
}


VALUE_TYPE *syncPredictionsNonblocking(int device_id) {
    torch::cuda::synchronize(device_id);

    return (VALUE_TYPE *) predictions_device.to(torch::TensorOptions().device(torch::kCPU),
            /*non_blocking*/ false, /*copy*/ true).data_ptr();
}


VALUE_TYPE *inferBatchFuture(NeuralFilter **filters, unsigned int num_filters, unsigned int num_streams,
                             unsigned int unroll_degree, int device_id) {
    unsigned int current_stream = 0, unroll_folds = num_filters / unroll_degree;
    torch::NoGradGuard no_grad;

    for (unsigned int i = 0; i < unroll_folds * unroll_degree; i += unroll_degree) {
        at::cuda::CUDAStream local_stream = (*streams_global)[current_stream];
        at::cuda::setCurrentCUDAStream(local_stream);
        at::cuda::CUDAStreamGuard guard(local_stream); // compiles with cuda

        for (unsigned int j = i; j < i + unroll_degree; ++j) {
            if (filters[j]->is_trained) {
                predictions_device[j + 1] = ((MLP1 *) filters[j]->net)->infer(global_tsr);
            }
        }

        current_stream = (current_stream + 1) % num_streams;
    }

    {
        at::cuda::CUDAStream local_stream = (*streams_global)[current_stream];
        at::cuda::setCurrentCUDAStream(local_stream);
        at::cuda::CUDAStreamGuard guard(local_stream); // compiles with cuda

        for (unsigned int i = unroll_folds * unroll_degree; i < num_filters; ++i) {
            if (filters[i]->is_trained) {
                predictions_device[i + 1] = ((MLP1 *) filters[i]->net)->infer(global_tsr);
            }
        }
    }

    for (unsigned int i = 0; i < num_streams; ++i) {
        C10_CUDA_CHECK(cudaStreamSynchronize((*streams_global)[i])); // compiles with cuda
    }

    return (VALUE_TYPE *) predictions_device.to(torch::TensorOptions().device(torch::kCPU),
            /*non_blocking*/ false, /*copy*/ true).data_ptr();
}


void dumpFilter(const Config *config, NeuralFilter *neural_filter, char *filter_dump_prefix) {
    char *filter_data_filepath = concat(2, filter_dump_prefix, config->dump_filename_postfix);
    FILE *filter_file;
    filter_file = fopen(filter_data_filepath, "wb");
    assert(filter_file != nullptr);

    size_t nitems = fwrite(&neural_filter->id, sizeof(int), 1, filter_file);
    assert(nitems == 1);
    nitems = fwrite(&neural_filter->is_activated, sizeof(bool), 1, filter_file);
    assert(nitems == 1);
    nitems = fwrite(&neural_filter->is_trained, sizeof(bool), 1, filter_file);
    assert(nitems == 1);

    nitems = fwrite(&neural_filter->num_global_query, sizeof(ID_TYPE), 1, filter_file);
    assert(nitems == 1);
    nitems = fwrite(&neural_filter->num_local_query, sizeof(ID_TYPE), 1, filter_file);
    assert(nitems == 1);
    if (neural_filter->num_local_query > 0) {
        assert(neural_filter->local_nn_distances != nullptr);
    }

    nitems = fwrite(&neural_filter->series_length, sizeof(unsigned int), 1, filter_file);
    assert(nitems == 1);
    nitems = fwrite(&neural_filter->dim_instance, sizeof(int), 1, filter_file);
    assert(nitems == 1);

    nitems = fwrite(neural_filter->global_node_distances, sizeof(VALUE_TYPE), neural_filter->num_global_query,
                    filter_file);
    assert(nitems == neural_filter->num_global_query);

    nitems = fwrite(neural_filter->global_bsf_distances, sizeof(VALUE_TYPE), neural_filter->num_global_query,
                    filter_file);
    assert(nitems == neural_filter->num_global_query);

    nitems = fwrite(neural_filter->global_nn_distances, sizeof(VALUE_TYPE), neural_filter->num_global_query,
                    filter_file);
    assert(nitems == neural_filter->num_global_query);

    if (neural_filter->is_trained) {
        nitems = fwrite(neural_filter->global_pred_distances, sizeof(VALUE_TYPE), neural_filter->num_global_query,
                        filter_file);
        assert(nitems == neural_filter->num_global_query);
    }

    if (neural_filter->num_local_query > 0) {
        nitems = fwrite(neural_filter->local_nn_distances, sizeof(VALUE_TYPE), neural_filter->num_local_query,
                        filter_file);
        assert(nitems == neural_filter->num_local_query);
    }

    nitems = fwrite(&neural_filter->is_distance_squared, sizeof(bool), 1, filter_file);
    assert(nitems == 1);

    nitems = fwrite(&neural_filter->node_size, sizeof(ID_TYPE), 1, filter_file);
    assert(nitems == 1);
    nitems = fwrite(&neural_filter->sax_prune_ratio, sizeof(VALUE_TYPE), 1, filter_file);
    assert(nitems == 1);
    nitems = fwrite(&neural_filter->estimated_gain, sizeof(VALUE_TYPE), 1, filter_file);
    assert(nitems == 1);

    assert(fclose(filter_file) == 0);

    if (neural_filter->is_trained) {
        char *filter_model_filepath = concat(2, filter_dump_prefix, config->dump_model_filename_postfix);
        torch::save(std::shared_ptr<MLP1>((MLP1 *) neural_filter->net), filter_model_filepath);
    }
}


std::vector<std::shared_ptr<MLP1>> loaded_nets;

void loadFilter(const Config *config, char *filter_load_prefix, NeuralFilter *filter) {
    char *filter_data_filepath = concat(2, filter_load_prefix, config->dump_filename_postfix);
    FILE *filter_file;
    filter_file = fopen(filter_data_filepath, "rb");
    if (filter_file == nullptr) {
        clog_error(CLOG(CLOGGER_ID), "filter %d - cannot find data %s", filter->id, filter_data_filepath);
        exit(-1);
    }

    int int_cache;
    unsigned int uint_cache;
    ID_TYPE id_cache;

    size_t nitems = fread(&int_cache, sizeof(int), 1, filter_file);
    assert(nitems == 1 && int_cache == filter->id);
    nitems = fread(&filter->is_activated, sizeof(bool), 1, filter_file);
    assert(nitems == 1);
    nitems = fread(&filter->is_trained, sizeof(bool), 1, filter_file);
    assert(nitems == 1);

    nitems = fread(&uint_cache, sizeof(ID_TYPE), 1, filter_file);
    assert(nitems == 1 && uint_cache == filter->num_global_query);
    nitems = fread(&filter->num_local_query, sizeof(ID_TYPE), 1, filter_file);
    assert(nitems == 1);
    nitems = fread(&uint_cache, sizeof(unsigned int), 1, filter_file);
    assert(nitems == 1 && uint_cache == filter->series_length);
    nitems = fread(&uint_cache, sizeof(int), 1, filter_file);
    assert(nitems == 1 && uint_cache == filter->dim_instance);

    filter->global_node_distances = static_cast<VALUE_TYPE *>(malloc(sizeof(VALUE_TYPE) * filter->num_global_query));
    filter->global_bsf_distances = static_cast<VALUE_TYPE *>(malloc(sizeof(VALUE_TYPE) * filter->num_global_query));
    filter->global_nn_distances = static_cast<VALUE_TYPE *>(malloc(sizeof(VALUE_TYPE) * filter->num_global_query));

    nitems = fread(filter->global_node_distances, sizeof(VALUE_TYPE), filter->num_global_query, filter_file);
    assert(nitems == filter->num_global_query);
    nitems = fread(filter->global_bsf_distances, sizeof(VALUE_TYPE), filter->num_global_query, filter_file);
    assert(nitems == filter->num_global_query);
    nitems = fread(filter->global_nn_distances, sizeof(VALUE_TYPE), filter->num_global_query, filter_file);
    assert(nitems == filter->num_global_query);

    if (filter->is_trained) {
        filter->global_pred_distances = static_cast<VALUE_TYPE *>(malloc(
                sizeof(VALUE_TYPE) * filter->num_global_query));
        nitems = fread(filter->global_pred_distances, sizeof(VALUE_TYPE), filter->num_global_query, filter_file);
        assert(nitems == filter->num_global_query);
    } else {
        filter->global_pred_distances = nullptr;
    }

    if (filter->num_local_query > 0) {
        filter->local_nn_distances = static_cast<VALUE_TYPE *>(malloc(sizeof(VALUE_TYPE) * filter->num_local_query));
        nitems = fread(filter->local_nn_distances, sizeof(VALUE_TYPE), filter->num_local_query, filter_file);
        assert(nitems == filter->num_local_query);
    }

    nitems = fread(&filter->is_distance_squared, sizeof(bool), 1, filter_file);
    assert(nitems == 1);
    if (filter->is_trained) {
        assert(!filter->is_distance_squared);
    }

    nitems = fread(&id_cache, sizeof(ID_TYPE), 1, filter_file);
    assert(nitems == 1 && id_cache == filter->node_size);
    nitems = fread(&filter->sax_prune_ratio, sizeof(VALUE_TYPE), 1, filter_file);
    assert(nitems == 1);
    nitems = fread(&filter->estimated_gain, sizeof(VALUE_TYPE), 1, filter_file);
    assert(nitems == 1);

    fclose(filter_file);

    if (filter->is_trained) {
        char *filter_model_filepath = concat(2, filter_load_prefix, config->dump_model_filename_postfix);

        std::shared_ptr<MLP1> net2load = std::make_shared<MLP1>(filter->dim_instance, config->dim_latent,
                                                                config->dropout_p, config->leaky_relu_negative_slope);
        torch::load(net2load, filter_model_filepath);
        net2load->to(*device_global);

        filter->net = (void *) net2load.get();
        loaded_nets.emplace_back(net2load); // avoid being freed

        if (config->is_filter_conformal) {
            ID_TYPE num_global_train = filter->num_global_query * config->filter_train_val_split;
            ID_TYPE num_global_valid = filter->num_global_query - num_global_train;
            ID_TYPE num_conformal_examples = num_global_valid;

            auto *conformal_adjuster = new ConformalRegressor(config->filter_conformal_core_type);

            trainConformalAdjustor(conformal_adjuster, num_conformal_examples,
                                   filter->global_pred_distances + num_global_train,
                                   filter->global_nn_distances + num_global_train, filter->id);

            filter->conformal_adjustor = (void *) conformal_adjuster;
        }
    }
}

void freeFilter(NeuralFilter *neural_filter) {
    if (neural_filter->net != nullptr) {
        free(neural_filter->net);
        neural_filter->net = nullptr;
    }

    if (neural_filter->local_queries != nullptr) {
        free(neural_filter->local_queries);
        neural_filter->local_queries = nullptr;
    }

    if (neural_filter->global_bsf_distances != nullptr) {
        free(neural_filter->global_bsf_distances);
        neural_filter->global_bsf_distances = nullptr;
    }
    if (neural_filter->global_node_distances != nullptr) {
        free(neural_filter->global_node_distances);
        neural_filter->global_node_distances = nullptr;
    }

    if (neural_filter->global_nn_distances != nullptr) {
        free(neural_filter->global_nn_distances);
        neural_filter->global_nn_distances = nullptr;
    }

    if (neural_filter->global_pred_distances != nullptr) {
        free(neural_filter->global_pred_distances);
        neural_filter->global_pred_distances = nullptr;
    }

    if (neural_filter->local_nn_distances != nullptr) {
        free(neural_filter->local_nn_distances);
        neural_filter->local_nn_distances = nullptr;
    }
}


void freeGlobalVariables() {
    if (streams_global) {
        free(streams_global);
        streams_global = nullptr;
    }

    if (device_global) {
        free(device_global);
        device_global = nullptr;
    }
}
