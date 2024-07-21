/******************************************************************************
 * Copyright (c) 2024, the LeaFi author(s).
 ******************************************************************************/

#ifndef ISAX_STATIC_VARIABLES_H
#define ISAX_STATIC_VARIABLES_H

#include <vector>
#include <numeric>
#include <iostream>

#include <torch/torch.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>


extern std::vector<at::cuda::CUDAStream> *streams_global;
//extern std::vector<torch::Tensor> *inputs_global;

//extern torch::Tensor *input_global_cpu;
//extern torch::Tensor *input_global;

extern torch::Device *device_global;

#endif //ISAX_STATIC_VARIABLES_H
