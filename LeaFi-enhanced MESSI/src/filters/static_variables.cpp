/******************************************************************************
 * Copyright (c) 2024, the LeaFi author(s).
 ******************************************************************************/

#include "static_variables.h"


std::vector <at::cuda::CUDAStream> *streams_global = nullptr;

torch::Device *device_global = nullptr;
