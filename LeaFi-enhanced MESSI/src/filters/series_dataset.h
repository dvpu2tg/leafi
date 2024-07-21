/******************************************************************************
 * Copyright (c) 2024, the LeaFi author(s).
 ******************************************************************************/

#ifndef ISAX_SERIES_DATASET_H
#define ISAX_SERIES_DATASET_H

#pragma once

#include <torch/data/datasets/base.h>
#include <torch/data/example.h>
#include <torch/types.h>
//#include <torch/csrc/Export.h>

#include <cstddef>
#include <string>
#include <vector>

#include "globals.h"


class TORCH_API SeriesDataset : public torch::data::datasets::Dataset<SeriesDataset> {
public:
    explicit SeriesDataset(VALUE_TYPE *series, VALUE_TYPE *targets, int num_instances, int dim_instances, torch::Device device);

    torch::data::Example<> get(size_t index) override;

    torch::optional<size_t> size() const override;

//private:
    torch::Tensor series_, targets_;
};

#endif //ISAX_SERIES_DATASET_H
