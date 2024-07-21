/******************************************************************************
 * Copyright (c) 2024, the LeaFi author(s).
 ******************************************************************************/

#include "series_dataset.h"


torch::Tensor loadSeries(VALUE_TYPE *series, int num_instances, int dim_instances, torch::Device device) {
    return torch::from_blob(series, {num_instances, dim_instances},
                            torch::TensorOptions().dtype(torch::kFloat32)).to(device);
}


torch::Tensor loadTargets(VALUE_TYPE *targets, int num_instances, torch::Device device) {
    return torch::from_blob(targets, num_instances,
                            torch::TensorOptions().dtype(torch::kFloat32)).to(device);
}


SeriesDataset::SeriesDataset(VALUE_TYPE *series, VALUE_TYPE *targets, int num_instances, int dim_instances, torch::Device device)
        : series_(loadSeries(series, num_instances, dim_instances, device)),
          targets_(loadTargets(targets, num_instances, device)) {}


torch::data::Example<> SeriesDataset::get(size_t index) {
    return {series_[index], targets_[index]};
}


torch::optional<size_t> SeriesDataset::size() const {
    return series_.size(0);
}
