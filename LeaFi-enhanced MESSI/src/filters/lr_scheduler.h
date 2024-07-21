/******************************************************************************
 * Copyright (c) 2024, the LeaFi author(s).
 ******************************************************************************/

#ifndef ISAX_LR_SCHEDULER_H
#define ISAX_LR_SCHEDULER_H

#include <torch/optim/schedulers/lr_scheduler.h>

#include "globals.h"

#ifdef __cplusplus
extern "C" {
#endif

enum METRICS_MODE {
    MIN = 0,
    MAX = 1
};

enum THRESHOLD_MODE {
    RELATIVE = 0,
    ABSOLUTE = 1
};

enum LR_RETURN_CODE {
    SAME = 0,
    REDUCED = 1,
    EARLY_STOP = 2
};

class TORCH_API ReduceLROnPlateau : public torch::optim::LRScheduler {
public:
    explicit ReduceLROnPlateau(torch::optim::Optimizer &optimizer,
                               ID_TYPE initial_cooldown = 0,
                               METRICS_MODE mode = MIN,
                               double factor = 0.1,
                               ID_TYPE patience = 10,
                               VALUE_TYPE threshold = 1e-4,
                               THRESHOLD_MODE threshold_mode = RELATIVE,
                               ID_TYPE cooldown = 10,
                               double min_lr = 1e-7,
                               double eps = 1e-7);

    LR_RETURN_CODE check_and_schedule(VALUE_TYPE metrics, ID_TYPE epoch = -1);

private:
    std::vector<double> get_lrs() override;

    bool is_better(VALUE_TYPE a, VALUE_TYPE best) const;

    bool in_cooldown() const; // does not have internal effects

    ID_TYPE patience_;
    ID_TYPE num_bad_epochs_;

    ID_TYPE cooldown_;
    ID_TYPE initial_cooldown_;
    ID_TYPE cooldown_counter_;

    ID_TYPE last_epoch_;
    VALUE_TYPE best_, mode_worst_;

    METRICS_MODE mode_;
    THRESHOLD_MODE threshold_mode_;
    VALUE_TYPE threshold_;

    std::vector<double> min_lrs;

    double factor_;
    double eps_;
};

#ifdef __cplusplus
}
#endif

#endif //ISAX_LR_SCHEDULER_H
