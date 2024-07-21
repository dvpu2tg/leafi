/******************************************************************************
 * Copyright (c) 2024, the LeaFi author(s).
 ******************************************************************************/

#include "lr_scheduler.h"

#include <cmath>


ReduceLROnPlateau::ReduceLROnPlateau(torch::optim::Optimizer &optimizer,
                                     ID_TYPE initial_cooldown,
                                     METRICS_MODE mode,
                                     double factor,
                                     ID_TYPE patience,
                                     VALUE_TYPE threshold,
                                     THRESHOLD_MODE threshold_mode,
                                     ID_TYPE cooldown,
                                     double min_lr,
                                     double eps) :
        torch::optim::LRScheduler(optimizer),
        mode_(mode),
        factor_(factor),
        patience_(patience),
        threshold_(threshold),
        threshold_mode_(threshold_mode),
        cooldown_(cooldown),
        initial_cooldown_(initial_cooldown),
        eps_(eps) {
    for (ID_TYPE i = 0; i < optimizer.param_groups().size(); ++i) {
        min_lrs.push_back(min_lr);
    }

    // _init_is_better
    if (mode == MIN) {
        mode_worst_ = VALUE_MAX;
    } else {
        mode_worst_ = VALUE_MIN;
    }

    // _reset
    best_ = mode_worst_;
    cooldown_counter_ = 0;
    num_bad_epochs_ = 0;
}


LR_RETURN_CODE ReduceLROnPlateau::check_and_schedule(VALUE_TYPE metrics, ID_TYPE epoch) {
    LR_RETURN_CODE code = SAME;

    if (epoch >= initial_cooldown_) {
        VALUE_TYPE current = metrics;

        if (epoch < 0) {
            epoch = last_epoch_ + 1;
        }

        if (is_better(current, best_)) {
            best_ = current;
            num_bad_epochs_ = 0;
        } else {
            num_bad_epochs_ += 1;
        }

        if (in_cooldown()) {
            cooldown_counter_ -= 1;
            num_bad_epochs_ = 0;
        }

        if (num_bad_epochs_ > patience_) {
#ifdef DEBUG
            double old_lr = get_current_lrs()[0];
#endif
            if (VALUE_LEQ(get_current_lrs()[0], min_lrs[0])) {
                code = EARLY_STOP;
            } else {
                step();

                cooldown_counter_ = cooldown_;
                num_bad_epochs_ = 0;

                code = REDUCED;
            }
#ifdef DEBUG
            double new_lr = get_current_lrs()[0];
            spdlog::debug("filter lr = {:f} -> {:f}", old_lr, new_lr);
#endif
        }
    }

    if (epoch > 0) {
        last_epoch_ = epoch;
    }

    return code;
}


std::vector<double> ReduceLROnPlateau::get_lrs() {
    std::vector<double> lrs = get_current_lrs();

    // _reduce_lr
    for (ID_TYPE i = 0; i < lrs.size(); ++i) {
        double new_lr = fmax(lrs[i] * factor_, min_lrs[i]);

        if (lrs[i] - new_lr > eps_) {
            lrs[i] = new_lr;
        }
    }

    return lrs;
}


bool ReduceLROnPlateau::is_better(VALUE_TYPE a, VALUE_TYPE best) const {
    if (mode_ == MIN && threshold_mode_ == RELATIVE) {
        VALUE_TYPE rel_epsilon = 1 - threshold_;
        return a < best * rel_epsilon;
    } else if (mode_ == MIN && threshold_mode_ == ABSOLUTE) {
        return a < best - threshold_;
    } else if (mode_ == MAX && threshold_mode_ == RELATIVE) {
        VALUE_TYPE rel_epsilon = 1 + threshold_;
        return a > best * rel_epsilon;
    } else if (mode_ == MAX && threshold_mode_ == ABSOLUTE) {
        return a > best + threshold_;
    }

    // should not be reached
    return false;
}


bool ReduceLROnPlateau::in_cooldown() const {
    return cooldown_counter_ > 0;
}
