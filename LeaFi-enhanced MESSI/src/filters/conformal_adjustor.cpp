/******************************************************************************
 * Copyright (c) 2024, the LeaFi author(s).
 ******************************************************************************/

#include "conformal_adjustor.h"

#include <algorithm>
#include <cstring>
#include <cassert>
#include <numeric>

#include "clog.h"

template<class T>
static std::string array2str(T *values, ID_TYPE length) {
    return std::accumulate(values + 1,
                           values + length,
                           std::to_string(values[0]),
                           [](const std::string &a, T b) {
                               return a + " " + std::to_string(b);
                           });
}


ConformalRegressor::ConformalRegressor(char *core_type_str,
                                       VALUE_TYPE confidence) :
        gsl_accel_(nullptr),
        gsl_spline_(nullptr) {
    if (strcmp(core_type_str, "discrete") == 0) {
        core_ = DISCRETE;
    } else if (strcmp(core_type_str, "spline") == 0) {
        core_ = SPLINE;
    } else {
        clog_error(CLOG(CLOGGER_ID),
                   "conformal core %s is not recognized; roll back to the default: discrete",
                   core_type_str);
        core_ = DISCRETE;
    }

    confidence_level_ = confidence;
}

ConformalRegressor::~ConformalRegressor() {
    if (gsl_accel_ != nullptr) {
        free(gsl_accel_);
    }
    if (gsl_spline_ != nullptr) {
        free(gsl_spline_);
    }
}

int ConformalRegressor::fit(std::vector<ERROR_TYPE> &residuals) {
    alphas_.assign(residuals.begin(), residuals.end());
    for (auto &residual: alphas_) { residual = residual < 0 ? -residual : residual; }

    std::sort(alphas_.begin(), alphas_.end()); //non-decreasing

    if (core_ == DISCRETE) {
        is_fitted_ = true;
        is_trial_ = false;

        abs_error_i_ = static_cast<ID_TYPE>(static_cast<VALUE_TYPE>(alphas_.size()) * confidence_level_);
        alpha_ = alphas_[abs_error_i_];
    } else { // core_ == SPLINE
        is_fitted_ = false;
    }

    return 0;
}

int ConformalRegressor::fit_spline(char *spline_core, std::vector<ERROR_TYPE> &recalls) {
#ifdef DEBUG
    assert(recalls.size() == alphas_.size());
#endif
    gsl_accel_ = gsl_interp_accel_alloc();

    if (strcmp(spline_core, "steffen") == 0) {
        gsl_spline_ = gsl_spline_alloc(gsl_interp_steffen, recalls.size());
    } else {
        clog_error(CLOG(CLOGGER_ID),
                   "conformal spline core %s is not recognized; roll back to the default: steffen",
                   spline_core);

        gsl_spline_ = gsl_spline_alloc(gsl_interp_steffen, recalls.size());
    }

    gsl_spline_init(gsl_spline_, recalls.data(), alphas_.data(), recalls.size());

    is_fitted_ = true;
    is_trial_ = false;

    return 0;
}

VALUE_TYPE ConformalRegressor::predict(VALUE_TYPE y_hat,
                                       VALUE_TYPE confidence_level,
                                       VALUE_TYPE y_max,
                                       VALUE_TYPE y_min) {
    if (is_fitted_) {
        if (confidence_level_ <= 1 && !VALUE_EQ(confidence_level_, confidence_level)) {
            assert(confidence_level >= 0 && confidence_level <= 1);

            abs_error_i_ = static_cast<ID_TYPE>(static_cast<VALUE_TYPE>(alphas_.size()) * confidence_level);
            alpha_ = alphas_[abs_error_i_];

            confidence_level_ = confidence_level;
        }

        if (y_hat < alpha_) {
            return 0;
        } else {
            return y_hat - alpha_;
        }
    } else if (is_trial_) {
        if (y_hat < alpha_) {
            return 0;
        } else {
            return y_hat - alpha_;
        }
    } else {
        return y_hat;
    }
}

VALUE_TYPE ConformalPredictor::get_alpha() const {
    if (is_fitted_) {
        return alpha_;
    } else if (is_trial_) {
        return alpha_;
    } else {
        return VALUE_MAX;
    }
}

int ConformalPredictor::set_alpha(VALUE_TYPE alpha, bool is_trial, bool to_intervene) {
    if (to_intervene) {
        alpha_ = alpha;
    } else {
        if (is_trial) {
            if (is_fitted_) {
                clog_error(CLOG(CLOGGER_ID),
                           "conformal model is already fitted; cannot run trial");
                return -1;
            } else {
                alpha_ = alpha;

                is_trial_ = true;
            }
        } else if (is_fitted_) {
            clog_error(CLOG(CLOGGER_ID),
                       "conformal model is already fitted; cannot directly adjust alpha");
            return -1;
        } else {
            alpha_ = alpha;
        }
    }

    return 0;
}

VALUE_TYPE ConformalPredictor::get_alpha_by_pos(ID_TYPE pos) const {
    if (pos >= 0 && pos < alphas_.size()) {
        return alphas_[pos];
    }

    return VALUE_MAX;
}

int ConformalPredictor::set_alpha_by_pos(ID_TYPE pos) {
    if (pos >= 0 && pos < alphas_.size()) {
        alpha_ = alphas_[pos];

        is_fitted_ = true;
        confidence_level_ = EXT_DISCRETE;

        return 0;
    }

    return -1;
}

int ConformalRegressor::set_alpha_by_recall(VALUE_TYPE recall) {
    assert(gsl_accel_ != nullptr && gsl_spline_ != nullptr);
    alpha_ = gsl_spline_eval(gsl_spline_, recall, gsl_accel_);

    is_fitted_ = true;
    confidence_level_ = EXT_SPLINE;

    return 0;
}
