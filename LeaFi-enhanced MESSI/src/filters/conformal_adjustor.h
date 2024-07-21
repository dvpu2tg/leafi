/******************************************************************************
 * Copyright (c) 2024, the LeaFi author(s).
 ******************************************************************************/

#ifndef ISAX_CONFORMAL_ADJUSTOR_H
#define ISAX_CONFORMAL_ADJUSTOR_H

#include <vector>
#include <iostream>

#include "globals.h"

#include <gsl/gsl_spline.h>

#ifdef __cplusplus
extern "C" {
#endif

enum CONFORMAL_CORE {
    DISCRETE = 0,
    SPLINE = 1 // smoothened
};

enum CONFIDENCE_LEVEL_EXTERNAL {
    EXT_DISCRETE = 2,
    EXT_SPLINE = 3
};

class ConformalPredictor {
public:
    ConformalPredictor() : is_fitted_(false), is_trial_(false) {};
    ~ConformalPredictor() = default;

    VALUE_TYPE get_alpha() const;
    int set_alpha(VALUE_TYPE alpha, bool is_trial = false, bool to_intervene = false);

    VALUE_TYPE get_alpha_by_pos(ID_TYPE pos) const;
    int set_alpha_by_pos(ID_TYPE pos);

//    int dump(std::ofstream &node_fos) const;
//    int load(std::ifstream &node_ifs, void *ifs_buf);

    bool is_fitted() const {
        return is_fitted_;
    }

    bool is_fitted_;
    bool is_trial_;
    CONFORMAL_CORE core_;

    VALUE_TYPE confidence_level_;
    ID_TYPE abs_error_i_;
    VALUE_TYPE alpha_;

    std::vector<ERROR_TYPE> alphas_;
};

class ConformalRegressor : public ConformalPredictor {
public:
    explicit ConformalRegressor(char *core_type_str, VALUE_TYPE confidence = -1);
    ~ConformalRegressor();

    int set_alpha_by_recall(VALUE_TYPE recall);

    int fit(std::vector<ERROR_TYPE> &residuals);

    int fit_spline(char *spline_core, std::vector<ERROR_TYPE> &recalls);

    VALUE_TYPE predict(VALUE_TYPE y_hat,
                       VALUE_TYPE confidence_level = -1,
                       VALUE_TYPE y_max = VALUE_MAX,
                       VALUE_TYPE y_min = VALUE_MIN);

    gsl_interp_accel *gsl_accel_;
    gsl_spline *gsl_spline_;
};

#ifdef __cplusplus
}
#endif

#endif //ISAX_CONFORMAL_ADJUSTOR_H
