/******************************************************************************
 * Copyright (c) 2024, the LeaFi author(s).
 ******************************************************************************/

#ifndef ISAX_STATS_H
#define ISAX_STATS_H

#include <math.h>

#include "globals.h"
#include "clog.h"


static int calculateStats(VALUE_TYPE const *values, unsigned int num_values, VALUE_TYPE *mean, VALUE_TYPE *std) {
    VALUE_TYPE diff;

    for (unsigned int value_i = 0; value_i < num_values; ++value_i) {
        *mean += values[value_i];
    }

    *mean /= (VALUE_TYPE) num_values;

    for (unsigned int value_i = 0; value_i < num_values; ++value_i) {
        diff = values[value_i] - *mean;
        *std += diff * diff;
    }

    *std = (VALUE_TYPE) sqrt(*std / (VALUE_TYPE) num_values);

    if (*std <= VALUE_EPSILON) {
        return -1;
    }

    return 0;
}


static int znormalizeInPlace(VALUE_TYPE *series, unsigned int series_length) {
    VALUE_TYPE mean = 0, std = 0, diff;

    for (unsigned int value_i = 0; value_i < series_length; ++value_i) {
        mean += series[value_i];
    }

    mean /= (VALUE_TYPE) series_length;

    for (unsigned int value_i = 0; value_i < series_length; ++value_i) {
        diff = series[value_i] - mean;
        std += diff * diff;
    }

    std = (VALUE_TYPE) sqrt(std / (VALUE_TYPE) series_length);

    if (std <= VALUE_EPSILON) {
        return -1;
    }

    for (unsigned int value_i = 0; value_i < series_length; ++value_i) {
        series[value_i] = (series[value_i] - mean) / std;
    }

    return 0;
}

#endif //ISAX_STATS_H
