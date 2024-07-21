/******************************************************************************
 * Copyright (c) 2024, the LeaFi author(s).
 ******************************************************************************/

#ifndef ISAX_PAA_H
#define ISAX_PAA_H

#include <stdlib.h>
#include <pthread.h>

#include "clog.h"
#include "globals.h"


void piecewiseAggregate(VALUE_TYPE const *values, ID_TYPE size, unsigned int series_length, VALUE_TYPE *paas_out,
                        unsigned int summarization_length);

VALUE_TYPE *piecewiseAggregateMT(VALUE_TYPE const *values, ID_TYPE size, unsigned int series_length, unsigned int summarization_length,
                                 unsigned int num_threads);

#endif //ISAX_PAA_H
