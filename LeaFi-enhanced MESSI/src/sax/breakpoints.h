/******************************************************************************
 * Copyright (c) 2024, the LeaFi author(s).
 ******************************************************************************/

#ifndef ISAX_BREAKPOINTS_H
#define ISAX_BREAKPOINTS_H

#include <float.h>
#include <stdlib.h>
#include <pthread.h>
#include <immintrin.h>

#include "globals.h"
#include "clog.h"
#include "config.h"


static unsigned int const OFFSETS_BY_CARDINALITY[9] = {
        0, 3, 8, 17, 34,
        67, 132, 261, 518
};


static unsigned int const OFFSETS_BY_SEGMENTS[17] = {
        0, 518, 518 * 2, 518 * 3, 518 * 4,
        518 * 5, 518 * 6, 518 * 7, 518 * 8, 518 * 9,
        518 * 10, 518 * 11, 518 * 12, 518 * 13, 518 * 14,
        518 * 15, 518 * 16
};


static unsigned int const OFFSETS_BY_MASK[129] = {
        0, 261, 132, 0, 67, 0, 0, 0, 34, 0, 0, 0, 0, 0, 0, 0, 17, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0
};


static unsigned int const LENGTHS_BY_MASK[129] = {
        0, 257, 129, 0, 65, 0, 0, 0, 33, 0, 0, 0, 0, 0, 0, 0, 17, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 3
};


void initializeM256IConstants();

VALUE_TYPE const *getNormalBreakpoints8(unsigned int num_segments);

VALUE_TYPE const *getAdhocBreakpoints8(VALUE_TYPE const *summarizations, size_t size, unsigned int num_segments,
                                       unsigned int num_threads);

VALUE_TYPE const *getSharedAdhocBreakpoints8(VALUE_TYPE const *summarizations, size_t size, unsigned int num_segments);

int dumpBreakpoints8(char const *breakpoints_dump_filepath, VALUE_TYPE const *breakpoints, unsigned int num_segments);
int loadBreakpoints8(char const *breakpoints_load_filepath, VALUE_TYPE const *breakpoints, unsigned int num_segments);

#endif //ISAX_BREAKPOINTS_H
