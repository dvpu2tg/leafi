/******************************************************************************
 * Copyright (c) 2024, the LeaFi author(s).
 ******************************************************************************/

#ifndef ISAX_DISTANCE_H
#define ISAX_DISTANCE_H

#include <stdlib.h>
#include <immintrin.h>

#include "globals.h"

VALUE_TYPE l2Square(unsigned int length, VALUE_TYPE const *left, VALUE_TYPE const *right);

VALUE_TYPE l2SquareSIMD(unsigned int length, VALUE_TYPE const *left, VALUE_TYPE const *right, VALUE_TYPE *cache);

VALUE_TYPE l2SquareEarly(unsigned int length, VALUE_TYPE const *left, VALUE_TYPE const *right, VALUE_TYPE threshold);

VALUE_TYPE l2SquareEarlySIMD(unsigned int length, VALUE_TYPE const *left, VALUE_TYPE const *right, VALUE_TYPE threshold, VALUE_TYPE *cache);

#endif //ISAX_DISTANCE_H
