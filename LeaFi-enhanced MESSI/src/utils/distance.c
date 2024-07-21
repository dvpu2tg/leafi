/******************************************************************************
 * Copyright (c) 2024, the LeaFi author(s).
 ******************************************************************************/

#include "distance.h"


VALUE_TYPE l2Square(unsigned int length, VALUE_TYPE const *left, VALUE_TYPE const *right) {
    VALUE_TYPE sum = 0;

    for (unsigned int i = 0; i < length; ++i) {
        sum += (left[i] - right[i]) * (left[i] - right[i]);
    }

    return sum;
}


VALUE_TYPE l2SquareSIMD(unsigned int length, VALUE_TYPE const *left, VALUE_TYPE const *right, VALUE_TYPE *cache) {
    __m256 m256_square_cumulated = _mm256_setzero_ps(), m256_diff, m256_sum, m256_left, m256_right;

    for (unsigned int i = 0; i < length; i += 8) {
        m256_left = _mm256_load_ps(left + i);
        m256_right = _mm256_load_ps(right + i);
        m256_diff = _mm256_sub_ps(m256_left, m256_right);
        m256_square_cumulated = _mm256_fmadd_ps(m256_diff, m256_diff, m256_square_cumulated);
    }

    m256_sum = _mm256_hadd_ps(m256_square_cumulated, m256_square_cumulated);
    _mm256_store_ps(cache, _mm256_hadd_ps(m256_sum, m256_sum));

    return cache[0] + cache[4];
}


VALUE_TYPE l2SquareEarly(unsigned int length, VALUE_TYPE const *left, VALUE_TYPE const *right, VALUE_TYPE threshold) {
    VALUE_TYPE sum = 0;

    for (unsigned int i = 0; i < length; ++i) {
        sum += ((left[i] - right[i]) * (left[i] - right[i]));

        if (VALUE_G(sum, threshold)) {
            return sum;
        }
    }

    return sum;
}


VALUE_TYPE l2SquareEarlySIMD(unsigned int length, VALUE_TYPE const *left, VALUE_TYPE const *right, VALUE_TYPE threshold, VALUE_TYPE *cache) {
    VALUE_TYPE sum = 0;

    __m256 m256_square, m256_diff, m256_sum, m256_left, m256_right;
    for (unsigned int i = 0; i < length; i += 8) {
        m256_left = _mm256_load_ps(left + i);
        m256_right = _mm256_load_ps(right + i);
        m256_diff = _mm256_sub_ps(m256_left, m256_right);
        m256_square = _mm256_mul_ps(m256_diff, m256_diff);
        m256_sum = _mm256_hadd_ps(m256_square, m256_square);
        _mm256_store_ps(cache, _mm256_hadd_ps(m256_sum, m256_sum));

        sum += (cache[0] + cache[4]);

        if (VALUE_G(sum, threshold)) {
            return sum;
        }
    }

    return sum;
}
