/******************************************************************************
 * Copyright (c) 2024, the LeaFi author(s).
 ******************************************************************************/

#include "sax.h"


typedef struct SAXCache {
    SAXSymbol *saxs;
    VALUE_TYPE const *summarizations;
    VALUE_TYPE const *breakpoints;

    ID_TYPE size;
    unsigned int sax_length;

    ID_TYPE *shared_processed_counter;
    unsigned int block_size;
} SAXCache;


void *summarizations2SAX16Thread(void *cache) {
    SAXCache *saxCache = (SAXCache *) cache;
    ID_TYPE *shared_processed_counter = saxCache->shared_processed_counter;
    unsigned int block_size = saxCache->block_size;
    ID_TYPE size = saxCache->size;
    VALUE_TYPE const *summarizations = saxCache->summarizations;
    VALUE_TYPE const *breakpoints = saxCache->breakpoints;
    SAXSymbol *saxs = saxCache->saxs;

    unsigned int sax_length = saxCache->sax_length;
//    unsigned int sax_cardinality = saxCache->sax_cardinality;

    ID_TYPE start_id, stop_id;
    ID_TYPE current_sax_id, current_summarization_id, current_segment;

    while ((start_id = __sync_fetch_and_add(shared_processed_counter, block_size)) < size) {
        stop_id = start_id + block_size;
        if (stop_id > size) {
            stop_id = size;
        }

        for (current_sax_id = SAX_SIMD_ALIGNED_LENGTH * start_id, current_summarization_id = sax_length * start_id;
             current_sax_id < SAX_SIMD_ALIGNED_LENGTH * stop_id;
             current_sax_id += SAX_SIMD_ALIGNED_LENGTH, current_summarization_id += sax_length) {
            for (current_segment = 0; current_segment < sax_length; ++current_segment) {
                saxs[current_sax_id + current_segment] = bSearchFloorValue(
                        summarizations[current_summarization_id + current_segment],
                        breakpoints + OFFSETS_BY_SEGMENTS[current_segment] + OFFSETS_BY_CARDINALITY[7],
                        0, 256);
            }
        }
    }

    return NULL;
}


void summarizations2SAX16(SAXSymbol *saxs, VALUE_TYPE const *summarizations, VALUE_TYPE const *breakpoints, ID_TYPE size,
                          unsigned int sax_length, unsigned int sax_cardinality, unsigned int num_threads) {
    pthread_t threads[num_threads];
    SAXCache saxCache[num_threads];

    ID_TYPE shared_processed_counter = 0;
    unsigned int block_size = 1 + size / (num_threads << 1u);
    for (unsigned int i = 0; i < num_threads; ++i) {
        saxCache[i].saxs = saxs;
        saxCache[i].summarizations = summarizations;
        saxCache[i].breakpoints = breakpoints;

        saxCache[i].size = size;
        saxCache[i].sax_length = sax_length;

        saxCache[i].shared_processed_counter = &shared_processed_counter;
        saxCache[i].block_size = block_size;

        pthread_create(&threads[i], NULL, summarizations2SAX16Thread, (void *) &saxCache[i]);
    }

    for (unsigned int i = 0; i < num_threads; ++i) {
        pthread_join(threads[i], NULL);
    }
}


VALUE_TYPE l2SquareValue2SAXByMask(unsigned int sax_length, VALUE_TYPE const *summarizations, SAXSymbol const *sax,
                                   SAXMask const *masks, VALUE_TYPE const *breakpoints, VALUE_TYPE scale_factor) {
    VALUE_TYPE sum = 0;
    VALUE_TYPE const *current_breakpoints;

    for (unsigned int i = 0; i < sax_length; ++i) {
        current_breakpoints = breakpoints + OFFSETS_BY_SEGMENTS[i] + OFFSETS_BY_MASK[masks[i]] +
                              ((unsigned int) sax[i] >> SHIFTS_BY_MASK[masks[i]]);

        if (VALUE_L(summarizations[i], *current_breakpoints)) {
            sum += (*current_breakpoints - summarizations[i]) * (*current_breakpoints - summarizations[i]);
        } else if (VALUE_G(summarizations[i], *(current_breakpoints + 1))) {
            sum += (summarizations[i] - *(current_breakpoints + 1)) * (summarizations[i] - *(current_breakpoints + 1));
        }
    }

    return sum * scale_factor;
}


VALUE_TYPE l2SquareValue2SAX8(unsigned int sax_length, VALUE_TYPE const *summarizations, SAXSymbol const *sax,
                              VALUE_TYPE const *breakpoints, VALUE_TYPE scale_factor) {
    VALUE_TYPE sum = 0;
    VALUE_TYPE const *current_breakpoints;

    for (unsigned int i = 0; i < sax_length; ++i) {
        current_breakpoints = breakpoints + OFFSETS_BY_SEGMENTS[i] + OFFSETS_BY_CARDINALITY[7] + (unsigned int) sax[i];

        if (VALUE_L(summarizations[i], *current_breakpoints)) {
            sum += (*current_breakpoints - summarizations[i]) * (*current_breakpoints - summarizations[i]);
        } else if (VALUE_G(summarizations[i], *(current_breakpoints + 1))) {
            sum += (summarizations[i] - *(current_breakpoints + 1)) * (summarizations[i] - *(current_breakpoints + 1));
        }
    }

    return sum * scale_factor;
}


VALUE_TYPE l2SquareValue2SAXByMaskSIMD(unsigned int sax_length, VALUE_TYPE const *summarizations, SAXSymbol const *sax,
                                       SAXMask const *masks, VALUE_TYPE const *breakpoints, VALUE_TYPE scale_factor, VALUE_TYPE *cache) {
    __m256i const *m256i_masks = (__m256i const *) masks;
    __m256i m256i_sax_packed = _mm256_cvtepu8_epi16(_mm_load_si128((__m128i const *) sax));

    __m256 m256_summarizations = _mm256_loadu_ps(summarizations);
    __m256i m256i_sax = _mm256_cvtepu16_epi32(_mm256_extractf128_si256(m256i_sax_packed, 0));

    __m256i m256i_mask_indices = _mm256_load_si256(m256i_masks);
    __m256i m256i_sax_shifts = _mm256_i32gather_epi32((const int *) SHIFTS_BY_MASK, m256i_mask_indices, 4);
    __m256i m256i_cardinality_offsets = _mm256_i32gather_epi32((const int *) OFFSETS_BY_MASK, m256i_mask_indices, 4);

    __m256i m256i_sax_offsets = _mm256_add_epi32(M256I_BREAKPOINTS_OFFSETS_0_7, m256i_cardinality_offsets);
    __m256i m256i_sax_indices = _mm256_srlv_epi32(m256i_sax, m256i_sax_shifts);

    __m256i m256i_floor_indices = _mm256_add_epi32(m256i_sax_offsets, m256i_sax_indices);
    __m256 m256_floor_breakpoints = _mm256_i32gather_ps(breakpoints, m256i_floor_indices, 4);

    __m256i m256i_ceiling_indices = _mm256_add_epi32(m256i_floor_indices, M256I_1);
    __m256 m256_ceiling_breakpoints = _mm256_i32gather_ps(breakpoints, m256i_ceiling_indices, 4);

    __m256 m256_floor_diff = _mm256_sub_ps(m256_floor_breakpoints, m256_summarizations);
    __m256 m256_floor_indicator = _mm256_cmp_ps(m256_summarizations, m256_floor_breakpoints, _CMP_LT_OS);
    __m256 m256_floor_l1 = _mm256_and_ps(m256_floor_diff, m256_floor_indicator);

    __m256 m256_ceiling_diff = _mm256_sub_ps(m256_summarizations, m256_ceiling_breakpoints);
    __m256 m256_ceiling_indicator = _mm256_cmp_ps(m256_summarizations, m256_ceiling_breakpoints, _CMP_GT_OS);
    __m256 m256_ceiling_l1 = _mm256_and_ps(m256_ceiling_diff, m256_ceiling_indicator);

    __m256 m256_l1 = _mm256_add_ps(m256_floor_l1, m256_ceiling_l1);
    __m256 m256_l2square = _mm256_mul_ps(m256_l1, m256_l1);

    if (sax_length == 16) {
        m256_summarizations = _mm256_loadu_ps(summarizations + 8);
        m256i_sax = _mm256_cvtepu16_epi32(_mm256_extractf128_si256(m256i_sax_packed, 1));

        m256i_mask_indices = _mm256_load_si256(m256i_masks + 1);
        m256i_sax_shifts = _mm256_i32gather_epi32((const int *) SHIFTS_BY_MASK, m256i_mask_indices, 4);
        m256i_cardinality_offsets = _mm256_i32gather_epi32((const int *) OFFSETS_BY_MASK, m256i_mask_indices, 4);

        m256i_sax_offsets = _mm256_add_epi32(M256I_BREAKPOINTS_OFFSETS_8_15, m256i_cardinality_offsets);
        m256i_sax_indices = _mm256_srlv_epi32(m256i_sax, m256i_sax_shifts);

        m256i_floor_indices = _mm256_add_epi32(m256i_sax_offsets, m256i_sax_indices);
        m256_floor_breakpoints = _mm256_i32gather_ps(breakpoints, m256i_floor_indices, 4);

        m256i_ceiling_indices = _mm256_add_epi32(m256i_floor_indices, M256I_1);
        m256_ceiling_breakpoints = _mm256_i32gather_ps(breakpoints, m256i_ceiling_indices, 4);

        m256_floor_diff = _mm256_sub_ps(m256_floor_breakpoints, m256_summarizations);
        m256_floor_indicator = _mm256_cmp_ps(m256_summarizations, m256_floor_breakpoints, _CMP_LT_OS);
        m256_floor_l1 = _mm256_and_ps(m256_floor_diff, m256_floor_indicator);

        m256_ceiling_diff = _mm256_sub_ps(m256_summarizations, m256_ceiling_breakpoints);
        m256_ceiling_indicator = _mm256_cmp_ps(m256_summarizations, m256_ceiling_breakpoints, _CMP_GT_OS);
        m256_ceiling_l1 = _mm256_and_ps(m256_ceiling_diff, m256_ceiling_indicator);

        m256_l1 = _mm256_add_ps(m256_floor_l1, m256_ceiling_l1);
        m256_l2square = _mm256_fmadd_ps(m256_l1, m256_l1, m256_l2square);
    }

    m256_l2square = _mm256_hadd_ps(m256_l2square, m256_l2square);
    _mm256_store_ps(cache, _mm256_hadd_ps(m256_l2square, m256_l2square));

    VALUE_TYPE distance = cache[0] + cache[4];

    if (sax_length == 8 || sax_length == 16) {
        return distance * scale_factor;
    }

    VALUE_TYPE const *current_breakpoints;

    for (unsigned int i = 8; i < sax_length; ++i) {
        current_breakpoints = breakpoints + OFFSETS_BY_SEGMENTS[i] + OFFSETS_BY_MASK[masks[i]] +
                              ((unsigned int) sax[i] >> SHIFTS_BY_MASK[masks[i]]);

        if (VALUE_L(summarizations[i], *current_breakpoints)) {
            distance += (*current_breakpoints - summarizations[i]) * (*current_breakpoints - summarizations[i]);
        } else if (VALUE_G(summarizations[i], *(current_breakpoints + 1))) {
            distance +=
                    (summarizations[i] - *(current_breakpoints + 1)) * (summarizations[i] - *(current_breakpoints + 1));
        }
    }

    return distance * scale_factor;
}


VALUE_TYPE l2SquareValue2EnvelopSIMD(unsigned int sax_length, VALUE_TYPE const *summarizations, VALUE_TYPE const *upper_envelops,
                                     VALUE_TYPE const *lower_envelops, VALUE_TYPE scale_factor, VALUE_TYPE *cache) {
    __m256 m256_summarizations = _mm256_loadu_ps(summarizations);
    __m256 m256_floor_breakpoints = _mm256_loadu_ps(lower_envelops);
    __m256 m256_ceiling_breakpoints = _mm256_loadu_ps(upper_envelops);

    __m256 m256_floor_diff = _mm256_sub_ps(m256_floor_breakpoints, m256_summarizations);
    __m256 m256_floor_indicator = _mm256_cmp_ps(m256_summarizations, m256_floor_breakpoints, _CMP_LT_OS);
    __m256 m256_floor_l1 = _mm256_and_ps(m256_floor_diff, m256_floor_indicator);

    __m256 m256_ceiling_diff = _mm256_sub_ps(m256_summarizations, m256_ceiling_breakpoints);
    __m256 m256_ceiling_indicator = _mm256_cmp_ps(m256_summarizations, m256_ceiling_breakpoints, _CMP_GT_OS);
    __m256 m256_ceiling_l1 = _mm256_and_ps(m256_ceiling_diff, m256_ceiling_indicator);

    __m256 m256_l1 = _mm256_add_ps(m256_floor_l1, m256_ceiling_l1);
    __m256 m256_l2square = _mm256_mul_ps(m256_l1, m256_l1);

    if (sax_length == 16) {
        m256_summarizations = _mm256_loadu_ps(summarizations + 8);
        m256_floor_breakpoints = _mm256_loadu_ps(lower_envelops + 8);
        m256_ceiling_breakpoints = _mm256_loadu_ps(upper_envelops + 8);

        m256_floor_diff = _mm256_sub_ps(m256_floor_breakpoints, m256_summarizations);
        m256_floor_indicator = _mm256_cmp_ps(m256_summarizations, m256_floor_breakpoints, _CMP_LT_OS);
        m256_floor_l1 = _mm256_and_ps(m256_floor_diff, m256_floor_indicator);

        m256_ceiling_diff = _mm256_sub_ps(m256_summarizations, m256_ceiling_breakpoints);
        m256_ceiling_indicator = _mm256_cmp_ps(m256_summarizations, m256_ceiling_breakpoints, _CMP_GT_OS);
        m256_ceiling_l1 = _mm256_and_ps(m256_ceiling_diff, m256_ceiling_indicator);

        m256_l1 = _mm256_add_ps(m256_floor_l1, m256_ceiling_l1);
        m256_l2square = _mm256_fmadd_ps(m256_l1, m256_l1, m256_l2square);
    }

    m256_l2square = _mm256_hadd_ps(m256_l2square, m256_l2square);
    _mm256_store_ps(cache, _mm256_hadd_ps(m256_l2square, m256_l2square));

    VALUE_TYPE distance = cache[0] + cache[4];

    if (sax_length == 8 || sax_length == 16) {
        return distance * scale_factor;
    }

    for (unsigned int i = 8; i < sax_length; ++i) {
        if (VALUE_L(summarizations[i], lower_envelops[i])) {
            distance += (lower_envelops[i] - summarizations[i]) * (lower_envelops[i] - summarizations[i]);
        } else if (VALUE_G(summarizations[i], upper_envelops[i])) {
            distance += (summarizations[i] - upper_envelops[i]) * (summarizations[i] - upper_envelops[i]);
        }
    }

    return distance * scale_factor;
}


VALUE_TYPE l2SquareSummarization2SAX8SIMD(unsigned int sax_length, VALUE_TYPE const *summarizations, SAXSymbol const *sax,
                                          VALUE_TYPE const *breakpoints, VALUE_TYPE scale_factor, VALUE_TYPE *cache) {
    __m256i m256i_sax_packed = _mm256_cvtepu8_epi16(_mm_load_si128((__m128i const *) sax));

    __m256 m256_summarizations = _mm256_loadu_ps(summarizations);
    __m256i m256i_sax = _mm256_cvtepu16_epi32(_mm256_extractf128_si256(m256i_sax_packed, 0));

    __m256i m256i_floor_indices = _mm256_add_epi32(m256i_sax, M256I_BREAKPOINTS8_OFFSETS_0_7);
    __m256 m256_floor_breakpoints = _mm256_i32gather_ps(breakpoints, m256i_floor_indices, 4);

    __m256i m256i_ceiling_indices = _mm256_add_epi32(m256i_floor_indices, M256I_1);
    __m256 m256_ceiling_breakpoints = _mm256_i32gather_ps(breakpoints, m256i_ceiling_indices, 4);

    __m256 m256_floor_diff = _mm256_sub_ps(m256_floor_breakpoints, m256_summarizations);
    __m256 m256_floor_indicator = _mm256_cmp_ps(m256_summarizations, m256_floor_breakpoints, _CMP_LT_OS);
    __m256 m256_floor_l1 = _mm256_and_ps(m256_floor_diff, m256_floor_indicator);

    __m256 m256_ceiling_diff = _mm256_sub_ps(m256_summarizations, m256_ceiling_breakpoints);
    __m256 m256_ceiling_indicator = _mm256_cmp_ps(m256_summarizations, m256_ceiling_breakpoints, _CMP_GT_OS);
    __m256 m256_ceiling_l1 = _mm256_and_ps(m256_ceiling_diff, m256_ceiling_indicator);

    __m256 m256_l1 = _mm256_add_ps(m256_floor_l1, m256_ceiling_l1);
    __m256 m256_l2square = _mm256_mul_ps(m256_l1, m256_l1);

    if (sax_length == 16) {
        m256_summarizations = _mm256_loadu_ps(summarizations + 8);
        m256i_sax = _mm256_cvtepu16_epi32(_mm256_extractf128_si256(m256i_sax_packed, 1));

        m256i_floor_indices = _mm256_add_epi32(m256i_sax, M256I_BREAKPOINTS8_OFFSETS_8_15);
        m256_floor_breakpoints = _mm256_i32gather_ps(breakpoints, m256i_floor_indices, 4);

        m256i_ceiling_indices = _mm256_add_epi32(m256i_floor_indices, M256I_1);
        m256_ceiling_breakpoints = _mm256_i32gather_ps(breakpoints, m256i_ceiling_indices, 4);

        m256_floor_diff = _mm256_sub_ps(m256_floor_breakpoints, m256_summarizations);
        m256_floor_indicator = _mm256_cmp_ps(m256_summarizations, m256_floor_breakpoints, _CMP_LT_OS);
        m256_floor_l1 = _mm256_and_ps(m256_floor_diff, m256_floor_indicator);

        m256_ceiling_diff = _mm256_sub_ps(m256_summarizations, m256_ceiling_breakpoints);
        m256_ceiling_indicator = _mm256_cmp_ps(m256_summarizations, m256_ceiling_breakpoints, _CMP_GT_OS);
        m256_ceiling_l1 = _mm256_and_ps(m256_ceiling_diff, m256_ceiling_indicator);

        m256_l1 = _mm256_add_ps(m256_floor_l1, m256_ceiling_l1);
        m256_l2square = _mm256_fmadd_ps(m256_l1, m256_l1, m256_l2square);
    }

    m256_l2square = _mm256_hadd_ps(m256_l2square, m256_l2square);
    _mm256_store_ps(cache, _mm256_hadd_ps(m256_l2square, m256_l2square));

    VALUE_TYPE distance = cache[0] + cache[4];

    if (sax_length == 8 || sax_length == 16) {
        return distance * scale_factor;
    }

    VALUE_TYPE const *current_breakpoints;

    for (unsigned int i = 8; i < sax_length; ++i) {
        current_breakpoints = breakpoints + OFFSETS_BY_SEGMENTS[i] + OFFSETS_BY_CARDINALITY[7] + (unsigned int) sax[i];

        if (VALUE_L(summarizations[i], *current_breakpoints)) {
            distance += (*current_breakpoints - summarizations[i]) * (*current_breakpoints - summarizations[i]);
        } else if (VALUE_G(summarizations[i], *(current_breakpoints + 1))) {
            distance +=
                    (summarizations[i] - *(current_breakpoints + 1)) * (summarizations[i] - *(current_breakpoints + 1));
        }
    }

    return distance * scale_factor;
}


char *sax2str(SAXSymbol const *sax, SAXMask const *masks, unsigned int sax_length, unsigned int sax_cardinality) {
    assert(sax_cardinality == 8);

    char *sax_str = malloc(sax_length * sax_cardinality);
    unsigned int sax_str_offset = 0, num_shifts, num_bits;

    for (unsigned int i = 0; i < sax_length; ++i) {
        num_shifts = sax_cardinality - 1;

        for (unsigned int j = 0; j < BITS_BY_MASK[masks[i]]; ++j, ++sax_str_offset, --num_shifts) {
            sax_str[sax_str_offset] = '0' + ((sax[i] & (1u << num_shifts)) >> num_shifts);
        }

        if (i == sax_length - 1) {
            sax_str[sax_str_offset] = '\0';
        } else {
            sax_str[sax_str_offset] = '_';
        }
        sax_str_offset += 1;
    }

    if (sax_str_offset < sax_length * sax_cardinality) {
        sax_str = (char *) realloc(sax_str, sax_str_offset);
    }

    return sax_str;
}


char *mask2str(SAXMask const *masks, char *mask_str, unsigned int sax_length, unsigned int sax_cardinality) {
    assert(sax_cardinality <= 8);

    mask_str[sax_length] = '\0';

    for (unsigned int i = 0; i < sax_length; ++i) {
        mask_str[i] = '0' + BITS_BY_MASK[masks[i]];
    }

    return mask_str;
}
