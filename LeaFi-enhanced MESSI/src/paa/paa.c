/******************************************************************************
 * Copyright (c) 2024, the LeaFi author(s).
 ******************************************************************************/

#include "paa.h"


typedef struct PAACache {
    VALUE_TYPE const *values;
    VALUE_TYPE *paas;

    size_t size;
    unsigned int series_length;
    unsigned int paa_length;

    ID_TYPE *shared_processed_counter;
    unsigned int block_size;

    unsigned int *segment_lengths;
    VALUE_TYPE *segment_length_floats;
} PAACache;


void piecewiseAggregate(VALUE_TYPE const *values, ID_TYPE size, unsigned int series_length, VALUE_TYPE *paas_out,
                        unsigned int summarization_length) {
    unsigned int *segment_lengths = malloc(sizeof(unsigned int) * summarization_length);
    VALUE_TYPE *segment_length_floats = malloc(sizeof(VALUE_TYPE) * summarization_length);

    for (unsigned int i = 0; i < summarization_length; ++i) {
        segment_lengths[i] = series_length / summarization_length;
    }

    for (unsigned int i = 0; i < series_length % summarization_length; ++i) {
        segment_lengths[i] += 1;
    }

    for (unsigned int i = 0; i < summarization_length; ++i) {
        segment_length_floats[i] = (VALUE_TYPE) segment_lengths[i];
    }

#ifdef ISAX_PROFILING
    unsigned int sum = 0;

    for (unsigned int i = 0; i < summarization_length; ++i) {
        sum += segment_lengths[i];
    }

    if (sum != series_length) {
        clog_info(CLOG(CLOGGER_ID), "cannot assign %d to %d (%d)", series_length, summarization_length, sum);
    }
#endif

    unsigned int segment_id;

    unsigned int value_counter;
    float segment_sum;
    VALUE_TYPE const *pt_value;
    VALUE_TYPE *pt_paa;

    for (pt_value = values, pt_paa = paas_out, value_counter = 0, segment_sum = 0, segment_id = 0;
         pt_value < values + series_length * size;
         ++pt_value) {
        segment_sum += *pt_value;
        value_counter += 1;

        if (value_counter == segment_lengths[segment_id]) {
            *pt_paa = segment_sum / segment_length_floats[segment_id];
            pt_paa += 1;

            value_counter = 0;
            segment_sum = 0;
            segment_id = (segment_id + 1) % summarization_length;
        }
    }

    free(segment_lengths);
    free(segment_length_floats);
}


void *piecewiseAggregateThread(void *cache) {
    PAACache *paaCache = (PAACache *) cache;

    VALUE_TYPE const *values = paaCache->values;
    ID_TYPE size = paaCache->size;
    unsigned int series_length = paaCache->series_length;

    VALUE_TYPE *paas = paaCache->paas;
    unsigned int paa_length = paaCache->paa_length;

    unsigned int segment_id;
    unsigned int *segment_lengths = paaCache->segment_lengths;
    VALUE_TYPE *segment_length_floats = paaCache->segment_length_floats;

    ID_TYPE *shared_processed_counter = paaCache->shared_processed_counter;
    unsigned int block_size = paaCache->block_size;

    ID_TYPE start_id, stop_id;
    unsigned int value_counter;
    float segment_sum;
    VALUE_TYPE const *pt_value;
    VALUE_TYPE *pt_paa;

    while ((start_id = __sync_fetch_and_add(shared_processed_counter, block_size)) < size) {
        stop_id = start_id + block_size;
        if (stop_id > size) {
            stop_id = size;
        }

        for (pt_value = values + series_length * start_id, pt_paa = paas + paa_length * start_id,
             value_counter = 0, segment_sum = 0, segment_id = 0;
             pt_value < values + series_length * stop_id; ++pt_value) {
            segment_sum += *pt_value;
            value_counter += 1;

            if (value_counter == segment_lengths[segment_id]) {
                *pt_paa = segment_sum / segment_length_floats[segment_id];

                value_counter = 0;
                segment_sum = 0;

                pt_paa += 1;
                segment_id = (segment_id + 1) % paa_length;
            }
        }
    }

    return NULL;
}


VALUE_TYPE *piecewiseAggregateMT(VALUE_TYPE const *values, ID_TYPE size, unsigned int series_length, unsigned int summarization_length,
                                 unsigned int num_threads) {
    unsigned int *segment_lengths = malloc(sizeof(unsigned int) * summarization_length);
    VALUE_TYPE *segment_length_floats = malloc(sizeof(VALUE_TYPE) * summarization_length);

    for (unsigned int i = 0; i < summarization_length; ++i) {
        segment_lengths[i] = series_length / summarization_length;
    }

    for (unsigned int i = 0; i < series_length % summarization_length; ++i) {
        segment_lengths[i] += 1;
    }

    for (unsigned int i = 0; i < summarization_length; ++i) {
        segment_length_floats[i] = (VALUE_TYPE) segment_lengths[i];
    }

#ifdef ISAX_PROFILING
    unsigned int sum = 0;

    for (unsigned int i = 0; i < summarization_length; ++i) {
        sum += segment_lengths[i];
    }

    if (sum != series_length) {
        clog_info(CLOG(CLOGGER_ID), "cannot assign %d to %d (%d)", series_length, summarization_length, sum);
    }
#endif

    VALUE_TYPE *paas = aligned_alloc(256, sizeof(VALUE_TYPE) * summarization_length * size);

    ID_TYPE shared_processed_counter = 0;
    unsigned int block_size = 1 + size / (num_threads << 2u);

    pthread_t threads[num_threads];
    PAACache paaCaches[num_threads];

    for (unsigned int i = 0; i < num_threads; ++i) {
        paaCaches[i].values = values;
        paaCaches[i].paas = paas;

        paaCaches[i].size = size;
        paaCaches[i].series_length = series_length;
        paaCaches[i].paa_length = summarization_length;

        paaCaches[i].segment_lengths = segment_lengths;
        paaCaches[i].segment_length_floats = segment_length_floats;

        paaCaches[i].shared_processed_counter = &shared_processed_counter;
        paaCaches[i].block_size = block_size;

        pthread_create(&threads[i], NULL, piecewiseAggregateThread, (void *) &paaCaches[i]);
    }

    for (unsigned int i = 0; i < num_threads; ++i) {
        pthread_join(threads[i], NULL);
    }

    free(segment_lengths);
    free(segment_length_floats);

    return paas;
}
