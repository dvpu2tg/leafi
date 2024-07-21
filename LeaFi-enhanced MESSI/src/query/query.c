/******************************************************************************
 * Copyright (c) 2024, the LeaFi author(s).
 ******************************************************************************/

#include "query.h"


QuerySet *initializeQuery(Config const *config, Index const *index) {
    QuerySet *queries = malloc(sizeof(QuerySet));

    queries->query_size = config->query_size;

#ifdef FINE_TIMING
    struct timespec start_timestamp, stop_timestamp;
    TimeDiff time_diff;
    clock_code = clock_gettime(CLK_ID, &start_timestamp);
#endif

    VALUE_TYPE *values = aligned_alloc(256, sizeof(VALUE_TYPE) * config->series_length * config->query_size);
    FILE *file_values = fopen(config->query_filepath, "rb");
    size_t read_values = fread(values, sizeof(VALUE_TYPE), config->series_length * config->query_size, file_values);
    fclose(file_values);
    assert(read_values == config->series_length * config->query_size);

    queries->values = (VALUE_TYPE const *) values;

    if (config->query_bsf_distance_filepath != NULL) {
        VALUE_TYPE *initial_bsf_distances = aligned_alloc(64, sizeof(VALUE_TYPE) * config->query_size);
        FILE *file_bsf_distances = fopen(config->query_bsf_distance_filepath, "rb");
        read_values = fread(initial_bsf_distances, sizeof(VALUE_TYPE), config->query_size, file_bsf_distances);
        fclose(file_bsf_distances);
        assert(read_values == config->query_size);

        queries->initial_bsf_distances = (VALUE_TYPE const *) initial_bsf_distances;
    } else {
        queries->initial_bsf_distances = NULL;
    }

#ifdef FINE_TIMING
    clock_code = clock_gettime(CLK_ID, &stop_timestamp);
    getTimeDiff(&time_diff, start_timestamp, stop_timestamp);
    clog_info(CLOG(CLOGGER_ID), "query - load series = %ld.%lds", time_diff.tv_sec, time_diff.tv_nsec);
    clock_code = clock_gettime(CLK_ID, &start_timestamp);
#endif

    if (config->query_summarization_filepath != NULL) {
        VALUE_TYPE *summarizations = aligned_alloc(256, sizeof(VALUE_TYPE) * config->sax_length * config->query_size);
        FILE *file_summarizations = fopen(config->query_summarization_filepath, "rb");
        read_values = fread(summarizations, sizeof(VALUE_TYPE), config->sax_length * config->query_size,
                            file_summarizations);
        fclose(file_summarizations);
        assert(read_values == config->sax_length * config->query_size);

        queries->summarizations = (VALUE_TYPE const *) summarizations;
    } else {
        queries->summarizations = (VALUE_TYPE const *) piecewiseAggregateMT(queries->values, config->query_size,
                                                                            config->series_length, config->sax_length,
                                                                            config->max_threads_query);
    }

#ifdef FINE_TIMING
    char *method4summarizations = "load";
    if (config->database_summarization_filepath == NULL) {
        method4summarizations = "calculate";
    }
    clock_code = clock_gettime(CLK_ID, &stop_timestamp);
    getTimeDiff(&time_diff, start_timestamp, stop_timestamp);
    clog_info(CLOG(CLOGGER_ID), "query - %s summarizations = %ld.%lds", method4summarizations, time_diff.tv_sec,
              time_diff.tv_nsec);
    clock_code = clock_gettime(CLK_ID, &start_timestamp);
#endif

    SAXSymbol *saxs = aligned_alloc(128, sizeof(SAXSymbol) * SAX_SIMD_ALIGNED_LENGTH * config->query_size);

    summarizations2SAX16(saxs, queries->summarizations, index->breakpoints, queries->query_size, config->sax_length,
                         config->sax_cardinality, config->max_threads_query);

    queries->saxs = (SAXSymbol const *) saxs;

#ifdef FINE_TIMING
    clock_code = clock_gettime(CLK_ID, &stop_timestamp);
    getTimeDiff(&time_diff, start_timestamp, stop_timestamp);
    clog_info(CLOG(CLOGGER_ID), "query - calculate SAXs = %ld.%lds", time_diff.tv_sec, time_diff.tv_nsec);
#endif

    return queries;
}


void freeQuery(QuerySet *queries) {
    if (queries->initial_bsf_distances != NULL) {
        free((VALUE_TYPE *) queries->initial_bsf_distances);
    }

    if (queries->saxs != NULL) {
        free((VALUE_TYPE *) queries->saxs);
    }

    free((VALUE_TYPE *) queries->summarizations);
    free((VALUE_TYPE *) queries->values);
    free(queries);
}
