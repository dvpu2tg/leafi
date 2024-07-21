/******************************************************************************
 * Copyright (c) 2024, the LeaFi author(s).
 ******************************************************************************/

#include "index.h"
#include "str.h"


SAXSymbol *rootID2SAX(unsigned int id, unsigned int num_segments, unsigned int cardinality) {
    SAXSymbol *sax = aligned_alloc(128, sizeof(SAXSymbol) * SAX_SIMD_ALIGNED_LENGTH);
    unsigned int offset = cardinality - 1;

    for (unsigned int i = 1; i <= num_segments; ++i) {
        sax[num_segments - i] = (SAXSymbol) ((id & 1u) << offset);
        id >>= 1u;
    }

    return sax;
}


unsigned int rootSAX2ID(SAXSymbol const *saxs, unsigned int num_segments, unsigned int cardinality) {
    unsigned int id = 0, offset = cardinality - 1;

    for (unsigned int i = 0; i < num_segments; ++i) {
        id <<= 1u;
        id += (unsigned int) ((saxs[i] & root_mask) >> offset);
    }

    return id;
}


Index *initializeIndex(Config const *config) {
    initializeM256IConstants();

    Index *index = malloc(sizeof(Index));
    if (index == NULL) {
        clog_error(CLOG(CLOGGER_ID), "could not allocate memory to initialize an index");
        exit(EXIT_FAILURE);
    }

    index->series_length = config->series_length;

    index->sax_length = config->sax_length;
    index->sax_cardinality = config->sax_cardinality;

    index->database_size = config->database_size;
    index->num_leaves = 0;
    index->num_filters = 0;
    index->leaves = NULL;

#ifdef FINE_TIMING
    struct timespec start_timestamp, stop_timestamp;
    TimeDiff time_diff;
#endif

    index->cardinality_checker = 1u;
    for (unsigned int i = 0; i < 8 - index->sax_cardinality; ++i) {
        index->cardinality_checker = (index->cardinality_checker << 1) + 1;
    }
#ifdef DEBUG
    clog_debug(CLOG(CLOGGER_ID), "index - cardinality_checker = %s", char2bin(index->cardinality_checker));
#endif
    root_mask = (SAXMask) (1u << 7);
#ifdef DEBUG
    clog_debug(CLOG(CLOGGER_ID), "index - root_mask = %s", char2bin(root_mask));
#endif
#ifdef FINE_TIMING
    clock_code = clock_gettime(CLK_ID, &start_timestamp);
#endif
    index->roots_size = 1u << config->sax_length;
    index->roots = malloc(sizeof(Node *) * index->roots_size);
    SAXMask *root_masks = aligned_alloc(128, sizeof(SAXMask) * config->sax_length);

    for (unsigned int i = 0; i < config->sax_length; ++i) {
        root_masks[i] = (SAXMask) (1u << 7);
    }

    for (unsigned int i = 0; i < index->roots_size; ++i) {
        index->roots[i] = initializeNode(rootID2SAX(i, config->sax_length, 8), root_masks,
                                         config->sax_length, config->sax_cardinality);
    }

#ifdef FINE_TIMING
    clock_code = clock_gettime(CLK_ID, &stop_timestamp);
    getTimeDiff(&time_diff, start_timestamp, stop_timestamp);
    clog_info(CLOG(CLOGGER_ID), "index - initialize roots = %ld.%lds", time_diff.tv_sec, time_diff.tv_nsec);
#endif

    index->values = NULL;
    index->saxs = NULL;
    index->summarizations = NULL;
    index->pos2id = NULL;

    if (!config->load_index) {
#ifdef FINE_TIMING
        clock_code = clock_gettime(CLK_ID, &start_timestamp);
#endif
        VALUE_TYPE *values = aligned_alloc(256, sizeof(VALUE_TYPE) * config->series_length * config->database_size);

        FILE *file_values = fopen(config->database_filepath, "rb");
        size_t read_values = fread(values, sizeof(VALUE_TYPE), config->series_length * config->database_size,
                                   file_values);
        fclose(file_values);
        assert(read_values == config->series_length * config->database_size);

        index->values = (VALUE_TYPE const *) values;

#ifdef FINE_TIMING
        clock_code = clock_gettime(CLK_ID, &stop_timestamp);
        getTimeDiff(&time_diff, start_timestamp, stop_timestamp);
        clog_info(CLOG(CLOGGER_ID), "index - load series = %ld.%lds", time_diff.tv_sec, time_diff.tv_nsec);
        clock_code = clock_gettime(CLK_ID, &start_timestamp);
#endif

        VALUE_TYPE *summarizations = aligned_alloc(256,
                                                   sizeof(VALUE_TYPE) * config->sax_length * config->database_size);

        if (config->database_summarization_filepath != NULL) {
            FILE *file_summarizations = fopen(config->database_summarization_filepath, "rb");
            read_values = fread(summarizations, sizeof(VALUE_TYPE), config->sax_length * config->database_size,
                                file_summarizations);
            fclose(file_summarizations);
            assert(read_values == config->sax_length * config->database_size);
        } else {
            summarizations = piecewiseAggregateMT(index->values, config->database_size, config->series_length,
                                                  config->sax_length, config->max_threads_index);
        }

#ifdef FINE_TIMING
        char *method4summarizations = "load";
        if (config->database_summarization_filepath == NULL) {
            method4summarizations = "calculate";
        }
        clock_code = clock_gettime(CLK_ID, &stop_timestamp);
        getTimeDiff(&time_diff, start_timestamp, stop_timestamp);
        clog_info(CLOG(CLOGGER_ID), "index - %s summarizations = %ld.%lds", method4summarizations, time_diff.tv_sec,
                  time_diff.tv_nsec);
        clock_code = clock_gettime(CLK_ID, &start_timestamp);
#endif

        if (config->use_adhoc_breakpoints) {
            if (config->share_breakpoints) {
                index->breakpoints = getSharedAdhocBreakpoints8(summarizations, config->database_size,
                                                                config->sax_length);
            } else {
                index->breakpoints = getAdhocBreakpoints8(summarizations, config->database_size, config->sax_length,
                                                          config->max_threads_index);
            }
        } else {
            index->breakpoints = getNormalBreakpoints8(config->sax_length);
        }

#ifdef FINE_TIMING
        char *method4breakpoints = "normal";
        if (config->use_adhoc_breakpoints) {
            method4breakpoints = "adhoc";
        }
        clock_code = clock_gettime(CLK_ID, &stop_timestamp);
        getTimeDiff(&time_diff, start_timestamp, stop_timestamp);
        clog_info(CLOG(CLOGGER_ID), "index - load %s breakpoints = %ld.%lds", method4breakpoints, time_diff.tv_sec,
                  time_diff.tv_nsec);
        clock_code = clock_gettime(CLK_ID, &start_timestamp);
#endif

        SAXSymbol *saxs = aligned_alloc(128, sizeof(SAXSymbol) * SAX_SIMD_ALIGNED_LENGTH * config->database_size);
        summarizations2SAX16(saxs, summarizations, index->breakpoints, index->database_size, index->sax_length,
                             index->sax_cardinality, config->max_threads_index);
        index->saxs = (SAXSymbol const *) saxs;

#ifdef FINE_TIMING
        clock_code = clock_gettime(CLK_ID, &stop_timestamp);
        getTimeDiff(&time_diff, start_timestamp, stop_timestamp);
        clog_info(CLOG(CLOGGER_ID), "index - calculate SAXs = %ld.%lds", time_diff.tv_sec, time_diff.tv_nsec);
#endif

        if (config->split_by_summarizations) {
            index->summarizations = (VALUE_TYPE const *) summarizations;
        } else {
            free(summarizations);
            index->summarizations = NULL;
        }
    } else {
        if (config->use_adhoc_breakpoints) {
            index->breakpoints = aligned_alloc(256, sizeof(VALUE_TYPE) * OFFSETS_BY_SEGMENTS[config->sax_length]);
        } else {
            index->breakpoints = getNormalBreakpoints8(config->sax_length);
        }
    }

    index->filter_global_queries = NULL;
    index->filter_global_query_summarizations = NULL;
    index->filter_global_query_saxs = NULL;

    if (config->require_neural_filter) {
        if (!config->load_filters) {
            if (config->filter_query_load_filepath) {
#ifdef FINE_TIMING
                clock_code = clock_gettime(CLK_ID, &start_timestamp);
#endif
                VALUE_TYPE *filter_train_queries = aligned_alloc(256, sizeof(VALUE_TYPE) * config->series_length *
                                                                      config->filter_query_load_size);

                FILE *file_filter_train_queries = fopen(config->filter_query_load_filepath, "rb");
                size_t read_filter_train_queries = fread(filter_train_queries, sizeof(VALUE_TYPE),
                                                         config->series_length * config->filter_query_load_size,
                                                         file_filter_train_queries);
                fclose(file_filter_train_queries);
                assert(read_filter_train_queries == config->series_length * config->filter_query_load_size);

                index->filter_global_queries = (VALUE_TYPE const *) filter_train_queries;

                VALUE_TYPE *filter_train_summarizations = piecewiseAggregateMT(index->filter_global_queries,
                                                                               config->filter_query_load_size,
                                                                               config->series_length,
                                                                               config->sax_length,
                                                                               config->max_threads_index);
                index->filter_global_query_summarizations = (VALUE_TYPE const *) filter_train_summarizations;

                SAXSymbol *filter_train_saxs = aligned_alloc(128, sizeof(SAXSymbol) * SAX_SIMD_ALIGNED_LENGTH *
                                                                  config->filter_query_load_size);
                summarizations2SAX16(filter_train_saxs, filter_train_summarizations, index->breakpoints,
                                     config->filter_query_load_size, index->sax_length,
                                     index->sax_cardinality, config->max_threads_index);
                index->filter_global_query_saxs = (SAXSymbol const *) filter_train_saxs;

#ifdef FINE_TIMING
                clock_code = clock_gettime(CLK_ID, &stop_timestamp);
                getTimeDiff(&time_diff, start_timestamp, stop_timestamp);
                clog_info(CLOG(CLOGGER_ID), "train - prepare queries = %ld.%lds", time_diff.tv_sec,
                          time_diff.tv_nsec);
#endif
            }
        }
    }

    split_range_cache_global = malloc(sizeof(VALUE_TYPE) * config->sax_length);
    split_segment_cache_global = malloc(sizeof(ID_TYPE) * config->sax_length);

    return index;
}


void freeIndex(Index *index) {
    if (index->values != NULL) {
        free((VALUE_TYPE *) index->values);
        index->values = NULL;
    }

    if (index->saxs != NULL) {
        free((SAXSymbol *) index->saxs);
        index->saxs = NULL;
    }
    free((VALUE_TYPE *) index->breakpoints);
    index->breakpoints = NULL;

    if (index->summarizations != NULL) {
        free((VALUE_TYPE *) index->summarizations);
        index->summarizations = NULL;
    }

    if (index->pos2id != NULL) {
        free(index->pos2id);
        index->pos2id = NULL;
    }

    if (index->filter_global_queries != NULL) {
        free((VALUE_TYPE *) index->filter_global_queries);
        index->filter_global_queries = NULL;
    }

    if (index->filter_global_query_summarizations != NULL) {
        free((VALUE_TYPE *) index->filter_global_query_summarizations);
        index->filter_global_query_summarizations = NULL;
    }

    if (index->filter_global_query_saxs != NULL) {
        free((VALUE_TYPE *) index->filter_global_query_saxs);
        index->filter_global_query_saxs = NULL;
    }

    if (index->roots != NULL) {
        bool first_root = true;
        for (unsigned int i = 0; i < index->roots_size; ++i) {
            if (index->roots[i] != NULL) {
                if (first_root) {
                    freeNode(index->roots[i], true, true);
                    first_root = false;
                    index->roots[i] = NULL;
                } else {
                    freeNode(index->roots[i], false, true);
                    index->roots[i] = NULL;
                }
            }
        }

        free(index->roots);
        index->roots = NULL;
    }

    if (index->leaves != NULL) {
        // leaves already freed
        free(index->leaves);
        index->leaves = NULL;
    }

    if (split_range_cache_global != NULL) {
        free(split_range_cache_global);
        split_range_cache_global = NULL;
    }

    if (split_segment_cache_global != NULL) {
        free(split_segment_cache_global);
        split_segment_cache_global = NULL;
    }

    freeGlobalVariables();
}


void logIndex(Config const *config, Index *index, bool print_leaf_size) {
    unsigned int num_series = 0, num_roots = 0, num_series_filter = 0, num_leaves = 0, num_filters = 0;
    for (unsigned int i = 0; i < index->roots_size; ++i) {
        inspectNode(index->roots[i], &num_series, &num_leaves, &num_roots, &num_filters,
                    &num_series_filter, print_leaf_size);
    }

    index->num_leaves = num_leaves;
    index->num_filters = num_filters;

    clog_info(CLOG(CLOGGER_ID), "index - %d series / %d filters in %d leaves from %d / %d roots of %d series",
              num_series_filter, index->num_filters, index->num_leaves, num_roots, index->roots_size, num_series);
#ifdef DEBUG
    if (config->sax_cardinality > 2) {
        if (config->share_breakpoints) {
            Value const *breakpoints2c = index->breakpoints + OFFSETS_BY_CARDINALITY[3];

            clog_debug(CLOG(CLOGGER_ID),
                       "index - breakpoints-4 = %.0f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %.0f",
                       breakpoints2c[0], breakpoints2c[1], breakpoints2c[2], breakpoints2c[3], breakpoints2c[4],
                       breakpoints2c[5], breakpoints2c[6], breakpoints2c[7], breakpoints2c[8], breakpoints2c[9],
                       breakpoints2c[10], breakpoints2c[11], breakpoints2c[12], breakpoints2c[13], breakpoints2c[14],
                       breakpoints2c[15], breakpoints2c[16], breakpoints2c[17]);
        } else {
            for (unsigned int i = 0; i < index->sax_length; ++i) {
                Value const *breakpoints2c = index->breakpoints + OFFSETS_BY_SEGMENTS[i] + OFFSETS_BY_CARDINALITY[3];

                clog_debug(CLOG(CLOGGER_ID),
                           "index - breakpoints-4 of seg-%d = %.0f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %.0f",
                           i, breakpoints2c[0], breakpoints2c[1], breakpoints2c[2], breakpoints2c[3], breakpoints2c[4],
                           breakpoints2c[5], breakpoints2c[6], breakpoints2c[7], breakpoints2c[8], breakpoints2c[9],
                           breakpoints2c[10], breakpoints2c[11], breakpoints2c[12], breakpoints2c[13],
                           breakpoints2c[14], breakpoints2c[15], breakpoints2c[16], breakpoints2c[17]);
            }
        }
    }
#endif
}
