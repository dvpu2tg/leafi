/******************************************************************************
 * Copyright (c) 2024, the LeaFi author(s).
 ******************************************************************************/

#ifndef CLOG_MAIN
#define CLOG_MAIN
#endif

#include <cstdio>
#include <ctime>
#include <cassert>
#include <pthread.h>
#include <iostream>

#include <torch/torch.h>

extern "C"
{
#include "clog.h"
#include "globals.h"
#include "config.h"
#include "breakpoints.h"
#include "node.h"
#include "index.h"
#include "index_engine.h"
#include "query.h"
#include "query_engine.h"
}

#include "allocator.h"
#include "neural_filter.h"
#include "query_engine_wfilter.h"


int main(int argc, char **argv) {
    Config const *config = initializeConfig(argc, argv);

    int clog_init_code = clog_init_path(CLOGGER_ID, config->log_filepath);
    if (clog_init_code != 0) {
        fprintf(stderr, "Logger initialization failed, log_filepath = %s\n", config->log_filepath);
        exit(EXIT_FAILURE);
    }

    logConfig(config);
#ifdef ISAX_PROFILING
    log_lock_profiling = static_cast<pthread_mutex_t *>(malloc(sizeof(pthread_mutex_t)));
    assert(pthread_mutex_init(log_lock_profiling, NULL) == 0);
#endif
    Index *index = NULL;

#ifdef TIMING
    struct timespec start_timestamp, stop_timestamp;
    TimeDiff time_diff;

    clock_code = clock_gettime(CLK_ID, &start_timestamp);
#endif
    index = initializeIndex(config);

    if (config->load_index) {
        if (config->use_adhoc_breakpoints) {
            loadBreakpoints8(config->breakpoints_load_filepath, index->breakpoints, config->sax_length);
        }

        for (unsigned int i = 0; i < index->roots_size; ++i) {
            index->roots[i] = loadNode(config, index->roots[i], false, true);
        }
    } else {
        buildIndex(config, index);
        finalizeIndex(config, index, true);
    }
#ifdef TIMING
    clock_code = clock_gettime(CLK_ID, &stop_timestamp);
    getTimeDiff(&time_diff, start_timestamp, stop_timestamp);
    clog_info(CLOG(CLOGGER_ID), "index - %s structure = %ld.%lds",
              config->load_index ? "load" : "build", time_diff.tv_sec, time_diff.tv_nsec);
#endif

    logIndex(config, index, true);

    if (config->require_neural_filter) {
#ifdef TIMING
        clock_code = clock_gettime(CLK_ID, &start_timestamp);
#endif
        int filter_id = 0;
        for (unsigned int i = 0; i < index->roots_size; ++i) {
            initFilterInfoRecursive(config, index->roots[i], &filter_id);
        }
        assert(filter_id == index->num_leaves);
#ifdef ISAX_PROFILING
        clog_info(CLOG(CLOGGER_ID), "index - %d filters (leaves) in total",
                  filter_id, config->load_filters ? "load" : "learn");
#endif
#ifdef TIMING
        clock_code = clock_gettime(CLK_ID, &start_timestamp);
#endif
        initGlobalVariables(config, index->num_filters + 1); // take 0th as guard

        FilterAllocator *allocator = initAllocator(config, index->num_leaves);
        for (unsigned int i = 0; i < index->roots_size; ++i) {
            pushFilters(allocator, index->roots[i]);
        }

        ID_TYPE num_active_filters = -1;
        if (config->load_filters) {
            for (unsigned int i = 0; i < index->roots_size; ++i) {
                loadFilterRecursive(config, index->roots[i]);
            }

            num_active_filters = selectNActivateFilters(allocator, config, false);
        } else {
            if (index->filter_global_queries == nullptr) {
                num_active_filters = selectNActivateFilters(allocator, config, true);
                clog_info(CLOG(CLOGGER_ID), "index - pre-activate %d filters", num_active_filters);

                generateFilterGlobalQueries((Config *) config, index, num_active_filters);
            }

            for (unsigned int i = 0; i < index->roots_size; ++i) {
                addFilterTrainQueryRecursive(config, index->roots[i], index->filter_global_queries, false);
            }

            searchFilterGlobalQueries(config, index);
            generateAndSearchFilterLocalQueries(config, index, num_active_filters);

            num_active_filters = selectNActivateFilters(allocator, config, false);
            trainNeuralFilters(config, index, num_active_filters);
        }

        adjustErr4Recall(allocator, config);

#ifdef TIMING
        clock_code = clock_gettime(CLK_ID, &stop_timestamp);
        getTimeDiff(&time_diff, start_timestamp, stop_timestamp);

        clog_info(CLOG(CLOGGER_ID), "index - %s %d filters = %ld.%lds",
                  config->load_filters ? "load" : "learn", num_active_filters,
                  time_diff.tv_sec, time_diff.tv_nsec);
#endif
        logIndex(config, index, false);
        freeAllocator(allocator);
    }

    if (config->on_disk || config->dump_index) {
        if (config->dump_index || (config->on_disk && !config->load_index)) {
#ifdef FINE_TIMING
            clock_code = clock_gettime(CLK_ID, &start_timestamp);
#endif
            if (config->use_adhoc_breakpoints) {
                dumpBreakpoints8(config->breakpoints_dump_filepath, index->breakpoints, config->sax_length);
            }

            for (unsigned int i = 0; i < index->roots_size; ++i) {
                if (index->roots[i] != NULL) {
                    dumpNode(config, index->roots[i], index->values, index->saxs);
                }
            }
#ifdef FINE_TIMING
            clock_code = clock_gettime(CLK_ID, &stop_timestamp);
            getTimeDiff(&time_diff, start_timestamp, stop_timestamp);
            clog_info(CLOG(CLOGGER_ID), "index - dump nodes = %ld.%lds", time_diff.tv_sec, time_diff.tv_nsec);
#endif
        }

        if (config->on_disk) {
            if (index->saxs != NULL) {
                free((SAXSymbol *) index->saxs);
                index->saxs = NULL;
            }
            if (index->values != NULL) {
                free((VALUE_TYPE *) index->values);
                index->values = NULL;
            }
        }
    }

    QuerySet *queries = initializeQuery(config, index);

#ifdef TIMING
    clock_code = clock_gettime(CLK_ID, &start_timestamp);
#endif

    if (config->to_profile) {
        profileQueries(config, queries, index);
    } else {
        if (config->require_neural_filter) {
            conductQueriesWFilter(config, queries, index);
        } else {
            conductQueries(config, queries, index);
        }
    }

#ifdef TIMING
    clock_code = clock_gettime(CLK_ID, &stop_timestamp);
    getTimeDiff(&time_diff, start_timestamp, stop_timestamp);

    clog_info(CLOG(CLOGGER_ID), "query - overall = %ld.%lds", time_diff.tv_sec, time_diff.tv_nsec);
#endif
#ifdef ISAX_PROFILING
    pthread_mutex_destroy(log_lock_profiling);
    free(log_lock_profiling);
#endif

    if (config->require_neural_filter && config->dump_filters) {
        assert(!config->load_filters);
#ifdef FINE_TIMING
        clock_code = clock_gettime(CLK_ID, &start_timestamp);
#endif
        for (unsigned int i = 0; i < index->roots_size; ++i) {
            if (index->roots[i] != NULL) {
                dumpFilters(config, index->roots[i]);
            }
        }
#ifdef FINE_TIMING
        clock_code = clock_gettime(CLK_ID, &stop_timestamp);
        getTimeDiff(&time_diff, start_timestamp, stop_timestamp);
        clog_info(CLOG(CLOGGER_ID), "index - dump filters = %ld.%lds", time_diff.tv_sec, time_diff.tv_nsec);
#endif
    }

    freeIndex(index);

    freeQuery(queries);
    freeConfig((Config *) config);
    free((Config *) config);

    clog_free(CLOGGER_ID);

    return 0;
}
