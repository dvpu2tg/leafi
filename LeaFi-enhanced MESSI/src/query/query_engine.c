/******************************************************************************
 * Copyright (c) 2024, the LeaFi author(s).
 ******************************************************************************/

#include "query_engine.h"
#include "index_commons.h"


void queryNodeThreadCore(Answer *answer, Node const *node, VALUE_TYPE const *values, unsigned int series_length,
                         SAXSymbol const *saxs, unsigned int sax_length, VALUE_TYPE const *breakpoints,
                         VALUE_TYPE scale_factor,
                         VALUE_TYPE const *query_values, VALUE_TYPE const *query_summarization,
                         VALUE_TYPE *m256_fetched_cache,
                         pthread_rwlock_t *lock, ID_TYPE *pos2id) {
    VALUE_TYPE local_l2SquareSAX, local_l2Square, local_bsf = getBSF(answer);
    unsigned long pos;

    VALUE_TYPE const *start_value_ptr = NULL, *current_series;
    if (values) {
        start_value_ptr = values + series_length * node->start_id;
    } else if (node->values != NULL) {
        start_value_ptr = node->values;
    }

    SAXSymbol const *outer_current_sax = NULL, *current_sax;
    if (saxs) {
        outer_current_sax = saxs + SAX_SIMD_ALIGNED_LENGTH * node->start_id;
    } else if (node->saxs != NULL) {
        outer_current_sax = node->saxs;
    }

    SAXSymbol *saxs2load = NULL;
    VALUE_TYPE *values2load = NULL;
    if (start_value_ptr == NULL && outer_current_sax == NULL) {
        saxs2load = aligned_alloc(128, sizeof(SAXSymbol) * SAX_SIMD_ALIGNED_LENGTH * node->size);
        values2load = aligned_alloc(256, sizeof(VALUE_TYPE) * series_length * node->size);

        FILE *data_file = fopen(node->data_load_filepath, "rb");

        size_t nitems = fread(saxs2load, sizeof(SAXSymbol), SAX_SIMD_ALIGNED_LENGTH * node->size, data_file);
        assert(nitems == SAX_SIMD_ALIGNED_LENGTH * node->size);

        nitems = fread(values2load, sizeof(VALUE_TYPE), series_length * node->size, data_file);
        assert(nitems == series_length * node->size);

        fclose(data_file);

        outer_current_sax = (SAXSymbol const *) saxs2load;
        start_value_ptr = (VALUE_TYPE const *) values2load;
    } else {
        assert(start_value_ptr != NULL & outer_current_sax != NULL);
    }

    for (current_series = start_value_ptr, current_sax = outer_current_sax;
         current_series < start_value_ptr + series_length * node->size;
         current_series += series_length, current_sax += SAX_SIMD_ALIGNED_LENGTH) {
#ifdef ISAX_PROFILING
        __sync_fetch_and_add(&sum2sax_counter_profiling, 1);
#endif
        local_l2SquareSAX = l2SquareSummarization2SAX8SIMD(sax_length, query_summarization, current_sax,
                                                           breakpoints, scale_factor, m256_fetched_cache);

        if (VALUE_G(local_bsf, local_l2SquareSAX)) {
#ifdef ISAX_PROFILING
            __sync_fetch_and_add(&l2square_counter_profiling, 1);
#endif
            local_l2Square = l2SquareEarlySIMD(series_length, query_values, current_series, local_bsf,
                                               m256_fetched_cache);

            if (VALUE_G(local_bsf, local_l2Square)) {
#ifdef DEBUG
                pthread_mutex_lock(log_lock_profiling);
                clog_info(CLOG(CLOGGER_ID),
                          "query %d - updated BSF = %f <- %f at %d l2square / %d sum2sax / %d entered",
                          query_id_profiling, local_l2Square, local_bsf,
                          l2square_counter_profiling, sum2sax_counter_profiling, leaf_counter_profiling);
                pthread_mutex_unlock(log_lock_profiling);
#endif
                pthread_rwlock_wrlock(lock);

                if (pos2id) {
                    if (checkBSF(answer, local_l2Square)) {
                        pos = node->start_id + (current_series - start_value_ptr) / series_length;
                        updateBSFWithID(answer, local_l2Square, pos2id[pos]);
                    }
                } else {
                    checkNUpdateBSF(answer, local_l2Square);
                }

                pthread_rwlock_unlock(lock);
                local_bsf = getBSF(answer);
            }
        }
    }

    if (saxs2load != NULL) {
        free(saxs2load);
    }
    if (values2load != NULL) {
        free(values2load);
    }
}


void
queryNodeNotBoundingThreadCore(Answer *answer, Node const *node, VALUE_TYPE const *values, unsigned int series_length,
                               VALUE_TYPE const *query_values, VALUE_TYPE *m256_fetched_cache, pthread_rwlock_t *lock,
                               ID_TYPE *pos2id) {
    VALUE_TYPE local_l2Square, local_bsf = getBSF(answer);
    unsigned long pos;

    VALUE_TYPE const *start_value_ptr, *current_series;
    if (values) {
        start_value_ptr = values + series_length * node->start_id;
    } else {
        VALUE_TYPE *values2load = aligned_alloc(256, sizeof(VALUE_TYPE) * series_length * node->size);

        FILE *file_values = fopen(node->data_load_filepath, "rb");
        size_t read_values = fread(values2load, sizeof(VALUE_TYPE), series_length * node->size, file_values);
        fclose(file_values);
        assert(read_values == series_length * node->size);

        start_value_ptr = (VALUE_TYPE const *) values2load;
    }

    for (current_series = start_value_ptr;
         current_series < start_value_ptr + series_length * node->size;
         current_series += series_length) {
#ifdef ISAX_PROFILING
        __sync_fetch_and_add(&l2square_counter_profiling, 1);
#endif
        local_l2Square = l2SquareEarlySIMD(series_length, query_values, current_series, local_bsf,
                                           m256_fetched_cache);

        if (VALUE_G(local_bsf, local_l2Square)) {
            pthread_rwlock_wrlock(lock);

            if (pos2id) {
                if (checkBSF(answer, local_l2Square)) {
                    pos = node->start_id + (current_series - start_value_ptr) / series_length;
                    updateBSFWithID(answer, local_l2Square, pos2id[pos]);
                }
            } else {
                checkNUpdateBSF(answer, local_l2Square);
            }

            pthread_rwlock_unlock(lock);
            local_bsf = getBSF(answer);
        }
    }

    if (!values) {
        free((VALUE_TYPE *) start_value_ptr);
    }
}


void *queryThread(void *cache) {
    QueryCache *queryCache = (QueryCache *) cache;

    VALUE_TYPE const *values = NULL;
    SAXSymbol const *saxs = NULL;
    VALUE_TYPE const *breakpoints = NULL;
    ID_TYPE *pos2id = NULL;

#ifdef FINE_PROFILING
    bool log_leaf_visits = queryCache->log_leaf_visits;
#endif

    if (queryCache->index != NULL) {
        breakpoints = queryCache->index->breakpoints;
        values = queryCache->index->values;
        saxs = queryCache->index->saxs;
        pos2id = queryCache->index->pos2id;
    }

    unsigned int series_length = queryCache->series_length;
    unsigned int sax_length = queryCache->sax_length;

    Node const *const *leaves = queryCache->leaves;
    VALUE_TYPE *leaf_distances = queryCache->leaf_distances;
    ID_TYPE *leaf_indices = queryCache->leaf_indices;

    VALUE_TYPE const *query_summarization = queryCache->query_summarization;
    VALUE_TYPE const *query_values = queryCache->query_values;
    Node *resident_node = queryCache->resident_node;

    Answer *answer = queryCache->answer;
    pthread_rwlock_t *lock = answer->lock;

    VALUE_TYPE *m256_fetched_cache = queryCache->m256_fetched_cache;
    VALUE_TYPE scale_factor = queryCache->scale_factor;

    unsigned int block_size = queryCache->query_block_size;
    unsigned int num_leaves = queryCache->num_leaves;
    ID_TYPE *shared_index_id = queryCache->shared_leaf_id;

    bool sort_leaves = queryCache->sort_leaves;
    bool lower_bounding = queryCache->lower_bounding;

    unsigned int series_limitations = queryCache->series_limitations;

    ID_TYPE leaf_id;
    unsigned int index_id, stop_index_id;
    VALUE_TYPE local_bsf;

    while ((index_id = __sync_fetch_and_add(shared_index_id, block_size)) < num_leaves) {
        stop_index_id = index_id + block_size;
        if (stop_index_id > num_leaves) {
            stop_index_id = num_leaves;
        }

        while (index_id < stop_index_id) {
            leaf_id = leaf_indices[index_id];

            if (resident_node != leaves[leaf_id]) {
                local_bsf = getBSF(answer);

                if ((VALUE_GEQ(local_bsf, leaf_distances[leaf_id]) || !lower_bounding)
                    && VALUE_G(VALUE_MAX, leaf_distances[leaf_id])) {
#ifdef ISAX_PROFILING
                    __sync_fetch_and_add(&leaf_counter_profiling, 1);

#ifdef FINE_PROFILING
                    if (log_leaf_visits) {
                        clog_info(CLOG(CLOGGER_ID), "query %d - BSF = %f when visit %d node %s",
                                  query_id_profiling, local_bsf, leaf_counter_profiling, leaves[leaf_id]->sax_str);
                    }
#endif
                    if (series_limitations > 0 && (sum2sax_counter_profiling > series_limitations ||
                                                   l2square_counter_profiling > series_limitations)) {
                        return NULL;
                    }
#endif
                    if (queryCache->index == NULL) {
                        Index *current_index = (Index *) leaves[leaf_id]->index;

                        breakpoints = current_index->breakpoints;
                        values = current_index->values;
                        saxs = current_index->saxs;
                        pos2id = current_index->pos2id;
                    }

                    if (lower_bounding) {
                        queryNodeThreadCore(answer, leaves[leaf_id], values, series_length, saxs, sax_length,
                                            breakpoints, scale_factor, query_values, query_summarization,
                                            m256_fetched_cache, lock, pos2id);
                    } else {
                        queryNodeNotBoundingThreadCore(answer, leaves[leaf_id], values, series_length, query_values,
                                                       m256_fetched_cache, lock, pos2id);
                    }
                } else if (lower_bounding && sort_leaves && VALUE_G(VALUE_MAX, leaf_distances[leaf_id])) {
                    return NULL;
                }
            }

            index_id += 1;
        }
    }

    return NULL;
}

void *leafThread(void *cache) {
    QueryCache *queryCache = (QueryCache *) cache;

    VALUE_TYPE const *breakpoints = NULL;
    if (queryCache->index != NULL) {
        breakpoints = queryCache->index->breakpoints;
    }
    unsigned int sax_length = queryCache->sax_length;

    Node *resident_node = queryCache->resident_node;
    VALUE_TYPE *leaf_distances = queryCache->leaf_distances;

    VALUE_TYPE const *query_summarization = queryCache->query_summarization;
    VALUE_TYPE scale_factor = queryCache->scale_factor;
    VALUE_TYPE *m256_fetched_cache = queryCache->m256_fetched_cache;

    unsigned int block_size = queryCache->leaf_block_size;
    unsigned int num_leaves = queryCache->num_leaves;
    ID_TYPE *shared_leaf_id = queryCache->shared_leaf_id;

    ID_TYPE leaf_id, stop_leaf_id;
    Node const *leaf;

    while ((leaf_id = __sync_fetch_and_add(shared_leaf_id, block_size)) < num_leaves) {
        stop_leaf_id = leaf_id + block_size;
        if (stop_leaf_id > num_leaves) {
            stop_leaf_id = num_leaves;
        }

        for (unsigned int i = leaf_id; i < stop_leaf_id; ++i) {
            leaf = queryCache->leaves[i];

            if (queryCache->index == NULL) {
                breakpoints = ((Index *) leaf->index)->breakpoints;
            }

            if (resident_node != NULL && leaf == resident_node) {
                leaf_distances[i] = VALUE_MAX;
            } else {
                if (leaf->upper_envelops != NULL) {
                    leaf_distances[i] = l2SquareValue2EnvelopSIMD(sax_length, query_summarization,
                                                                  leaf->upper_envelops, leaf->lower_envelops,
                                                                  scale_factor, m256_fetched_cache);
                } else if (leaf->squeezed_masks != NULL) {
                    leaf_distances[i] = l2SquareValue2SAXByMaskSIMD(sax_length, query_summarization, leaf->sax,
                                                                    leaf->squeezed_masks, breakpoints, scale_factor,
                                                                    m256_fetched_cache);
                } else {
                    leaf_distances[i] = l2SquareValue2SAXByMaskSIMD(sax_length, query_summarization, leaf->sax,
                                                                    leaf->masks, breakpoints, scale_factor,
                                                                    m256_fetched_cache);
                }
            }
        }
    }

    return NULL;
}


void queryNode(Answer *answer, Node const *node,
               VALUE_TYPE const *values, unsigned int series_length,
               SAXSymbol const *saxs, unsigned int sax_length,
               VALUE_TYPE const *breakpoints, VALUE_TYPE scale_factor,
               VALUE_TYPE const *query_values, VALUE_TYPE const *query_summarization,
               VALUE_TYPE *m256_fetched_cache,
               ID_TYPE *pos2id) {
    VALUE_TYPE local_l2SquareSAX8, local_l2Square, local_bsf = getBSF(answer);
    unsigned long pos;

    VALUE_TYPE const *start_value_ptr = NULL;
    if (values) {
        start_value_ptr = values + series_length * node->start_id;
    } else if (node->values != NULL) {
        start_value_ptr = node->values;
    }

    SAXSymbol const *outer_current_sax = NULL;
    if (saxs) {
        outer_current_sax = saxs + SAX_SIMD_ALIGNED_LENGTH * node->start_id;
    } else if (node->saxs != NULL) {
        outer_current_sax = node->saxs;
    }

    SAXSymbol *saxs2load = NULL;
    VALUE_TYPE *values2load = NULL;
    if (start_value_ptr == NULL && outer_current_sax == NULL) {
        saxs2load = aligned_alloc(128, sizeof(SAXSymbol) * SAX_SIMD_ALIGNED_LENGTH * node->size);
        values2load = aligned_alloc(256, sizeof(VALUE_TYPE) * series_length * node->size);

        FILE *data_file = fopen(node->data_load_filepath, "rb");

        size_t nitems = fread(saxs2load, sizeof(SAXSymbol), SAX_SIMD_ALIGNED_LENGTH * node->size, data_file);
        assert(nitems == SAX_SIMD_ALIGNED_LENGTH * node->size);

        nitems = fread(values2load, sizeof(VALUE_TYPE), series_length * node->size, data_file);
        assert(nitems == series_length * node->size);

        fclose(data_file);

        outer_current_sax = (SAXSymbol const *) saxs2load;
        start_value_ptr = (VALUE_TYPE const *) values2load;
    } else {
        assert(start_value_ptr != NULL & outer_current_sax != NULL);
    }

    VALUE_TYPE const *outer_current_series = start_value_ptr;
    while (outer_current_series < start_value_ptr + series_length * node->size) {
#ifdef ISAX_PROFILING
        sum2sax_counter_profiling += 1;
#endif
        local_l2SquareSAX8 = l2SquareSummarization2SAX8SIMD(sax_length, query_summarization, outer_current_sax,
                                                            breakpoints, scale_factor, m256_fetched_cache);

        if (VALUE_G(local_bsf, local_l2SquareSAX8)) {
#ifdef ISAX_PROFILING
            l2square_counter_profiling += 1;
#endif
            local_l2Square = l2SquareEarlySIMD(series_length, query_values, outer_current_series, local_bsf,
                                               m256_fetched_cache);

            if (VALUE_G(local_bsf, local_l2Square)) {
                if (pos2id) {
                    if (checkBSF(answer, local_l2Square)) {
                        pos = node->start_id + (outer_current_series - start_value_ptr) / series_length;
                        updateBSFWithID(answer, local_l2Square, pos2id[pos]);
                    }
                } else {
                    checkNUpdateBSF(answer, local_l2Square);
                }

                local_bsf = getBSF(answer);
            }
        }

        outer_current_series += series_length;
        outer_current_sax += SAX_SIMD_ALIGNED_LENGTH;
    }

    if (saxs2load != NULL) {
        free(saxs2load);
    }
    if (values2load != NULL) {
        free(values2load);
    }
}


void queryNodeNotBounding(Answer *answer, Node const *node, VALUE_TYPE const *values, unsigned int series_length,
                          VALUE_TYPE const *query_values, VALUE_TYPE *m256_fetched_cache, ID_TYPE *pos2id) {
    VALUE_TYPE local_l2Square, local_bsf = getBSF(answer);
    unsigned long pos;

    VALUE_TYPE const *start_value_ptr;
    if (values) {
        start_value_ptr = values + series_length * node->start_id;
    } else {
        VALUE_TYPE *values2load = aligned_alloc(256, sizeof(VALUE_TYPE) * series_length * node->size);

        FILE *file_values = fopen(node->data_load_filepath, "rb");
        size_t read_values = fread(values2load, sizeof(VALUE_TYPE), series_length * node->size, file_values);
        fclose(file_values);
        assert(read_values == series_length * node->size);

        start_value_ptr = (VALUE_TYPE const *) values2load;
    }

    VALUE_TYPE const *outer_current_series = start_value_ptr;
    while (outer_current_series < start_value_ptr + series_length * node->size) {
#ifdef ISAX_PROFILING
        l2square_counter_profiling += 1;
#endif
        local_l2Square = l2SquareEarlySIMD(series_length, query_values, outer_current_series, local_bsf,
                                           m256_fetched_cache);

        if (VALUE_G(local_bsf, local_l2Square)) {
            if (pos2id) {
                if (checkBSF(answer, local_l2Square)) {
                    pos = node->start_id + (outer_current_series - start_value_ptr) / series_length;
                    updateBSFWithID(answer, local_l2Square, pos2id[pos]);
                }
            } else {
                checkNUpdateBSF(answer, local_l2Square);
            }

            local_bsf = getBSF(answer);
        }

        outer_current_series += series_length;
    }

    if (!values) {
        free((VALUE_TYPE *) start_value_ptr);
    }
}


void profileNodeThreadCore(Answer *answer, Node const *node, VALUE_TYPE const *values, unsigned int series_length,
                           SAXSymbol const *saxs, unsigned int sax_length, VALUE_TYPE const *breakpoints,
                           VALUE_TYPE scale_factor,
                           VALUE_TYPE const *query_values, VALUE_TYPE const *query_summarization,
                           VALUE_TYPE *m256_fetched_cache,
                           pthread_rwlock_t *lock, ID_TYPE *pos2id, VALUE_TYPE node_dist) {
    VALUE_TYPE local_l2Square, local_bsf = getBSF(answer), local_bsf_l2Square = VALUE_MAX;
    unsigned long pos;

    VALUE_TYPE const *start_value_ptr = NULL, *current_series;
    if (values) {
        start_value_ptr = values + series_length * node->start_id;
    } else if (node->values != NULL) {
        start_value_ptr = node->values;
    }

    SAXSymbol const *outer_current_sax = NULL, *current_sax;
    if (saxs) {
        outer_current_sax = saxs + SAX_SIMD_ALIGNED_LENGTH * node->start_id;
    } else if (node->saxs != NULL) {
        outer_current_sax = node->saxs;
    }

    SAXSymbol *saxs2load = NULL;
    VALUE_TYPE *values2load = NULL;
    if (start_value_ptr == NULL && outer_current_sax == NULL) {
        saxs2load = aligned_alloc(128, sizeof(SAXSymbol) * SAX_SIMD_ALIGNED_LENGTH * node->size);
        values2load = aligned_alloc(256, sizeof(VALUE_TYPE) * series_length * node->size);

        FILE *data_file = fopen(node->data_load_filepath, "rb");

        size_t nitems = fread(saxs2load, sizeof(SAXSymbol), SAX_SIMD_ALIGNED_LENGTH * node->size, data_file);
        assert(nitems == SAX_SIMD_ALIGNED_LENGTH * node->size);

        nitems = fread(values2load, sizeof(VALUE_TYPE), series_length * node->size, data_file);
        assert(nitems == series_length * node->size);

        fclose(data_file);

        outer_current_sax = (SAXSymbol const *) saxs2load;
        start_value_ptr = (VALUE_TYPE const *) values2load;
    } else {
        assert(start_value_ptr != NULL & outer_current_sax != NULL);
    }

    for (current_series = start_value_ptr;
         current_series < start_value_ptr + series_length * node->size;
         current_series += series_length) {
#ifdef ISAX_PROFILING
        __sync_fetch_and_add(&l2square_counter_profiling, 1);
#endif
        local_l2Square = l2SquareEarlySIMD(series_length, query_values, current_series, local_bsf_l2Square,
                                           m256_fetched_cache);

        if (VALUE_G(local_bsf_l2Square, local_l2Square)) {
            local_bsf_l2Square = local_l2Square;

            if (VALUE_G(local_bsf, local_bsf_l2Square)) {
                pthread_rwlock_wrlock(lock);
                checkNUpdateBSF(answer, local_l2Square);
                pthread_rwlock_unlock(lock);
            }
        }
    }

#ifdef ISAX_PROFILING
    pthread_mutex_lock(log_lock_profiling);

    clog_info(CLOG(CLOGGER_ID), "query %d - node %s lb %.5f bsf %.5f lnn2 %.5f",
              query_id_profiling, node->sax_str, node_dist, local_bsf, local_bsf_l2Square);

    pthread_mutex_unlock(log_lock_profiling);
#endif

    local_bsf = getBSF(answer);

    if (saxs2load != NULL) {
        free(saxs2load);
    }
    if (values2load != NULL) {
        free(values2load);
    }
}


void *profileThread(void *cache) {
    QueryCache *queryCache = (QueryCache *) cache;

    VALUE_TYPE const *values = NULL;
    SAXSymbol const *saxs = NULL;
    VALUE_TYPE const *breakpoints = NULL;
    ID_TYPE *pos2id = NULL;

#ifdef FINE_PROFILING
    bool log_leaf_visits = queryCache->log_leaf_visits;
#endif

    if (queryCache->index != NULL) {
        breakpoints = queryCache->index->breakpoints;
        values = queryCache->index->values;
        saxs = queryCache->index->saxs;
        pos2id = queryCache->index->pos2id;
    }

    unsigned int series_length = queryCache->series_length;
    unsigned int sax_length = queryCache->sax_length;

    Node const *const *leaves = queryCache->leaves;
    VALUE_TYPE *leaf_distances = queryCache->leaf_distances;
    ID_TYPE *leaf_indices = queryCache->leaf_indices;

    VALUE_TYPE const *query_summarization = queryCache->query_summarization;
    VALUE_TYPE const *query_values = queryCache->query_values;
    Node *resident_node = queryCache->resident_node;

    Answer *answer = queryCache->answer;
    pthread_rwlock_t *lock = answer->lock;

    VALUE_TYPE *m256_fetched_cache = queryCache->m256_fetched_cache;
    VALUE_TYPE scale_factor = queryCache->scale_factor;

    unsigned int block_size = queryCache->query_block_size;
    unsigned int num_leaves = queryCache->num_leaves;
    ID_TYPE *shared_index_id = queryCache->shared_leaf_id;

    bool sort_leaves = queryCache->sort_leaves;
    bool lower_bounding = queryCache->lower_bounding;

    unsigned int series_limitations = queryCache->series_limitations;

    ID_TYPE leaf_id;
    unsigned int index_id, stop_index_id;
    VALUE_TYPE local_bsf;

    while ((index_id = __sync_fetch_and_add(shared_index_id, block_size)) < num_leaves) {
        stop_index_id = index_id + block_size;
        if (stop_index_id > num_leaves) {
            stop_index_id = num_leaves;
        }

        while (index_id < stop_index_id) {
            leaf_id = leaf_indices[index_id];

            if (resident_node != leaves[leaf_id]) {
                __sync_fetch_and_add(&leaf_counter_profiling, 1);
                local_bsf = getBSF(answer);

                if ((VALUE_GEQ(local_bsf, leaf_distances[leaf_id]) && VALUE_G(VALUE_MAX, leaf_distances[leaf_id]))
                    || (series_limitations > 0 && l2square_counter_profiling < series_limitations)) {
                    profileNodeThreadCore(answer, leaves[leaf_id], values, series_length, saxs, sax_length,
                                          breakpoints, scale_factor, query_values, query_summarization,
                                          m256_fetched_cache, lock, pos2id, leaf_distances[leaf_id]);
                } else {
#ifdef ISAX_PROFILING
                    __sync_fetch_and_add(&l2square_counter_profiling, leaves[leaf_id]->size);

                    pthread_mutex_lock(log_lock_profiling);
                    clog_info(CLOG(CLOGGER_ID), "query %d - node %s lb %.5f bsf %.5f lnn2 %.5f",
                              query_id_profiling, leaves[leaf_id]->sax_str,
                              leaf_distances[leaf_id], local_bsf, VALUE_MAX);
                    pthread_mutex_unlock(log_lock_profiling);
#endif
                }
            }

            index_id += 1;
        }
    }

    return NULL;
}


void profileNode(Answer *answer, Node const *node,
                 VALUE_TYPE const *values, unsigned int series_length,
                 SAXSymbol const *saxs, unsigned int sax_length,
                 VALUE_TYPE const *breakpoints, VALUE_TYPE scale_factor,
                 VALUE_TYPE const *query_values, VALUE_TYPE const *query_summarization,
                 VALUE_TYPE *m256_fetched_cache,
                 ID_TYPE *pos2id,
                 ID_TYPE query_id, VALUE_TYPE node_dist) {
    VALUE_TYPE local_l2SquareSAX8, local_l2Square, local_bsf = getBSF(answer), local_bsf_l2Square = VALUE_MAX;
    unsigned long pos;

    VALUE_TYPE const *start_value_ptr = NULL;
    if (values) {
        start_value_ptr = values + series_length * node->start_id;
    } else if (node->values != NULL) {
        start_value_ptr = node->values;
    }

    SAXSymbol const *outer_current_sax = NULL;
    if (saxs) {
        outer_current_sax = saxs + SAX_SIMD_ALIGNED_LENGTH * node->start_id;
    } else if (node->saxs != NULL) {
        outer_current_sax = node->saxs;
    }

    SAXSymbol *saxs2load = NULL;
    VALUE_TYPE *values2load = NULL;
    if (start_value_ptr == NULL && outer_current_sax == NULL) {
        saxs2load = aligned_alloc(128, sizeof(SAXSymbol) * SAX_SIMD_ALIGNED_LENGTH * node->size);
        values2load = aligned_alloc(256, sizeof(VALUE_TYPE) * series_length * node->size);

        FILE *data_file = fopen(node->data_load_filepath, "rb");

        size_t nitems = fread(saxs2load, sizeof(SAXSymbol), SAX_SIMD_ALIGNED_LENGTH * node->size, data_file);
        assert(nitems == SAX_SIMD_ALIGNED_LENGTH * node->size);

        nitems = fread(values2load, sizeof(VALUE_TYPE), series_length * node->size, data_file);
        assert(nitems == series_length * node->size);

        fclose(data_file);

        outer_current_sax = (SAXSymbol const *) saxs2load;
        start_value_ptr = (VALUE_TYPE const *) values2load;
    } else {
        assert(start_value_ptr != NULL & outer_current_sax != NULL);
    }

    VALUE_TYPE const *outer_current_series = start_value_ptr;
    while (outer_current_series < start_value_ptr + series_length * node->size) {
#ifdef ISAX_PROFILING
        l2square_counter_profiling += 1;
#endif

        local_l2Square = l2SquareEarlySIMD(series_length, query_values, outer_current_series, local_bsf_l2Square,
                                           m256_fetched_cache);

        if (VALUE_G(local_bsf_l2Square, local_l2Square)) {
            local_bsf_l2Square = local_l2Square;

            if (VALUE_G(local_bsf, local_bsf_l2Square)) {
                checkNUpdateBSF(answer, local_bsf_l2Square);
            }
        }

        outer_current_series += series_length;
    }

#ifdef ISAX_PROFILING
    leaf_counter_profiling += 1;

    clog_info(CLOG(CLOGGER_ID), "query %d - node %s lb %.5f bsf %.5f lnn2 %.5f",
              query_id, node->sax_str, node_dist, local_bsf, local_bsf_l2Square);
#endif

    local_bsf = getBSF(answer);

    if (saxs2load != NULL) {
        free(saxs2load);
    }
    if (values2load != NULL) {
        free(values2load);
    }
}


void profileQueries(Config const *config, QuerySet const *querySet, Index const *index) {
    Answer *answer = initializeAnswer(config);

    VALUE_TYPE const *values = index->values;
    SAXSymbol const *saxs = index->saxs;
    ID_TYPE *pos2id = index->pos2id;
    VALUE_TYPE const *breakpoints = index->breakpoints;
    unsigned int series_length = config->series_length;
    unsigned int sax_length = config->sax_length;
    VALUE_TYPE scale_factor = config->scale_factor;

    ID_TYPE shared_leaf_id;

    unsigned int max_threads_query = config->max_threads_query;
    QueryCache queryCache[max_threads_query];

#ifdef FINE_TIMING
    struct timespec start_timestamp, stop_timestamp;
    TimeDiff time_diff;
    clock_code = clock_gettime(CLK_ID, &start_timestamp);
#endif

    Node **leaves = malloc(sizeof(Node *) * index->num_leaves);
    unsigned int num_leaves = 0;
    for (unsigned int j = 0; j < index->roots_size; ++j) {
        enqueueLeaf(index->roots[j], leaves, &num_leaves, NULL);
    }
    assert(num_leaves == index->num_leaves);

#ifdef FINE_TIMING
    clock_code = clock_gettime(CLK_ID, &stop_timestamp);
    getTimeDiff(&time_diff, start_timestamp, stop_timestamp);
    clog_info(CLOG(CLOGGER_ID), "query - fetch leaves = %ld.%lds", time_diff.tv_sec, time_diff.tv_nsec);
#endif

    ID_TYPE *leaf_indices = malloc(sizeof(ID_TYPE) * num_leaves);
    for (ID_TYPE j = 0; j < num_leaves; ++j) {
        leaf_indices[j] = j;
    }

    VALUE_TYPE *query_and_bsf = malloc(sizeof(VALUE_TYPE) * (series_length + 1));

    VALUE_TYPE *leaf_distances = malloc(sizeof(VALUE_TYPE) * num_leaves);
    unsigned int leaf_block_size = 1 + num_leaves / (max_threads_query << 1u);
    unsigned int query_block_size = 2 + num_leaves / (max_threads_query << 3u);

    for (unsigned int i = 0; i < max_threads_query; ++i) {
        queryCache[i].answer = answer;
        queryCache[i].index = index;

        queryCache[i].num_leaves = num_leaves;
        queryCache[i].leaves = (Node const *const *) leaves;
        queryCache[i].leaf_indices = leaf_indices;
        queryCache[i].leaf_distances = leaf_distances;

        queryCache[i].scale_factor = scale_factor;
        queryCache[i].m256_fetched_cache = aligned_alloc(256, sizeof(VALUE_TYPE) * 8);

        queryCache[i].shared_leaf_id = &shared_leaf_id;
        queryCache[i].sort_leaves = config->sort_leaves;

        queryCache[i].series_limitations = config->series_limitations;
        queryCache[i].lower_bounding = config->lower_bounding;

        queryCache[i].series_length = config->series_length;
        queryCache[i].sax_length = config->sax_length;

        queryCache[i].log_leaf_visits = config->log_leaf_visits;

        queryCache[i].query_block_size = query_block_size;
    }

    VALUE_TYPE *local_m256_fetched_cache = queryCache[0].m256_fetched_cache;

    VALUE_TYPE const *query_values, *query_summarization;
    SAXSymbol const *query_sax;
    VALUE_TYPE local_bsf;
    Node *node;

    for (unsigned int query_id = 0; query_id < querySet->query_size; ++query_id) {
#ifdef ISAX_PROFILING
        query_id_profiling = query_id;
        leaf_counter_profiling = 0;
        sum2sax_counter_profiling = 0;
        l2square_counter_profiling = 0;
#endif
        if (querySet->initial_bsf_distances == NULL) {
            resetAnswer(answer);
        }
        local_bsf = getBSF(answer);

        query_values = querySet->values + series_length * query_id;
        query_summarization = querySet->summarizations + sax_length * query_id;
        query_sax = querySet->saxs + SAX_SIMD_ALIGNED_LENGTH * query_id;

        node = index->roots[rootSAX2ID(query_sax, sax_length, 8)];

        if (node != NULL) {
            while (node->left != NULL) {
                node = route(node, query_sax, sax_length);
            }

            profileNode(answer, node, values, series_length, saxs, sax_length, breakpoints, scale_factor,
                        query_values, query_summarization, local_m256_fetched_cache, pos2id, query_id, 0);

            local_bsf = getBSF(answer);
        } else {
            clog_info(CLOG(CLOGGER_ID), "query %d - no resident node", query_id);
        }

        pthread_t leaves_threads[max_threads_query];
        shared_leaf_id = 0;

        for (unsigned int j = 0; j < max_threads_query; ++j) {
            queryCache[j].query_values = query_values;
            queryCache[j].query_summarization = query_summarization;
            queryCache[j].leaf_block_size = leaf_block_size;

            pthread_create(&leaves_threads[j], NULL, leafThread, (void *) &queryCache[j]);
        }

        for (unsigned int j = 0; j < max_threads_query; ++j) {
            pthread_join(leaves_threads[j], NULL);
        }

        qSortIndicesBy(leaf_indices, leaf_distances, 0, (int) (num_leaves - 1));

        if (node == NULL) {
            node = leaves[leaf_indices[0]];

            profileNode(answer, node, values, series_length, saxs, sax_length, breakpoints, scale_factor,
                        query_values, query_summarization, local_m256_fetched_cache, pos2id,
                        query_id, leaf_distances[leaf_indices[0]]);

            local_bsf = getBSF(answer);
        }

        leaf_distances[leaf_indices[0]] = VALUE_MAX; // already recorded

        pthread_t query_threads[max_threads_query];
        shared_leaf_id = 0;

        for (unsigned int j = 0; j < max_threads_query; ++j) {
            queryCache[j].resident_node = node;

            pthread_create(&query_threads[j], NULL, profileThread, (void *) &queryCache[j]);
        }

        for (unsigned int j = 0; j < max_threads_query; ++j) {
            pthread_join(query_threads[j], NULL);
        }


        if ((config->exact_search && !(VALUE_EQ(local_bsf, 0) && answer->size == answer->k)) || node == NULL) {
#ifdef FINE_TIMING
            clock_code = clock_gettime(CLK_ID, &start_timestamp);
#endif
            if (config->exact_search && !(VALUE_EQ(local_bsf, 0) && answer->size == answer->k)) {
#ifdef FINE_TIMING
                clock_code = clock_gettime(CLK_ID, &start_timestamp);
#endif
#ifdef FINE_TIMING
                clock_code = clock_gettime(CLK_ID, &stop_timestamp);
                getTimeDiff(&time_diff, start_timestamp, stop_timestamp);
                clog_info(CLOG(CLOGGER_ID), "query %d - exact search = %ld.%lds", query_id, time_diff.tv_sec,
                          time_diff.tv_nsec);
#endif
            }
        }
#ifdef ISAX_PROFILING
        clog_info(CLOG(CLOGGER_ID), "query %d - %d l2square / %d sum2sax / %d entered", query_id,
                  l2square_counter_profiling, sum2sax_counter_profiling, leaf_counter_profiling);
#endif
        logAnswer(query_id, answer);
    }

    for (unsigned int i = 0; i < max_threads_query; ++i) {
        free(queryCache[i].m256_fetched_cache);
    }

    freeAnswer(answer);

    free(leaves);
    free(leaf_distances);
    free(leaf_indices);

    free(query_and_bsf);
}


void conductQueries(Config const *config, QuerySet const *querySet, Index const *index) {
    Answer *answer = initializeAnswer(config);

    VALUE_TYPE const *values = index->values;
    SAXSymbol const *saxs = index->saxs;
    ID_TYPE *pos2id = index->pos2id;
    VALUE_TYPE const *breakpoints = index->breakpoints;
    unsigned int series_length = config->series_length;
    unsigned int sax_length = config->sax_length;
    VALUE_TYPE scale_factor = config->scale_factor;

    ID_TYPE shared_leaf_id;

    unsigned int max_threads_query = config->max_threads_query;
    QueryCache queryCache[max_threads_query];

#ifdef FINE_TIMING
    struct timespec start_timestamp, stop_timestamp;
    TimeDiff time_diff;
    clock_code = clock_gettime(CLK_ID, &start_timestamp);
#endif

    Node **leaves = malloc(sizeof(Node *) * index->num_leaves);
    unsigned int num_leaves = 0;
    for (unsigned int j = 0; j < index->roots_size; ++j) {
        enqueueLeaf(index->roots[j], leaves, &num_leaves, NULL);
    }
    assert(num_leaves == index->num_leaves);

#ifdef FINE_TIMING
    clock_code = clock_gettime(CLK_ID, &stop_timestamp);
    getTimeDiff(&time_diff, start_timestamp, stop_timestamp);
    clog_info(CLOG(CLOGGER_ID), "query - fetch leaves = %ld.%lds", time_diff.tv_sec, time_diff.tv_nsec);
#endif

    ID_TYPE *leaf_indices = malloc(sizeof(ID_TYPE) * num_leaves);
    for (ID_TYPE j = 0; j < num_leaves; ++j) {
        leaf_indices[j] = j;
    }

    VALUE_TYPE *query_and_bsf = malloc(sizeof(VALUE_TYPE) * (series_length + 1));

    VALUE_TYPE *leaf_distances = malloc(sizeof(VALUE_TYPE) * num_leaves);
    unsigned int leaf_block_size = 1 + num_leaves / (max_threads_query << 1u);
    unsigned int query_block_size = 2 + num_leaves / (max_threads_query << 3u);

    for (unsigned int i = 0; i < max_threads_query; ++i) {
        queryCache[i].answer = answer;
        queryCache[i].index = index;

        queryCache[i].num_leaves = num_leaves;
        queryCache[i].leaves = (Node const *const *) leaves;
        queryCache[i].leaf_indices = leaf_indices;
        queryCache[i].leaf_distances = leaf_distances;

        queryCache[i].scale_factor = scale_factor;
        queryCache[i].m256_fetched_cache = aligned_alloc(256, sizeof(VALUE_TYPE) * 8);

        queryCache[i].shared_leaf_id = &shared_leaf_id;
        queryCache[i].sort_leaves = config->sort_leaves;

        queryCache[i].series_limitations = config->series_limitations;
        queryCache[i].lower_bounding = config->lower_bounding;

        queryCache[i].series_length = config->series_length;
        queryCache[i].sax_length = config->sax_length;

        queryCache[i].log_leaf_visits = config->log_leaf_visits;

        queryCache[i].query_block_size = query_block_size;
    }

    VALUE_TYPE *local_m256_fetched_cache = queryCache[0].m256_fetched_cache;

    VALUE_TYPE const *query_values, *query_summarization;
    SAXSymbol const *query_sax;
    VALUE_TYPE local_bsf;
    Node *node;

    for (unsigned int query_id = 0; query_id < querySet->query_size; ++query_id) {
#ifdef ISAX_PROFILING
        query_id_profiling = query_id;
        leaf_counter_profiling = 0;
        sum2sax_counter_profiling = 0;
        l2square_counter_profiling = 0;
#endif
        if (querySet->initial_bsf_distances == NULL) {
            resetAnswer(answer);
        } else {
            resetAnswerBy(answer, querySet->initial_bsf_distances[query_id]);
            clog_info(CLOG(CLOGGER_ID), "query %d - initial 1bsf = %f", query_id,
                      querySet->initial_bsf_distances[query_id]);
        }
        local_bsf = getBSF(answer);

        query_values = querySet->values + series_length * query_id;
        query_summarization = querySet->summarizations + sax_length * query_id;
        query_sax = querySet->saxs + SAX_SIMD_ALIGNED_LENGTH * query_id;

#ifdef FINE_TIMING
        clock_code = clock_gettime(CLK_ID, &start_timestamp);
#endif
        node = index->roots[rootSAX2ID(query_sax, sax_length, 8)];

        if ((querySet->initial_bsf_distances == NULL && config->exact_search) && node != NULL) {
            while (node->left != NULL) {
                node = route(node, query_sax, sax_length);
            }
#ifdef ISAX_PROFILING
            leaf_counter_profiling += 1;

#ifdef FINE_PROFILING
            if (config->log_leaf_visits) {
                clog_info(CLOG(CLOGGER_ID), "query %d - BSF = %f when visit %d node %s",
                          query_id_profiling, local_bsf, leaf_counter_profiling, node->sax_str);
            }
#endif
            if (config->leaf_compactness) {
                clog_info(CLOG(CLOGGER_ID), "query %d - resident leaf size %d compactness %f",
                          query_id, node->size, getCompactness(node, values, series_length));
            }

            if (config->log_leaf_only) {
                continue;
            }
#endif
            if (config->lower_bounding) {
                queryNode(answer, node, values, series_length, saxs, sax_length, breakpoints, scale_factor,
                          query_values, query_summarization, local_m256_fetched_cache, pos2id);
            } else {
                queryNodeNotBounding(answer, node, values, series_length, query_values, local_m256_fetched_cache,
                                     pos2id);
            }
            local_bsf = getBSF(answer);
#ifdef ISAX_PROFILING
            clog_info(CLOG(CLOGGER_ID), "query %d - %d l2square / %d sum2sax in resident leaf",
                      query_id, l2square_counter_profiling, sum2sax_counter_profiling);
#endif
#ifdef FINE_TIMING
            clock_code = clock_gettime(CLK_ID, &stop_timestamp);
            getTimeDiff(&time_diff, start_timestamp, stop_timestamp);
            clog_info(CLOG(CLOGGER_ID), "query %d - resident-leaf approximate search = %ld.%lds", query_id,
                      time_diff.tv_sec, time_diff.tv_nsec);
#endif
        } else {
            if (!(querySet->initial_bsf_distances == NULL && config->exact_search)) {
                clog_info(CLOG(CLOGGER_ID), "query %d - no resident node", query_id);
            }
        }

        if ((config->exact_search && !(VALUE_EQ(local_bsf, 0) && answer->size == answer->k)) || node == NULL) {
#ifdef FINE_TIMING
            clock_code = clock_gettime(CLK_ID, &start_timestamp);
#endif
            pthread_t leaves_threads[max_threads_query];
            shared_leaf_id = 0;

            for (unsigned int j = 0; j < max_threads_query; ++j) {
                queryCache[j].query_values = query_values;
                queryCache[j].query_summarization = query_summarization;
                queryCache[j].leaf_block_size = leaf_block_size;

                pthread_create(&leaves_threads[j], NULL, leafThread, (void *) &queryCache[j]);
            }

            for (unsigned int j = 0; j < max_threads_query; ++j) {
                pthread_join(leaves_threads[j], NULL);
            }

            if (config->sort_leaves) {
                qSortFirstHalfIndicesByValue(leaf_indices, leaf_distances, 0, (int) (num_leaves - 1), local_bsf);

                leaf_distances[leaf_indices[0]] = VALUE_MAX;
            }
#ifdef FINE_TIMING
            clock_code = clock_gettime(CLK_ID, &stop_timestamp);
            getTimeDiff(&time_diff, start_timestamp, stop_timestamp);
            clog_info(CLOG(CLOGGER_ID), "query %d - cal&sort leaf distances = %ld.%lds", query_id,
                      time_diff.tv_sec, time_diff.tv_nsec);
#endif

            if (!(querySet->initial_bsf_distances == NULL && config->exact_search) && node == NULL) {
                node = leaves[leaf_indices[0]];

#ifdef ISAX_PROFILING
                leaf_counter_profiling += 1;

#ifdef FINE_PROFILING
                if (config->log_leaf_visits) {
                    clog_info(CLOG(CLOGGER_ID), "query %d - BSF = %f when visit %d node %s",
                              query_id_profiling, local_bsf, leaf_counter_profiling, node->sax_str);
                }
#endif
                if (config->leaf_compactness) {
                    for (unsigned int j = 0; j < index->sax_length; ++j) {
                        clog_info(CLOG(CLOGGER_ID), "query %d - nearest leaf segment %d = %d - %d", query_id, j,
                                  node->sax[j],
                                  node->masks[j]);
                    }

                    clog_info(CLOG(CLOGGER_ID), "query %d - nearest leaf size %d compactness %f", query_id, node->size,
                              getCompactness(node, values, series_length));
                }

                if (config->log_leaf_only) {
                    continue;
                }
#endif
#ifdef FINE_TIMING
                clock_code = clock_gettime(CLK_ID, &start_timestamp);
#endif
                if (config->lower_bounding) {
                    queryNode(answer, node, values, series_length, saxs, sax_length, breakpoints, scale_factor,
                              query_values, query_summarization, local_m256_fetched_cache, pos2id);
                } else {
                    queryNodeNotBounding(answer, node, values, series_length, query_values, local_m256_fetched_cache,
                                         pos2id);
                }
                local_bsf = getBSF(answer);

#ifdef ISAX_PROFILING
                clog_info(CLOG(CLOGGER_ID), "query %d - %d l2square / %d sum2sax in closest leaf",
                          query_id, l2square_counter_profiling, sum2sax_counter_profiling);
#endif
#ifdef FINE_TIMING
                clock_code = clock_gettime(CLK_ID, &stop_timestamp);
                getTimeDiff(&time_diff, start_timestamp, stop_timestamp);
                clog_info(CLOG(CLOGGER_ID), "query %d - closest-leaf approximate search = %ld.%lds", query_id,
                          time_diff.tv_sec, time_diff.tv_nsec);
#endif
            }

            if (config->exact_search && !(VALUE_EQ(local_bsf, 0) && answer->size == answer->k)) {
#ifdef FINE_TIMING
                clock_code = clock_gettime(CLK_ID, &start_timestamp);
#endif
                pthread_t query_threads[max_threads_query];
                shared_leaf_id = 0;

                for (unsigned int j = 0; j < max_threads_query; ++j) {
                    queryCache[j].resident_node = node;

                    pthread_create(&query_threads[j], NULL, queryThread, (void *) &queryCache[j]);
                }

                for (unsigned int j = 0; j < max_threads_query; ++j) {
                    pthread_join(query_threads[j], NULL);
                }
#ifdef FINE_TIMING
                clock_code = clock_gettime(CLK_ID, &stop_timestamp);
                getTimeDiff(&time_diff, start_timestamp, stop_timestamp);
                clog_info(CLOG(CLOGGER_ID), "query %d - exact search = %ld.%lds", query_id, time_diff.tv_sec,
                          time_diff.tv_nsec);

#endif
            }
        }
#ifdef ISAX_PROFILING
        clog_info(CLOG(CLOGGER_ID), "query %d - %d l2square / %d sum2sax / %d entered", query_id,
                  l2square_counter_profiling, sum2sax_counter_profiling, leaf_counter_profiling);
#endif
        logAnswer(query_id, answer);
    }

    for (unsigned int i = 0; i < max_threads_query; ++i) {
        free(queryCache[i].m256_fetched_cache);
    }

    freeAnswer(answer);

    free(leaves);
    free(leaf_distances);
    free(leaf_indices);

    free(query_and_bsf);
}
