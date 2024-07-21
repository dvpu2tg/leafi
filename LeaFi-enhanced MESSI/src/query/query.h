/******************************************************************************
 * Copyright (c) 2024, the LeaFi author(s).
 ******************************************************************************/

#ifndef ISAX_QUERY_H
#define ISAX_QUERY_H

#include <time.h>
#include <stdlib.h>

#include "globals.h"
#include "config.h"
#include "paa.h"
#include "breakpoints.h"
#include "sax.h"
#include "index.h"
#include "clog.h"
#include "answer.h"


typedef struct QuerySet {
    VALUE_TYPE const *values;
    VALUE_TYPE const *summarizations;
    VALUE_TYPE const *initial_bsf_distances;
    SAXSymbol const *saxs;

    unsigned int query_size;
} QuerySet;


typedef struct QueryCache {
    Index const *index;

    unsigned int series_length;
    unsigned int sax_length;

    Node const *const *leaves;
    ID_TYPE *leaf_indices;
    VALUE_TYPE *leaf_distances;
    unsigned int num_leaves;

    Answer *answer;
    VALUE_TYPE const *query_values;
    VALUE_TYPE const *query_summarization;
    Node *resident_node;

    ID_TYPE *shared_leaf_id;
    unsigned int leaf_block_size;
    unsigned int query_block_size;

    VALUE_TYPE *m256_fetched_cache;
    VALUE_TYPE scale_factor;

    bool sort_leaves;
    bool lower_bounding;

    unsigned int series_limitations;

    // for SIMS and Ideal
    SAXSymbol const *saxs;
    VALUE_TYPE const *summarizations;
    VALUE_TYPE const *breakpoints;

    ID_TYPE size;
    unsigned int sax_cardinality;

    unsigned int block_size;
    ID_TYPE *shared_processed_counter;

    VALUE_TYPE *distances;

    bool log_leaf_visits;

    int stream_id;
    VALUE_TYPE *filter_predictions;

    int thread_id;
} QueryCache;


QuerySet *initializeQuery(Config const *config, Index const *index);

void freeQuery(QuerySet *queries);


#endif //ISAX_QUERY_H
