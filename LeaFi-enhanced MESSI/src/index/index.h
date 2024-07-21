/******************************************************************************
 * Copyright (c) 2024, the LeaFi author(s).
 ******************************************************************************/

#ifndef ISAX_INDEX_H
#define ISAX_INDEX_H

#include <time.h>

#include "globals.h"
#include "node.h"
#include "config.h"
#include "clog.h"
#include "breakpoints.h"
#include "paa.h"
#include "sax.h"


typedef struct Index {
    Node **roots;
    Node **leaves;

    unsigned int roots_size;
    unsigned int num_leaves;
    unsigned int num_filters;

    VALUE_TYPE const *values;

    ID_TYPE database_size;
    unsigned int series_length;

    SAXSymbol const *saxs;
    VALUE_TYPE const *breakpoints;
    VALUE_TYPE const *summarizations;
    ID_TYPE *pos2id;

    unsigned int sax_length;
    unsigned int sax_cardinality;
    SAXMask cardinality_checker;

    VALUE_TYPE const *filter_global_queries;
    VALUE_TYPE const *filter_global_query_summarizations;
    SAXSymbol const *filter_global_query_saxs;
} Index;

Node *route(Node const *parent, SAXSymbol const *sax, unsigned int num_segments);

SAXSymbol *rootID2SAX(unsigned int id, unsigned int num_segments, unsigned int cardinality);

unsigned int rootSAX2ID(SAXSymbol const *saxs, unsigned int num_segments, unsigned int cardinality);

Index *initializeIndex(Config const *config);

void freeIndex(Index *index);

void logIndex(Config const *config, Index *index, bool print_leaf_size);

#endif //ISAX_INDEX_H
