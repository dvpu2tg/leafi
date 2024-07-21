/******************************************************************************
 * Copyright (c) 2024, the LeaFi author(s).
 ******************************************************************************/

#ifndef ISAX_NODE_H
#define ISAX_NODE_H

#include <assert.h>
#include <stdlib.h>
#include <stdbool.h>
#include <pthread.h>

#include "str.h"
#include "clog.h"
#include "globals.h"
#include "config.h"
#include "distance.h"
#include "neural_filter.h"
#include "allocator.h"


typedef struct Node {
    void *index;

    pthread_mutex_t *lock;
    SAXSymbol *sax;
    SAXMask *masks;
    char *sax_str;
    SAXMask *squeezed_masks;

    ID_TYPE *ids;
    ID_TYPE start_id;
    unsigned int size;
    unsigned int capacity;

    SAXSymbol *saxs;
    VALUE_TYPE *values;

    struct Node *left;
    struct Node *right;

    VALUE_TYPE compactness;

    VALUE_TYPE *upper_envelops;
    VALUE_TYPE *lower_envelops;

    NeuralFilter *filter;
    int filter_id;

    char *node_dump_filepath;
    char *data_dump_filepath;
    char *filter_dump_filepath;
//    char *query_dump_filepath;

    char *node_load_filepath;
    char *data_load_filepath;
    char *filter_load_filepath;
} Node;

Node *initializeNode(SAXSymbol *saxWord, SAXMask *saxMask, unsigned int sax_length, unsigned int sax_cardinality);

Node *initializeNode4Load(char *sax_str);

void initFilterInfoRecursive(Config const *config, Node *node, int *filter_id);
void addFilterTrainQueryRecursive(Config const *config, Node *node, VALUE_TYPE const *filter_global_queries,
                                  bool if_check_activate);

void pushFilters(FilterAllocator *allocator, Node *node);

void inspectNode(Node *node, unsigned int *num_series, unsigned int *num_leaves, unsigned int *num_roots,
                 unsigned int *num_filters, unsigned int *num_series_filter, bool print_leaf_size);

bool testSeriesInNodeEnvelope(Node const *node, VALUE_TYPE const *summarizations, unsigned int sax_length,
                              VALUE_TYPE const *breakpoints);

VALUE_TYPE getCompactness(Node *leaf_node, VALUE_TYPE const *values, unsigned int series_length);

void insertNode(Node *leaf, ID_TYPE id, unsigned int initial_leaf_size, unsigned int leaf_size);

void dumpNode(Config const *config, Node *node, VALUE_TYPE const *values, SAXSymbol const *saxs);

void dumpFilters(Config const *config, Node *node);

Node *loadNode(Config const *config, Node *node, bool free_mask, bool free_sax);

void loadFilterRecursive(Config const *config, Node *node);

void cleanNode(Node *node);

void freeNode(Node *node, bool free_mask, bool free_sax);

#endif //ISAX_NODE_H
