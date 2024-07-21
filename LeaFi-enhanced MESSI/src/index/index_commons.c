/******************************************************************************
 * Copyright (c) 2024, the LeaFi author(s).
 ******************************************************************************/

#include "index_commons.h"


void enqueueLeaf(Node *node, Node **leaves, unsigned int *num_leaves, Index *index) {
    if (node != NULL) {
        if (node->size != 0) {
            if (index != NULL) {
                node->index = index;
            }

            leaves[*num_leaves] = node;
            *num_leaves += 1;
        } else if (node->left != NULL) {
            enqueueLeaf(node->left, leaves, num_leaves, index);
            enqueueLeaf(node->right, leaves, num_leaves, index);
        }
    }
}


void enqueueFilters(Node *node, NeuralFilter **filters, unsigned int *num_filters) {
    if (node != NULL) {
        if (node->size != 0 && node->filter) {
            filters[*num_filters] = node->filter;

            *num_filters += 1;
            assert(node->filter_id == *num_filters);
        } else if (node->left != NULL) {
            enqueueFilters(node->left, filters, num_filters);
            enqueueFilters(node->right, filters, num_filters);
        }
    }
}
