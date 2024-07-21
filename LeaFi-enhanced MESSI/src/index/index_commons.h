/******************************************************************************
 * Copyright (c) 2024, the LeaFi author(s).
 ******************************************************************************/

#ifndef ISAX_INDEX_COMMONS_H
#define ISAX_INDEX_COMMONS_H

#include "node.h"
#include "index.h"


void enqueueLeaf(Node *node, Node **leaves, unsigned int *num_leaves, Index *index);

void enqueueFilters(Node *node, NeuralFilter **filters, unsigned int *num_filters);

#endif //ISAX_INDEX_COMMONS_H
