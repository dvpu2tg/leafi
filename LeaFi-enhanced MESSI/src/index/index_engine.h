/******************************************************************************
 * Copyright (c) 2024, the LeaFi author(s).
 ******************************************************************************/

#ifndef ISAX_INDEX_ENGINE_H
#define ISAX_INDEX_ENGINE_H

#include <time.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#include "globals.h"
#include "index.h"
#include "clog.h"
#include "config.h"
#include "breakpoints.h"
#include "paa.h"
#include "str.h"
#include "neural_filter.h"


void buildIndex(Config const *config, Index *index);

void finalizeIndex(Config const *config, Index *index, bool free_summarizations);

void generateFilterGlobalQueries(Config *config, Index *index, ID_TYPE num_active_filters);
void searchFilterGlobalQueries(Config const *config, Index *index);
void generateAndSearchFilterLocalQueries(Config const *config, Index *index, ID_TYPE num_active_filters);

void trainNeuralFilters(Config const *config, Index *index, ID_TYPE num_active_filters);

#endif //ISAX_INDEX_ENGINE_H
