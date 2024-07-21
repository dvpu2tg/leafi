/******************************************************************************
 * Copyright (c) 2024, the LeaFi author(s).
 ******************************************************************************/

#ifndef ISAX_QUERY_ENGINE_WFILTER_H
#define ISAX_QUERY_ENGINE_WFILTER_H

extern "C"
{
#include "globals.h"
#include "config.h"
#include "index.h"
#include "distance.h"
#include "query.h"
#include "clog.h"
#include "answer.h"
#include "sort.h"
#include "str.h"

#include "index_commons.h"
}

#include "static_variables.h"
#include "neural_filter.h"


void conductQueriesWFilter(Config const *config, QuerySet const *querySet, Index const *index);

#endif //ISAX_QUERY_ENGINE_WFILTER_H
