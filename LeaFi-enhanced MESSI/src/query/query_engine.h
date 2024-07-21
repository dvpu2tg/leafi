/******************************************************************************
 * Copyright (c) 2024, the LeaFi author(s).
 ******************************************************************************/

#ifndef ISAX_QUERY_ENGINE_H
#define ISAX_QUERY_ENGINE_H

#include <time.h>
#include <math.h>

#include "globals.h"
#include "config.h"
#include "index.h"
#include "distance.h"
#include "query.h"
#include "clog.h"
#include "answer.h"
#include "sort.h"
#include "str.h"

void profileQueries(Config const *config, QuerySet const *querySet, Index const *index);

void conductQueries(Config const *config, QuerySet const *querySet, Index const *index);

#endif //ISAX_QUERY_ENGINE_H
