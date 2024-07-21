/******************************************************************************
 * Copyright (c) 2024, the LeaFi author(s).
 ******************************************************************************/

#ifndef ISAX_SORT_H
#define ISAX_SORT_H

#include <stdbool.h>

#include "globals.h"
#include "clog.h"


unsigned int bSearchFloorValue(VALUE_TYPE value, VALUE_TYPE const *sorted, unsigned int first_inclusive, unsigned int last_inclusive);
ID_TYPE bSearchFloorID(ID_TYPE target, ID_TYPE const *sorted, ID_TYPE first_inclusive, ID_TYPE last_inclusive);

unsigned int bSearchByIndicesFloor(VALUE_TYPE value, ID_TYPE const *indices, VALUE_TYPE const *values,
                                   unsigned int first_inclusive, unsigned int last_inclusive);

void qSortFirstHalfIndicesByValue(ID_TYPE *indices, VALUE_TYPE *orders, int first_inclusive, int last_inclusive, VALUE_TYPE pivot);

void qSortFirstHalfIndicesByIndex(ID_TYPE *indices, VALUE_TYPE *orders, ID_TYPE first_inclusive, ID_TYPE last_inclusive, ID_TYPE index_pivot);

void qSortIndicesBy(ID_TYPE *indices, VALUE_TYPE *values, unsigned int first_inclusive, unsigned int last_inclusive);

#endif //ISAX_SORT_H
