/******************************************************************************
 * Copyright (c) 2024, the LeaFi author(s).
 ******************************************************************************/

#include "sort.h"


unsigned int bSearchFloorValue(VALUE_TYPE value, VALUE_TYPE const *sorted, unsigned int first_inclusive, unsigned int last_inclusive) {
    while (first_inclusive + 1 < last_inclusive) {
        unsigned int mid = (first_inclusive + last_inclusive) >> 1u;

        if (VALUE_L(value, sorted[mid])) {
            last_inclusive = mid;
        } else {
            first_inclusive = mid;
        }
    }

    return first_inclusive;
}


unsigned int bSearchByIndicesFloor(VALUE_TYPE value, ID_TYPE const *indices, VALUE_TYPE const *values,
                                   unsigned int first_inclusive, unsigned int last_inclusive) {
    while (first_inclusive + 1 < last_inclusive) {
        unsigned int mid = (first_inclusive + last_inclusive) >> 1u;

        if (VALUE_L(value, values[indices[mid]])) {
            last_inclusive = mid;
        } else {
            first_inclusive = mid;
        }
    }

    return first_inclusive;
}


// following C.A.R. Hoare as in https://en.wikipedia.org/wiki/Quicksort#Hoare_partition_scheme
void qSortIndicesBy(ID_TYPE *indices, VALUE_TYPE *values, unsigned int first_inclusive, unsigned int last_inclusive) {
    if (first_inclusive < last_inclusive) {
        unsigned int tmp_index;
        VALUE_TYPE pivot = values[indices[(unsigned int) (first_inclusive + last_inclusive) >> 1u]];

        if (first_inclusive + 3 < last_inclusive) {
            VALUE_TYPE smaller = values[indices[first_inclusive]], larger = values[indices[last_inclusive]];
            if (VALUE_L(larger, smaller)) {
                smaller = values[indices[last_inclusive]], larger = values[indices[first_inclusive]];
            }

            if (VALUE_L(pivot, smaller)) {
                pivot = smaller;
            } else if (VALUE_G(pivot, larger)) {
                pivot = larger;
            }
        }

        int first_g = (int) first_inclusive - 1, last_l = (int) last_inclusive + 1;
        while (true) {
            do {
                first_g += 1;
            } while (VALUE_L(values[indices[first_g]], pivot));

            do {
                last_l -= 1;
            } while (VALUE_G(values[indices[last_l]], pivot));

            if (first_g >= last_l) {
                break;
            }

            tmp_index = indices[first_g];
            indices[first_g] = indices[last_l];
            indices[last_l] = tmp_index;
        }

        qSortIndicesBy(indices, values, first_inclusive, last_l);
        qSortIndicesBy(indices, values, last_l + 1, last_inclusive);
    }
}


void qSortFirstHalfIndicesByValue(ID_TYPE *indices, VALUE_TYPE *orders, int first_inclusive, int last_inclusive, VALUE_TYPE pivot) {
    if (first_inclusive < last_inclusive) {
        unsigned int tmp_index;

        int first_g = first_inclusive - 1, last_l = last_inclusive + 1;

        while (true) {
            do {
                first_g += 1;
            } while (first_g <= last_inclusive && VALUE_L(orders[indices[first_g]], pivot));

            do {
                last_l -= 1;
            } while (last_l >= first_inclusive && VALUE_G(orders[indices[last_l]], pivot));

            if (first_g >= last_l) {
                break;
            }

            tmp_index = indices[first_g];
            indices[first_g] = indices[last_l];
            indices[last_l] = tmp_index;
        }

        qSortIndicesBy(indices, orders, first_inclusive, last_l < last_inclusive ? last_l : last_inclusive);
    }
}


void qSortFirstHalfIndicesByIndex(ID_TYPE *indices, VALUE_TYPE *orders, ID_TYPE first_inclusive, ID_TYPE last_inclusive, ID_TYPE index_pivot) {
    if (first_inclusive < last_inclusive) {
        unsigned int tmp_index;
        VALUE_TYPE pivot = orders[indices[(unsigned int) (first_inclusive + last_inclusive) >> 1u]];

        if (first_inclusive + 3 < last_inclusive) {
            VALUE_TYPE smaller = orders[indices[first_inclusive]], larger = orders[indices[last_inclusive]];

            if (last_inclusive - first_inclusive >= 5 * index_pivot) { // based on estimation of quartiles
                if (VALUE_G(smaller, larger)) {
                    smaller = larger;
                }

                if (VALUE_G(pivot, smaller)) {
                    pivot = smaller;
                }
            } else {
                if (VALUE_L(larger, smaller)) {
                    smaller = orders[indices[last_inclusive]], larger = orders[indices[first_inclusive]];
                }

                if (VALUE_L(pivot, smaller)) {
                    pivot = smaller;
                } else if (VALUE_G(pivot, larger)) {
                    pivot = larger;
                }
            }
        }

        ID_TYPE first_g = first_inclusive - 1, last_l = last_inclusive + 1;
        while (true) {
            do {
                first_g += 1;
            } while (first_g <= last_inclusive && VALUE_L(orders[indices[first_g]], pivot));

            do {
                last_l -= 1;
            } while (last_l >= first_inclusive && VALUE_G(orders[indices[last_l]], pivot));

            if (first_g >= last_l) {
                break;
            }

            tmp_index = indices[first_g];
            indices[first_g] = indices[last_l];
            indices[last_l] = tmp_index;
        }

        qSortFirstHalfIndicesByIndex(indices, orders, first_inclusive, last_l, index_pivot);
        if (last_l < index_pivot) {
            qSortFirstHalfIndicesByIndex(indices, orders, last_l + 1, last_inclusive, index_pivot);
        }
    }
}


ID_TYPE bSearchFloorID(ID_TYPE target, const ID_TYPE *sorted, ID_TYPE first_inclusive, ID_TYPE last_inclusive) {
    while (first_inclusive + 1 < last_inclusive) {
        unsigned int mid = (first_inclusive + last_inclusive) >> 1u;

        if (VALUE_L(target, sorted[mid])) {
            last_inclusive = mid;
        } else {
            first_inclusive = mid;
        }
    }

    return first_inclusive;
}
