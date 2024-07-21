/******************************************************************************
 * Copyright (c) 2024, the LeaFi author(s).
 ******************************************************************************/

#ifndef ISAX_SAX_H
#define ISAX_SAX_H

#include <stdlib.h>
#include <pthread.h>
#include <immintrin.h>

#include "globals.h"
#include "sort.h"
#include "distance.h"
#include "breakpoints.h"


static unsigned int const SHIFTS_BY_MASK[129] = {
        0, 0, 1, 0, 2, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 7
};


static unsigned int const BITS_BY_MASK[129] = {
        0, 8, 7, 0, 6, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 1
};


static unsigned int const PREFIX_MASKS_BY_MASK[129] = {
        0, 255, 254, 0, 252, 0, 0, 0, 248, 0, 0, 0, 0, 0, 0, 0, 240, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 224, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 192, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 128
};


static SAXMask const MASKS_BY_BITS[9] = {0u, 1u << 7u, 1u << 6u, 1u << 5u, 1u << 4u,
                                         1u << 3u, 1u << 2u, 1u << 1u, 1u};


void summarizations2SAX16(SAXSymbol *saxs, VALUE_TYPE const *summarizations, VALUE_TYPE const *breakpoints, ID_TYPE size,
                          unsigned int sax_length, unsigned int sax_cardinality, unsigned int num_threads);

VALUE_TYPE l2SquareValue2SAXByMask(unsigned int sax_length, VALUE_TYPE const *summarizations, SAXSymbol const *sax,
                                   SAXMask const *masks, VALUE_TYPE const *breakpoints, VALUE_TYPE scale_factor);

VALUE_TYPE l2SquareValue2SAX8(unsigned int sax_length, VALUE_TYPE const *summarizations, SAXSymbol const *sax,
                              VALUE_TYPE const *breakpoints, VALUE_TYPE scale_factor);

VALUE_TYPE l2SquareValue2SAXByMaskSIMD(unsigned int sax_length, VALUE_TYPE const *summarizations, SAXSymbol const *sax,
                                       SAXMask const *masks, VALUE_TYPE const *breakpoints, VALUE_TYPE scale_factor, VALUE_TYPE *cache);

VALUE_TYPE l2SquareValue2EnvelopSIMD(unsigned int sax_length, VALUE_TYPE const *summarizations, VALUE_TYPE const *upper_envelops,
                                     VALUE_TYPE const *lower_envelops, VALUE_TYPE scale_factor, VALUE_TYPE *cache);

VALUE_TYPE l2SquareSummarization2SAX8SIMD(unsigned int sax_length, VALUE_TYPE const *summarizations, SAXSymbol const *sax,
                                          VALUE_TYPE const *breakpoints, VALUE_TYPE scale_factor, VALUE_TYPE *cache);

char *sax2str(SAXSymbol const *sax, SAXMask const *masks, unsigned int sax_length, unsigned int sax_cardinality);

char *mask2str(SAXMask const *masks, char *mask_str, unsigned int sax_length, unsigned int sax_cardinality);

#endif //ISAX_SAX_H
