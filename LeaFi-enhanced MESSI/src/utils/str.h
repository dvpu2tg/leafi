/******************************************************************************
 * Copyright (c) 2024, the LeaFi author(s).
 ******************************************************************************/

#ifndef ISAX_STR_H
#define ISAX_STR_H

#include <stdlib.h>       // calloc
#include <stdarg.h>       // va_*
#include <string.h>       // strlen, strcpy
#include <limits.h>

#include "globals.h"


int sax_symbol_to_binary_str(SAXSymbol symbol, char *symbol_str);

char* concat(int count, ...);

#endif //ISAX_STR_H
