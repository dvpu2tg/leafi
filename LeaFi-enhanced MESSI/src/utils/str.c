/******************************************************************************
 * Copyright (c) 2024, the LeaFi author(s).
 ******************************************************************************/

#include "str.h"


int sax_symbol_to_binary_str(SAXSymbol symbol, char *symbol_str) {
    size_t bits = sizeof(SAXSymbol) * CHAR_BIT;

//    char *str = malloc(bits + 1);
    if (symbol_str == NULL) {
        return -1;
    }
    symbol_str[bits] = 0;

    // type punning because signed shift is implementation-defined
    for (unsigned u = *(unsigned *) &symbol; bits--; u >>= 1) {
        symbol_str[bits] = u & 1 ? '1' : '0';
    }

    return 0;
}


// credit: https://stackoverflow.com/a/11394336
char* concat(int count, ...)
{
    va_list ap;
    int i;

    // Find required length to store merged string
    int len = 1; // room for NULL
    va_start(ap, count);
    for(i=0 ; i<count ; i++)
        len += strlen(va_arg(ap, char*));
    va_end(ap);

    // Allocate memory to concat strings
    char *merged = calloc(sizeof(char),len);
    int null_pos = 0;

    // Actually concatenate strings
    va_start(ap, count);
    for(i=0 ; i<count ; i++)
    {
        char *s = va_arg(ap, char*);
        strcpy(merged+null_pos, s);
        null_pos += strlen(s);
    }
    va_end(ap);

    return merged;
}