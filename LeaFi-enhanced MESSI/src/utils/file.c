/******************************************************************************
 * Copyright (c) 2024, the LeaFi author(s).
 ******************************************************************************/

#include "file.h"

#include <unistd.h>

int checkFileExists(char *filepath) { return access(filepath, F_OK) >= 0; }
