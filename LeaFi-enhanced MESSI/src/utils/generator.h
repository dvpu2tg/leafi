/******************************************************************************
 * Copyright (c) 2024, the LeaFi author(s).
 ******************************************************************************/

#ifndef ISAX_GENERATOR_H
#define ISAX_GENERATOR_H

#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "globals.h"

// Function to generate Gaussian noise
static VALUE_TYPE generateGaussianNoise(double mu, double sigma) {
    int a = rand() % 999 + 1;
    float f = (float)a;
    f = f / 1000.0f;
    float c = sqrt(-2.0 * log(f));

    int a2 = rand() % 1000;
    float f2 = (float)a2;
    f2 = f2 / 1000.0f;
    float b = 2 * 3.1415926 * f2;

    return (float)c * cos(b) * sigma;

////    float nosie = (float)c * cos(b) * sqrt(noiselevel);
//    float noise =
//    // float nosie=(float)c*cos(b)*noiselevel;
//    // printf("old value = %g", ts_buffer[j]);
//    ts_buffer[j] = ts_buffer[j] + nosie;
//    // printf("           new value =%g\n", ts_buffer[j]);

    static const double epsilon = 0.0000001;
    static const double two_pi = 2.0 * 3.14159265358979323846;
    static double z0, z1;
    static int generate;
    generate = !generate;

    if (!generate)
        return z1 * sigma + mu;

    double u1, u2;
    do {
        u1 = rand() * (1.0 / RAND_MAX);
        u2 = rand() * (1.0 / RAND_MAX);
    } while ( u1 <= epsilon );

    z0 = sqrt(-2.0 * log(u1)) * cos(two_pi * u2);
    z1 = sqrt(-2.0 * log(u1)) * sin(two_pi * u2);
    return z0 * sigma + mu;
}

#endif //ISAX_GENERATOR_H
