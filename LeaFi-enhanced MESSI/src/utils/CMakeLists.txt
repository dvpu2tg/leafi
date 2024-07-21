cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

add_library(utils SHARED generator.h distance.c sort.c config.c file.c str.c globals.c)

# -mavx -mavx2 -mavx512 -msse4.2 are supported by Intel(R) Xeon(R) Gold 6134 CPU @ 3.20GHz in andromache
#target_compile_options(utils PUBLIC -march=native -msse4.2 -mavx)
target_compile_options(utils PUBLIC -march=native -msse3 -msse4.2 -mavx -mavx2)