cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

add_library(paa SHARED paa.c)
target_link_libraries(paa PUBLIC utils pthread m)
