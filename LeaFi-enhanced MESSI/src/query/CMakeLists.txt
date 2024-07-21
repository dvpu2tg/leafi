cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(query SHARED query_engine_wfilter.cpp query_engine.c query.c)
target_link_libraries(query PUBLIC index answers sax paa utils pthread m)
