cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(index SHARED index_engine.c index.c index_commons.c node.c)
target_link_libraries(index PUBLIC answers sax paa utils pthread m ${GSL_LIBRARIES})
