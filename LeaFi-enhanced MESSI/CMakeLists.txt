cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

set(CMAKE_VERBOSE_MAKEFILE ON)

project(leafi_messi CXX C)
set(TARGET leafi_messi)
add_executable(${TARGET} main.cpp)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17) #libtorch
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

find_package(GSL REQUIRED)

include_directories(
        ${GSL_INCLUDE_DIRS}
        ${TORCH_INCLUDE_DIRS}
        src/utils
        src/paa
        src/sax
        src/answers
        src/filters
        src/index
        src/query
)

add_subdirectory(src/utils)
add_subdirectory(src/paa)
add_subdirectory(src/sax)
add_subdirectory(src/answers)
add_subdirectory(src/filters)
add_subdirectory(src/index)
add_subdirectory(src/query)

add_compile_options(-O3)

target_link_libraries(${TARGET} query index filters answers sax paa utils)
target_link_libraries(${TARGET} ${TORCH_LIBRARIES} ${GSL_LIBRARIES} pthread m)

install(TARGETS ${TARGET} DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/bin)
