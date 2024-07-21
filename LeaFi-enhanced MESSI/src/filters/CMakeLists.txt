cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(filters SHARED allocator.cpp neural_filter.cpp conformal_adjustor.cpp lr_scheduler.cpp series_dataset.cpp
        static_variables.cpp)
target_link_libraries(filters utils ${TORCH_LIBRARIES} ${GSL_LIBRARIES})
