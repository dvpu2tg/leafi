/******************************************************************************
 * Copyright (c) 2024, the LeaFi author(s).
 ******************************************************************************/

#ifndef ISAX_GLOBALS_H
#define ISAX_GLOBALS_H

#include <time.h>
#include <float.h>
#include <stdlib.h>
#include <pthread.h>
#include <immintrin.h>


//#define DEBUG

#define FINE_TIMING
#define TIMING

#ifdef FINE_TIMING
#ifndef TIMING
#define TIMING
#endif
#endif

#ifdef TIMING

#define CLK_ID CLOCK_MONOTONIC
#define NSEC_INSEC 1000000000L

extern int clock_code;

typedef struct TimeDiff {
    long tv_nsec;
    long tv_sec;
} TimeDiff;

inline void getTimeDiff(TimeDiff *t_diff, struct timespec t_start, struct timespec t_stop) {
    t_diff->tv_nsec = t_stop.tv_nsec - t_start.tv_nsec;
    t_diff->tv_sec = t_stop.tv_sec - t_start.tv_sec;
    if (t_diff->tv_nsec < 0) {
        t_diff->tv_sec -= 1;
        t_diff->tv_nsec += NSEC_INSEC;
    }
}

#endif


#define CLOGGER_ID 0


#define SAX_SIMD_ALIGNED_LENGTH 16

#define VALUE_MAX (1e7)
#define VALUE_MAX_TH (1e5)
#define VALUE_MIN (-1e7)
#define VALUE_MIN_TH (-1e5)
#define VALUE_MARGIN (1e-5)
#define VALUE_EPSILON (1e-7)

typedef float VALUE_TYPE;
// TODO only supports sax_cardinality <= 8
typedef unsigned char SAXSymbol;
typedef unsigned int SAXMask;
typedef ssize_t ID_TYPE;


#define FINE_PROFILING
#define ISAX_PROFILING

#ifdef FINE_PROFILING
#ifndef ISAX_PROFILING
#define PROFILING
#endif
#endif

#ifdef ISAX_PROFILING

extern ID_TYPE query_id_profiling;
extern ID_TYPE leaf_counter_profiling;
extern ID_TYPE sum2sax_counter_profiling;
extern ID_TYPE l2square_counter_profiling;

//extern ID node_nn_cnt;

extern ID_TYPE neural_pruned_leaves;
extern ID_TYPE neural_pruned_series;
extern ID_TYPE neural_passed_leaves;
extern ID_TYPE neural_passed_series;
extern ID_TYPE neural_missed_leaves;
extern ID_TYPE neural_missed_series;

extern pthread_mutex_t *log_lock_profiling;

//extern unsigned int neural_filter_counter;
extern unsigned int num_filters_learned;

extern int log_model_size;

extern size_t free_gpu_size;
extern size_t total_gpu_size;

#endif


extern __m256i M256I_1;
extern __m256i M256I_BREAKPOINTS_OFFSETS_0_7, M256I_BREAKPOINTS_OFFSETS_8_15;
extern __m256i M256I_BREAKPOINTS8_OFFSETS_0_7, M256I_BREAKPOINTS8_OFFSETS_8_15;
extern __m256i *M256I_OFFSETS_BY_SEGMENTS;


extern SAXMask root_mask;

#define VALUE_L(left, right) ((right) - (left) > VALUE_EPSILON)
#define VALUE_G(left, right) ((left) - (right) > VALUE_EPSILON)
#define VALUE_LEQ(left, right) ((left) - (right) <= VALUE_EPSILON)
#define VALUE_GEQ(left, right) ((right) - (left) <= VALUE_EPSILON)
#define VALUE_EQ(left, right) (VALUE_LEQ(left, right) && VALUE_GEQ(left, right))
#define VALUE_NEQ(left, right) (VALUE_L(left, right) || VALUE_G(left, right))

#define SWAP(T, a, b) do { T tmp = a; (a) = b; (b) = tmp; } while (0)

static inline int VALUE_COMPARE(void const *left, void const *right) {
    if (VALUE_L(*(VALUE_TYPE *) left, *(VALUE_TYPE *) right)) {
        return -1;
    }

    if (VALUE_G(*(VALUE_TYPE *) left, *(VALUE_TYPE *) right)) {
        return 1;
    }

    return 0;
}

#define SIMS_MEMORY 0
#define SIMS_SSD 1
#define SIMS_DISK 2

extern VALUE_TYPE *split_range_cache_global;
extern ID_TYPE *split_segment_cache_global;

typedef double ERROR_TYPE;
extern VALUE_TYPE EPSILON_GAP;

extern int global_node_id;

#endif //ISAX_GLOBALS_H
