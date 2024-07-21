/******************************************************************************
 * Copyright (c) 2024, the LeaFi author(s).
 ******************************************************************************/

#include "answer.h"


void heapifyTopDown(VALUE_TYPE *heap, unsigned int parent, unsigned int size) {
    unsigned int left = (parent << 1u) + 1, right = left + 1;

    if (right < size) {
        unsigned int next = left;
        if (VALUE_L(heap[left], heap[right])) {
            next = right;
        }

        if (VALUE_L(heap[parent], heap[next])) {
            SWAP(VALUE_TYPE, heap[parent], heap[next]);

            heapifyTopDown(heap, next, size);
        }
    } else if (left < size && VALUE_L(heap[parent], heap[left])) {
        SWAP(VALUE_TYPE, heap[parent], heap[left]);
    }
}


void heapifyBottomUp(VALUE_TYPE *heap, unsigned int child) {
    if (child != 0) {
        unsigned int parent = (child - 1) >> 1u;

        if (VALUE_L(heap[parent], heap[child])) {
            SWAP(VALUE_TYPE, heap[parent], heap[child]);
        }

        heapifyBottomUp(heap, parent);
    }
}


int checkNUpdateBSF(Answer *answer, VALUE_TYPE distance) {
    if (answer->size < answer->k) {
        answer->distances[answer->size] = distance;
        heapifyBottomUp(answer->distances, answer->size);

        answer->size += 1;
    } else if (VALUE_L(distance, answer->distances[0])) {
        answer->distances[0] = distance;
        heapifyTopDown(answer->distances, 0, answer->size);
    } else {
        return 1;
    }

#ifdef ISAX_PROFILING
    pthread_mutex_lock(log_lock_profiling);
    clog_info(CLOG(CLOGGER_ID), "query %d - updated BSF = %f at %d l2square / %d sum2sax / %d entered",
              query_id_profiling, distance, l2square_counter_profiling, sum2sax_counter_profiling,
              leaf_counter_profiling);
    pthread_mutex_unlock(log_lock_profiling);
#endif

    return 0;
}


void heapifyTopDownWithID(VALUE_TYPE *heap, ID_TYPE *ids, unsigned int parent, unsigned int size) {
    unsigned int left = (parent << 1u) + 1, right = left + 1;

    if (right < size) {
        unsigned int next = left;
        if (VALUE_L(heap[left], heap[right])) {
            next = right;
        }

        if (VALUE_L(heap[parent], heap[next])) {
            SWAP(VALUE_TYPE, heap[parent], heap[next]);
            SWAP(ID_TYPE, ids[parent], ids[next]);

            heapifyTopDownWithID(heap, ids, next, size);
        }
    } else if (left < size && VALUE_L(heap[parent], heap[left])) {
        SWAP(VALUE_TYPE, heap[parent], heap[left]);
        SWAP(ID_TYPE, ids[parent], ids[left]);
    }
}


void heapifyBottomUpWithID(VALUE_TYPE *heap, ID_TYPE *ids, unsigned int child) {
    if (child != 0) {
        unsigned int parent = (child - 1) >> 1u;

        if (VALUE_L(heap[parent], heap[child])) {
            SWAP(VALUE_TYPE, heap[parent], heap[child]);
            SWAP(ID_TYPE, ids[parent], ids[child]);

            heapifyBottomUpWithID(heap, ids, parent);
        }
    }
}


int checkBSF(Answer *answer, VALUE_TYPE distance) {
    return answer->size < answer->k || VALUE_L(distance, answer->distances[0]);
}


void updateBSF(Answer *answer, VALUE_TYPE distance) {
    if (answer->size < answer->k) {
        answer->distances[answer->size] = distance;

        heapifyBottomUpWithID(answer->distances, answer->ids, answer->size);

        answer->size += 1;
    } else {
        answer->distances[0] = distance;

        heapifyTopDownWithID(answer->distances, answer->ids, 0, answer->size);
    }

#ifdef ISAX_PROFILING
    pthread_mutex_lock(log_lock_profiling);
    clog_info(CLOG(CLOGGER_ID), "query %d - updated BSF = %f after %d l2square / %d sum2sax / %d entered",
              query_id_profiling, distance, l2square_counter_profiling, sum2sax_counter_profiling, leaf_counter_profiling);
    pthread_mutex_unlock(log_lock_profiling);
#endif
}


void updateBSFWithID(Answer *answer, VALUE_TYPE distance, ID_TYPE id) {
    if (answer->size < answer->k) {
        answer->distances[answer->size] = distance;
        answer->ids[answer->size] = id;

        heapifyBottomUpWithID(answer->distances, answer->ids, answer->size);

        answer->size += 1;
    } else {
        answer->distances[0] = distance;
        answer->ids[0] = id;

        heapifyTopDownWithID(answer->distances, answer->ids, 0, answer->size);
    }

#ifdef ISAX_PROFILING
    pthread_mutex_lock(log_lock_profiling);
    clog_info(CLOG(CLOGGER_ID), "query %d - updated BSF = %f by %d after %d l2square / %d sum2sax / %d entered",
              query_id_profiling, distance, id, l2square_counter_profiling, sum2sax_counter_profiling,
              leaf_counter_profiling);
    pthread_mutex_unlock(log_lock_profiling);
#endif
}


int checkNUpdateBSFWithID(Answer *answer, VALUE_TYPE distance, ID_TYPE id) {
    if (answer->size < answer->k) {
        answer->distances[answer->size] = distance;
        answer->ids[answer->size] = id;

        heapifyBottomUpWithID(answer->distances, answer->ids, answer->size);

        answer->size += 1;
    } else if (VALUE_L(distance, answer->distances[0])) {
        answer->distances[0] = distance;
        answer->ids[0] = id;

        heapifyTopDownWithID(answer->distances, answer->ids, 0, answer->size);
    } else {
        return 1;
    }

    // commented for ideal search
//#ifdef PROFILING
//    pthread_mutex_lock(log_lock_profiling);
//    clog_info(CLOG(CLOGGER_ID), "query %d - updated BSF = %f by %d after %d l2square / %d sum2sax / %d entered",
//              query_id_profiling, distance, id, l2square_counter_profiling, sum2sax_counter_profiling,
//              leaf_counter_profiling);
//    pthread_mutex_unlock(log_lock_profiling);
//#endif

    return 0;
}


Answer *initializeAnswer(Config const *config) {
    Answer *answer = malloc(sizeof(Answer));

    answer->size = 0;
    answer->k = config->k;
    answer->distances = malloc(sizeof(VALUE_TYPE) * config->k);
    answer->distances[0] = VALUE_MAX;

    if (config->with_id) {
        answer->ids = malloc(sizeof(ID_TYPE) * config->k);
    } else {
        answer->ids = NULL;
    }

    answer->lock = malloc(sizeof(pthread_rwlock_t));
    assert(pthread_rwlock_init(answer->lock, NULL) == 0);

    return answer;
}


void resetAnswer(Answer *answer) {
    answer->size = 1;

    answer->distances[0] = VALUE_MAX;

    if (answer->ids != NULL) {
        answer->ids[0] = -1;
    }
}


void resetAnswerBy(Answer *answer, VALUE_TYPE initial_bsf_distance) {
    answer->size = 1;

    answer->distances[0] = initial_bsf_distance;

    if (answer->ids != NULL) {
        answer->ids[0] = -1;
    }
}


void freeAnswer(Answer *answer) {
    free(answer->distances);

    if (answer->ids != NULL) {
        free(answer->ids);
    }

    pthread_rwlock_destroy(answer->lock);
    free(answer->lock);

    free(answer);
}


void logAnswer(unsigned int query_id, Answer *answer) {
//    if (answer->size == 0) {
//        clog_info(CLOG(CLOGGER_ID), "query %d NO closer neighbors than initial %f", query_id, answer->distances[0]);
//    }

    if (answer->ids) {
        for (unsigned int i = 0; i < answer->size; ++i) {
            clog_info(CLOG(CLOGGER_ID), "query %d - %d / %luNN = %f by %d",
                      query_id, i, answer->k, answer->distances[i], answer->ids[i]);
        }
    } else {
        for (unsigned int i = 0; i < answer->size; ++i) {
            clog_info(CLOG(CLOGGER_ID), "query %d - %d / %luNN = %f",
                      query_id, i, answer->k, answer->distances[i]);
        }
    }
}


VALUE_TYPE getBSF(Answer *answer) {
    return answer->distances[0];
}
