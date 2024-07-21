/******************************************************************************
 * Copyright (c) 2024, the LeaFi author(s).
 ******************************************************************************/

#ifndef ISAX_ANSWER_H
#define ISAX_ANSWER_H

#include <stdlib.h>
#include <pthread.h>

#include "globals.h"
#include "config.h"
#include "clog.h"


typedef struct Answer {
    pthread_rwlock_t *lock;

    VALUE_TYPE *distances; // max-heap
    ID_TYPE *ids; // auxiliary max-heap

    unsigned int size;
    unsigned int k;
} Answer;


Answer *initializeAnswer(Config const *config);

void resetAnswer(Answer *answer);
void resetAnswerBy(Answer *answer, VALUE_TYPE initial_bsf_distance);

VALUE_TYPE getBSF(Answer * answer);
int checkBSF(Answer *answer, VALUE_TYPE distance);

void updateBSF(Answer *answer, VALUE_TYPE distance);
int checkNUpdateBSF(Answer * answer, VALUE_TYPE distance);

void updateBSFWithID(Answer *answer, VALUE_TYPE distance, ID_TYPE id);
int checkNUpdateBSFWithID(Answer * answer, VALUE_TYPE distance, ID_TYPE id);

void logAnswer(unsigned int query_id, Answer *answer);

void freeAnswer(Answer *answer);

#endif //ISAX_ANSWER_H
