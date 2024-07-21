/******************************************************************************
 * Copyright (c) 2024, the LeaFi author(s).
 ******************************************************************************/

#include "index_engine.h"

#include <stdlib.h>

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

#include "index_commons.h"
#include "answer.h"
#include "generator.h"
#include "stats.h"


typedef struct IndexCache {
    Index *index;

    ID_TYPE *shared_start_id;
    unsigned int block_size;

    unsigned int initial_leaf_size;
    unsigned int leaf_size;
    unsigned int leaf_min_split_size;

    bool split_by_summarizations;
    bool split_by_sigma;

    ID_TYPE thread_id;
} IndexCache;


Node *route(Node const *parent, SAXSymbol const *sax, unsigned int num_segments) {
    for (unsigned int i = 0; i < num_segments; ++i) {
        if (parent->right->masks[i] != parent->masks[i]) {
            if (parent->right->masks[i] & sax[i]) {
                return parent->right;
            } else {
                return parent->left;
            }
        }
    }

    char *mask_str = malloc(num_segments + 1);
    clog_error(CLOG(CLOGGER_ID), "node %s failed to find affiliated child node, mask %s",
               parent->sax_str, mask2str(parent->masks, mask_str, num_segments, 8));
    if (parent->left != NULL) {
        clog_error(CLOG(CLOGGER_ID), "left child %s, mask %s",
                   parent->left->sax_str, mask2str(parent->left->masks, mask_str, num_segments, 8));
    }
    if (parent->right != NULL) {
        clog_error(CLOG(CLOGGER_ID), "right child %s, mask %s",
                   parent->right->sax_str, mask2str(parent->right->masks, mask_str, num_segments, 8));
    }
    free(mask_str);

    exit(EXIT_FAILURE);
}


int decideSplitSegmentByNextBit(Index *index, Node *parent, unsigned int num_segments, SAXMask cardinality_checker) {
    int segment_to_split = -1;
    int bsf_difference = (int) parent->size + 1, local_difference;
    SAXMask next_bit;

    for (unsigned int i = 0; i < num_segments; ++i) {
        if (parent->masks[i] ^ 1u && (parent->masks[i] & cardinality_checker) == 0) {
            local_difference = 0;
            next_bit = parent->masks[i] >> 1u;

            for (unsigned int j = 0; j < parent->size; ++j) {
                if (index->saxs[SAX_SIMD_ALIGNED_LENGTH * parent->ids[j] + i] & next_bit) {
                    local_difference += 1;
                } else {
                    local_difference -= 1;
                }
            }

            local_difference = abs(local_difference);
            if (local_difference < bsf_difference) {
                segment_to_split = (int) i;
                bsf_difference = abs(local_difference);
            } else if (local_difference == bsf_difference && parent->masks[i] > parent->masks[segment_to_split]) {
                segment_to_split = (int) i;
            }
        }
    }

    return segment_to_split;
}


int decideSplitSegmentByDistribution(Index *index,
                                     Node *parent,
                                     unsigned int num_segments,
                                     SAXMask cardinality_checker,
                                     ID_TYPE thread_id) {
    int segment_to_split = -1;
    double bsf_range = VALUE_MAX, norm_range, diff, range_mid, mean, std;
    SAXMask next_mask;

    size_t sax_num_bits = sizeof(SAXSymbol) * CHAR_BIT;
    char *sax_str = malloc(sax_num_bits + 1);

    for (unsigned int segment_i = 0; segment_i < num_segments; ++segment_i) {
        if (parent->masks[segment_i] ^ 1u && (parent->masks[segment_i] & cardinality_checker) == 0) {
            next_mask = parent->masks[segment_i] >> 1u;
            mean = 0, std = 0;

            for (unsigned int j = 0; j < parent->size; ++j) {
                mean += index->summarizations[num_segments * parent->ids[j] + segment_i];
            }
            mean /= parent->size;

            for (unsigned int j = 0; j < parent->size; ++j) {
                diff = index->summarizations[num_segments * parent->ids[j] + segment_i] - mean;
                std += diff * diff;
            }
            std = sqrt(std / parent->size);

            range_mid = index->breakpoints[OFFSETS_BY_SEGMENTS[segment_i] + OFFSETS_BY_MASK[next_mask] +
                                           (((unsigned int) parent->sax[segment_i] >> SHIFTS_BY_MASK[next_mask]) | 1u)];
            norm_range = fabs(range_mid - mean) / std;
            split_range_cache_global[segment_i] = norm_range;

            sax_symbol_to_binary_str(parent->sax[segment_i], sax_str);
            clog_debug(CLOG(CLOGGER_ID),
                       "index thread %d node %s - segment %d sax %s mask %d - mean %.4f std %.4f mid %.4f normed %.4f",
                       thread_id, parent->sax_str,
                       segment_i, sax_str, BITS_BY_MASK[parent->masks[segment_i]],
                       mean, std, range_mid, norm_range);

            if (VALUE_L(norm_range, bsf_range)) {
                bsf_range = norm_range;
                segment_to_split = (int) segment_i;
            } else if (VALUE_EQ(norm_range, bsf_range) && parent->masks[segment_i] > parent->masks[segment_to_split]) {
                // prefer the segments with fewer splits
                clog_debug(CLOG(CLOGGER_ID),
                           "index thread %d node %s - segment %d (%d bits used) replaces segment %d (%d bits)",
                           thread_id, parent->sax_str,
                           segment_i, BITS_BY_MASK[parent->masks[segment_i]],
                           segment_to_split, BITS_BY_MASK[parent->masks[segment_to_split]]);

                segment_to_split = (int) segment_i;
            }
        } else {
            split_range_cache_global[segment_i] = VALUE_MAX;
        }
    }

    clog_debug(CLOG(CLOGGER_ID), "index thread %d node %s - split segment %d",
               thread_id, parent->sax_str, segment_to_split);

    free(sax_str);
    return segment_to_split;
}


int decideSplitSegmentBySigma(Index *index, Node *parent, unsigned int num_segments, SAXMask cardinality_checker) {
    int segment_to_split = -1;
    double bsf = VALUE_MIN, diff, mean, std;

    for (unsigned int i = 0; i < num_segments; ++i) {
        if (parent->masks[i] ^ 1u && (parent->masks[i] & cardinality_checker) == 0) {
            mean = 0, std = 0;

            for (unsigned int j = 0; j < parent->size; ++j) {
                mean += index->summarizations[num_segments * parent->ids[j] + i];
            }
            mean /= parent->size;

            for (unsigned int j = 0; j < parent->size; ++j) {
                diff = index->summarizations[num_segments * parent->ids[j] + i] - mean;
                std += diff * diff;
            }
            std = sqrt(std / parent->size);

            if (VALUE_L(bsf, std)) {
                bsf = std;
                segment_to_split = (int) i;
            } else if (VALUE_EQ(bsf, std) && parent->masks[i] > parent->masks[segment_to_split]) {
                segment_to_split = (int) i;
            }
        }
    }

    return segment_to_split;
}


void splitNode(Index *index, Node *parent, unsigned int num_segments, bool split_by_summarizations,
               bool split_by_sigma, SAXMask cardinality_checker, ID_TYPE leaf_min_split_size, ID_TYPE thread_id) {
    int segment_to_split;

    if (split_by_sigma) {
        segment_to_split = decideSplitSegmentBySigma(index, parent, num_segments, cardinality_checker);
    } else if (split_by_summarizations) {
        segment_to_split = decideSplitSegmentByDistribution(index, parent, num_segments, cardinality_checker,
                                                            thread_id);
    } else {
        segment_to_split = decideSplitSegmentByNextBit(index, parent, num_segments, cardinality_checker);
    }

    if (segment_to_split == -1) {
        clog_error(CLOG(CLOGGER_ID), "cannot find segment to split");
        exit(EXIT_FAILURE);
    }

    SAXMask *child_masks = aligned_alloc(256, sizeof(SAXMask) * num_segments);
    memcpy(child_masks, parent->masks, sizeof(SAXMask) * num_segments);
    child_masks[segment_to_split] >>= 1u;

    SAXSymbol *right_sax = aligned_alloc(128, sizeof(SAXSymbol) * SAX_SIMD_ALIGNED_LENGTH);
    memcpy(right_sax, parent->sax, sizeof(SAXSymbol) * num_segments);
    right_sax[segment_to_split] |= child_masks[segment_to_split];

    parent->left = initializeNode(parent->sax, child_masks, index->sax_length, index->sax_cardinality);
    parent->right = initializeNode(right_sax, child_masks, index->sax_length, index->sax_cardinality);

    for (unsigned int series_i = 0; series_i < parent->size; ++series_i) {
        if (index->saxs[SAX_SIMD_ALIGNED_LENGTH * parent->ids[series_i] + segment_to_split] &
            child_masks[segment_to_split]) {
            insertNode(parent->right, parent->ids[series_i], parent->capacity, parent->capacity);
        } else {
            insertNode(parent->left, parent->ids[series_i], parent->capacity, parent->capacity);
        }
    }

    clog_debug(CLOG(CLOGGER_ID), "index - parent %s = %lu, split segment %d, left %s = %lu, right %s = %lu",
               parent->sax_str, parent->size, segment_to_split,
               parent->left->sax_str, parent->left->size,
               parent->right->sax_str, parent->right->size);

    if (parent->left->size < leaf_min_split_size || parent->right->size < leaf_min_split_size) {
        for (ID_TYPE i = 0; i < num_segments; ++i) {
            split_segment_cache_global[i] = i;
        }
        qSortIndicesBy(split_segment_cache_global, split_range_cache_global, 0, num_segments - 1);

        for (ID_TYPE sorted_i = 1; sorted_i < num_segments; ++sorted_i) {
            segment_to_split = (int) split_segment_cache_global[sorted_i];

            freeNode(parent->left, false, false);
            memcpy(child_masks, parent->masks, sizeof(SAXMask) * num_segments);
            child_masks[segment_to_split] >>= 1u;

            freeNode(parent->right, false, false);
            memcpy(right_sax, parent->sax, sizeof(SAXSymbol) * num_segments);
            right_sax[segment_to_split] |= child_masks[segment_to_split];

            parent->left = initializeNode(parent->sax, child_masks, index->sax_length, index->sax_cardinality);
            parent->right = initializeNode(right_sax, child_masks, index->sax_length, index->sax_cardinality);

            for (unsigned int j = 0; j < parent->size; ++j) {
                if (index->saxs[SAX_SIMD_ALIGNED_LENGTH * parent->ids[j] + segment_to_split] &
                    child_masks[segment_to_split]) {
                    insertNode(parent->right, parent->ids[j], parent->capacity, parent->capacity);
                } else {
                    insertNode(parent->left, parent->ids[j], parent->capacity, parent->capacity);
                }
            }

            clog_debug(CLOG(CLOGGER_ID),
                       "index - reverted parent %s = %lu, split segment %d (normed %.4f), left %s = %lu, right %s = %lu",
                       parent->sax_str, parent->size, segment_to_split, split_range_cache_global[sorted_i],
                       parent->left->sax_str, parent->left->size,
                       parent->right->sax_str, parent->right->size);

            if (parent->left->size >= leaf_min_split_size && parent->right->size >= leaf_min_split_size) {
                break;
            }
        }
    }

    cleanNode(parent);
}


void *buildIndexThread(void *cache) {
    IndexCache *index_cache = (IndexCache *) cache;
    Index *index = index_cache->index;

    ID_TYPE database_size = index->database_size, local_start_id, local_stop_id;
    ID_TYPE *shared_start_id = index_cache->shared_start_id;
    unsigned int block_size = index_cache->block_size;

    SAXSymbol const *sax;
    Node *node, *parent;

    char *mask_str = malloc(17);
    ID_TYPE thread_id = index_cache->thread_id;

    while ((local_start_id = __sync_fetch_and_add(shared_start_id, block_size)) < database_size) {
        local_stop_id = local_start_id + block_size;
        if (local_stop_id > database_size) {
            local_stop_id = database_size;
        }

        for (ID_TYPE i = local_start_id; i < local_stop_id; ++i) {
            sax = index->saxs + SAX_SIMD_ALIGNED_LENGTH * i;
            node = index->roots[rootSAX2ID(sax, index->sax_length, 8)];

            pthread_mutex_lock(node->lock);

            while (node->left != NULL || (node->capacity != 0 && node->size == index_cache->leaf_size)) {
                parent = node;

                if (node->size == index_cache->leaf_size) {
                    splitNode(index, parent, index->sax_length, index_cache->split_by_summarizations,
                              index_cache->split_by_sigma, index->cardinality_checker, index_cache->leaf_min_split_size,
                              thread_id);
                }

                node = route(parent, sax, index->sax_length);

                pthread_mutex_lock(node->lock);
                pthread_mutex_unlock(parent->lock);
            }

            insertNode(node, i, index_cache->initial_leaf_size, index_cache->leaf_size);

            pthread_mutex_unlock(node->lock);
        }
    }
    free(mask_str);

    return NULL;
}


void buildIndex(Config const *config, Index *index) {
    unsigned int num_threads = config->max_threads_index;
    unsigned int num_blocks = (unsigned int) ceil((double) config->database_size / (double) config->index_block_size);
    if (num_threads > num_blocks) {
        num_threads = num_blocks;
    }

    pthread_t threads[num_threads];
    IndexCache index_caches[num_threads];
#ifdef FINE_TIMING
    struct timespec start_timestamp, stop_timestamp;
    TimeDiff time_diff;
    clock_code = clock_gettime(CLK_ID, &start_timestamp);
#endif
    ID_TYPE shared_start_id = 0;
    for (unsigned int i = 0; i < num_threads; ++i) {
        index_caches[i].thread_id = i;

        index_caches[i].index = index;
        index_caches[i].leaf_size = config->leaf_size;
        index_caches[i].leaf_min_split_size = config->leaf_min_split_size;

        index_caches[i].initial_leaf_size = config->initial_leaf_size;
        index_caches[i].block_size = config->index_block_size;
        index_caches[i].shared_start_id = &shared_start_id;
        index_caches[i].split_by_summarizations = config->split_by_summarizations;
        index_caches[i].split_by_sigma = config->split_by_sigma;

        pthread_create(&threads[i], NULL, buildIndexThread, (void *) &index_caches[i]);
    }

    for (unsigned int i = 0; i < num_threads; ++i) {
        pthread_join(threads[i], NULL);
    }
#ifdef FINE_TIMING
    clock_code = clock_gettime(CLK_ID, &stop_timestamp);
    getTimeDiff(&time_diff, start_timestamp, stop_timestamp);
    clog_info(CLOG(CLOGGER_ID), "index - build = %ld.%lds", time_diff.tv_sec, time_diff.tv_nsec);
#endif
}

void permuteMemory(VALUE_TYPE *values, VALUE_TYPE *summarizations, SAXSymbol *saxs, ID_TYPE *permutation, ID_TYPE size,
                   unsigned int series_length, unsigned int sax_length) {
    unsigned int series_bytes = sizeof(VALUE_TYPE) * series_length;
    unsigned int summarization_bytes = sizeof(VALUE_TYPE) * sax_length;
    unsigned int sax_bytes = sizeof(SAXSymbol) * SAX_SIMD_ALIGNED_LENGTH;

    VALUE_TYPE *values_cache = aligned_alloc(256, series_bytes);
    VALUE_TYPE *summarization_cache = aligned_alloc(256, summarization_bytes);
    SAXSymbol *sax_cache = aligned_alloc(128, sax_bytes);

    ID_TYPE tmp;
    for (ID_TYPE next, i = 0; i < size; ++i) {
        next = i;

        while (permutation[next] >= 0) {
            memcpy(values_cache, values + series_length * i, series_bytes);
            memcpy(values + series_length * i, values + series_length * permutation[next], series_bytes);
            memcpy(values + series_length * permutation[next], values_cache, series_bytes);

            if (summarizations != NULL) {
                memcpy(summarization_cache, summarizations + sax_length * i, summarization_bytes);
                memcpy(summarizations + sax_length * i, summarizations + sax_length * permutation[next],
                       summarization_bytes);
                memcpy(summarizations + sax_length * permutation[next], summarization_cache, summarization_bytes);
            }

            memcpy(sax_cache, saxs + SAX_SIMD_ALIGNED_LENGTH * i, sax_bytes);
            memcpy(saxs + SAX_SIMD_ALIGNED_LENGTH * i, saxs + SAX_SIMD_ALIGNED_LENGTH * permutation[next], sax_bytes);
            memcpy(saxs + SAX_SIMD_ALIGNED_LENGTH * permutation[next], sax_cache, sax_bytes);

            tmp = permutation[next];
            permutation[next] -= size;
            next = tmp;
        }
    }

    free(values_cache);
    free(summarization_cache);
    free(sax_cache);
}


void squeezeNode(Node *node, Index *index, bool *segment_flags) {
    if (node->left != NULL) {
        squeezeNode(node->left, index, segment_flags);
        squeezeNode(node->right, index, segment_flags);
    } else {
        memcpy(node->sax, index->saxs + SAX_SIMD_ALIGNED_LENGTH * node->start_id,
               sizeof(SAXSymbol) * SAX_SIMD_ALIGNED_LENGTH);

        node->squeezed_masks = aligned_alloc(256, sizeof(SAXMask) * index->sax_length);
        for (unsigned int i = 0; i < index->sax_length; ++i) {
            node->squeezed_masks[i] = 1u;
        }

        if (node->size > 1) {
            int segments_nonsqueezable = 0;
            for (unsigned int i = 0; i < index->sax_length; ++i) {
                if (node->masks[i] & node->squeezed_masks[i]) {
                    segments_nonsqueezable += 1;
                    segment_flags[i] = false;
                } else {
                    segment_flags[i] = true;
                }
            }

            for (ID_TYPE i = SAX_SIMD_ALIGNED_LENGTH * (node->start_id + 1);
                 i < SAX_SIMD_ALIGNED_LENGTH * (node->start_id + node->size) &&
                 segments_nonsqueezable < index->sax_length;
                 i += SAX_SIMD_ALIGNED_LENGTH) {
                for (unsigned j = 0; j < index->sax_length; ++j) {
                    if (segment_flags[j]) {
                        for (unsigned int k = BITS_BY_MASK[node->squeezed_masks[j]];
                             k > BITS_BY_MASK[node->masks[j]];
                             --k) {
                            if (((unsigned) index->saxs[i + j] ^ (unsigned) node->sax[j]) & MASKS_BY_BITS[k]) {
                                node->squeezed_masks[j] = MASKS_BY_BITS[k - 1];

                                if (node->squeezed_masks[j] & node->masks[j]) {
                                    segment_flags[j] = false;
                                    segments_nonsqueezable += 1;
                                }
                            }
                        }
                    }
                }
            }
        }

#ifdef FINE_PROFILING
        VALUE_TYPE const *squeezed_breakpoint, *original_breakpoint;
        for (unsigned int i = 0; i < index->sax_length; ++i) {
            if (node->squeezed_masks[i] ^ node->masks[i]) {
                squeezed_breakpoint =
                        index->breakpoints + OFFSETS_BY_SEGMENTS[i] + OFFSETS_BY_MASK[node->squeezed_masks[i]] +
                        ((unsigned int) node->sax[i] >> SHIFTS_BY_MASK[node->squeezed_masks[i]]);
                original_breakpoint = index->breakpoints + OFFSETS_BY_SEGMENTS[i] + OFFSETS_BY_MASK[node->masks[i]] +
                                      ((unsigned int) node->sax[i] >> SHIFTS_BY_MASK[node->masks[i]]);

                clog_info(CLOG(CLOGGER_ID), "index - segment %d (node.size %d) squeezed %d -> %d (%f -> %f, %f -> %f)",
                          i, node->size, BITS_BY_MASK[node->masks[i]], BITS_BY_MASK[node->squeezed_masks[i]],
                          *original_breakpoint, *squeezed_breakpoint,
                          *(original_breakpoint + 1), *(squeezed_breakpoint + 1));

#ifdef DEBUG
                for (ID j = SAX_SIMD_ALIGNED_LENGTH * (node->start_id + 1);
                     j < SAX_SIMD_ALIGNED_LENGTH * (node->start_id + node->size);
                     j += SAX_SIMD_ALIGNED_LENGTH) {
                    if (((unsigned) index->saxs[j + i] ^ (unsigned) node->sax[i]) &
                        PREFIX_MASKS_BY_MASK[node->squeezed_masks[i]]) {
                        clog_error(CLOG(CLOGGER_ID),
                                   "index - segment %d of series %d unbounded %s (!= %s), masked %d (<- %d)",
                                   i, j - node->start_id,
                                   char2bin(index->saxs[j + i]), char2bin(node->sax[i]),
                                   BITS_BY_MASK[node->squeezed_masks[i]], BITS_BY_MASK[node->masks[i]]);
                    }
                }
#endif
            }
        }
#endif
    }
}


void peelNode(Node *node, Index *index) {
    if (node->left != NULL) {
        peelNode(node->left, index);
        peelNode(node->right, index);
    } else {
        node->upper_envelops = aligned_alloc(256, sizeof(VALUE_TYPE) * index->sax_length);
        node->lower_envelops = aligned_alloc(256, sizeof(VALUE_TYPE) * index->sax_length);

        memcpy(node->upper_envelops, index->summarizations + index->sax_length * node->start_id,
               sizeof(VALUE_TYPE) * index->sax_length);
        memcpy(node->lower_envelops, index->summarizations + index->sax_length * node->start_id,
               sizeof(VALUE_TYPE) * index->sax_length);

        for (VALUE_TYPE const *pt_summarizations = index->summarizations + index->sax_length * (node->start_id + 1);
             pt_summarizations < index->summarizations + index->sax_length * (node->start_id + node->size);
             pt_summarizations += index->sax_length) {
            for (unsigned int j = 0; j < index->sax_length; ++j) {
                if (*(pt_summarizations + j) > node->upper_envelops[j]) {
                    node->upper_envelops[j] = *(pt_summarizations + j);
                }

                if (*(pt_summarizations + j) < node->lower_envelops[j]) {
                    node->lower_envelops[j] = *(pt_summarizations + j);
                }
            }
        }

#ifdef FINE_PROFILING
        SAXMask *masks = node->masks;

        for (unsigned int i = 0; i < index->sax_length; ++i) {
            VALUE_TYPE const *breakpoint = index->breakpoints + OFFSETS_BY_SEGMENTS[i] + OFFSETS_BY_MASK[masks[i]] +
                                           ((unsigned int) node->sax[i] >> SHIFTS_BY_MASK[masks[i]]);

            clog_info(CLOG(CLOGGER_ID), "index - segment %d (node.size %d) peeled by %f -> %f, %f -> %f",
                      i, node->size,
                      *breakpoint, node->lower_envelops[i],
                      *(breakpoint + 1), node->upper_envelops[i]);
        }
#endif
    }
}

void fetchPermutation(Node *node, ID_TYPE *permutation, ID_TYPE *counter) {
    if (node->left != NULL) {
        fetchPermutation(node->left, permutation, counter);
        fetchPermutation(node->right, permutation, counter);
    } else {
        node->start_id = (ID_TYPE) *counter;

        for (unsigned int i = 0; i < node->size; ++i) {
            permutation[node->ids[i]] = (ID_TYPE) *counter;
            *counter += 1;
        }

        free(node->ids);
        node->ids = NULL;
    }
}


void finalizeIndex(Config const *config, Index *index, bool free_summarizations) {
#ifdef FINE_TIMING
    struct timespec start_timestamp, stop_timestamp;
    TimeDiff time_diff;
    clock_code = clock_gettime(CLK_ID, &start_timestamp);
#endif
    ID_TYPE *permutation = aligned_alloc(sizeof(ID_TYPE), sizeof(ID_TYPE) * index->database_size);
    ID_TYPE counter = 0;

    for (unsigned int i = 0; i < index->roots_size; ++i) {
        if (index->roots[i]->size == 0 && index->roots[i]->left == NULL) {
            freeNode(index->roots[i], false, true);
            index->roots[i] = NULL;
        } else {
            fetchPermutation(index->roots[i], permutation, &counter);
        }
    }

    assert(counter == index->database_size);

    if (config->with_id) {
        index->pos2id = aligned_alloc(sizeof(ID_TYPE), sizeof(ID_TYPE) * index->database_size);

        for (unsigned int i = 0; i < index->database_size; ++i) {
            index->pos2id[permutation[i]] = i;
        }
    }

    permuteMemory((VALUE_TYPE *) index->values, (VALUE_TYPE *) index->summarizations, (SAXSymbol *) index->saxs,
                  permutation,
                  (ID_TYPE) index->database_size, index->series_length, index->sax_length);

#ifdef FINE_TIMING
    clock_code = clock_gettime(CLK_ID, &stop_timestamp);
    getTimeDiff(&time_diff, start_timestamp, stop_timestamp);
    clog_info(CLOG(CLOGGER_ID), "index - permute for memory locality = %ld.%lds", time_diff.tv_sec, time_diff.tv_nsec);
#endif

    free(permutation);

    if (free_summarizations && index->summarizations != NULL) {
        free((VALUE_TYPE *) index->summarizations);
        index->summarizations = NULL;
    }
}


void fetchLeaves(Index *index) {
#ifdef FINE_TIMING
    struct timespec start_timestamp, stop_timestamp;
    TimeDiff time_diff;
    clock_code = clock_gettime(CLK_ID, &start_timestamp);
#endif
    index->leaves = malloc(sizeof(Node *) * index->num_leaves);

    unsigned int num_leaves = 0;
    for (unsigned int i = 0; i < index->roots_size; ++i) {
        enqueueLeaf(index->roots[i], index->leaves, &num_leaves, NULL);
    }
    assert(num_leaves == index->num_leaves);
#ifdef FINE_TIMING
    clock_code = clock_gettime(CLK_ID, &stop_timestamp);
    getTimeDiff(&time_diff, start_timestamp, stop_timestamp);
    clog_info(CLOG(CLOGGER_ID), "train - fetch leaves = %ld.%lds", time_diff.tv_sec, time_diff.tv_nsec);
#endif
}


void generateFilterGlobalQueries(Config *config, Index *index, ID_TYPE num_active_filters) {
    unsigned int num_leaves = index->num_leaves;
    if (index->leaves == NULL) {
        fetchLeaves(index);
    }
    Node **leaves = index->leaves;

#ifdef FINE_TIMING
    struct timespec start_timestamp, stop_timestamp;
    TimeDiff time_diff;
    clock_code = clock_gettime(CLK_ID, &start_timestamp);
#endif
    ID_TYPE *filter_i_leaf_i_map = malloc(sizeof(ID_TYPE) * num_active_filters);
    ID_TYPE *leaf_cumulative_sizes = malloc(sizeof(ID_TYPE) * num_active_filters);

    for (ID_TYPE leaf_i = 0, filter_pos = 0; leaf_i < num_leaves; ++leaf_i) {
        if (leaves[leaf_i]->filter != NULL && leaves[leaf_i]->filter->is_activated) {
            filter_i_leaf_i_map[filter_pos] = leaf_i;
            leaf_cumulative_sizes[filter_pos] = leaves[leaf_i]->size;

            if (filter_pos > 0) {
                leaf_cumulative_sizes[filter_pos] += leaf_cumulative_sizes[filter_pos - 1];
            }

            filter_pos += 1;
            if (filter_pos == num_active_filters) {
                clog_debug(CLOG(CLOGGER_ID), "train - all filters were enqueued");
                break;
            }
        }
    }

    ID_TYPE num_series_within_filters = leaf_cumulative_sizes[num_active_filters - 1];
    ID_TYPE num_query_global = config->filter_num_synthetic_query_global;

    const gsl_rng_type *T;
    gsl_rng *r;
    gsl_rng_env_setup();
    T = gsl_rng_default;
    r = gsl_rng_alloc(T);
    gsl_rng_set(r, (unsigned long) time(NULL));

    int totalLength = (int) config->series_length * 32;
    char *combinedString = malloc(totalLength + config->series_length);

    VALUE_TYPE *filter_query_global = aligned_alloc(256, sizeof(VALUE_TYPE) * config->series_length * num_query_global);
    VALUE_TYPE *filter_summarizations_global = aligned_alloc(256, sizeof(VALUE_TYPE) * config->sax_length *
                                                                  num_query_global);

    for (ID_TYPE query_i = 0; query_i < num_query_global; ++query_i) {
        VALUE_TYPE *current_series_to_generate = filter_query_global + config->series_length * query_i;

        ID_TYPE random_i = (ID_TYPE) gsl_rng_uniform_int(r, num_series_within_filters);
        ID_TYPE filter_i = bSearchFloorID(random_i, leaf_cumulative_sizes, 0, num_active_filters - 1);

        ID_TYPE leaf_i = filter_i_leaf_i_map[filter_i];
        ID_TYPE series_i = leaves[leaf_i]->start_id + gsl_rng_uniform_int(r, leaves[leaf_i]->size);
        VALUE_TYPE const *series = index->values + config->series_length * series_i;

        VALUE_TYPE noise_level = config->filter_synthetic_query_min_noise_level +
                                 gsl_rng_uniform(r) * (config->filter_synthetic_query_max_noise_level -
                                                       config->filter_synthetic_query_min_noise_level);
        for (ID_TYPE value_i = 0; value_i < config->series_length; ++value_i) {
            current_series_to_generate[value_i] = series[value_i] + (VALUE_TYPE) gsl_ran_gaussian(r, noise_level);
        }

        if (znormalizeInPlace(current_series_to_generate, config->series_length) != 0) {
            clog_error(CLOG(CLOGGER_ID),
                       "train - adding %.3f noise to series %d from filter (node) %d broke; regenerate",
                       noise_level, series_i - leaves[leaf_i]->start_id, leaves[leaf_i]->filter->id);
            query_i -= 1;
        } else {
            VALUE_TYPE *current_summarization = filter_summarizations_global + config->sax_length * query_i;
            piecewiseAggregate(current_series_to_generate, 1, config->series_length, current_summarization,
                               config->sax_length);

            if (testSeriesInNodeEnvelope(leaves[leaf_i], current_summarization, config->sax_length,
                                         index->breakpoints)) {
                clog_error(CLOG(CLOGGER_ID),
                           "train - adding %.3f noise to series %d failed to escape filter (node) %d; regenerate",
                           noise_level, series_i - leaves[leaf_i]->start_id, leaves[leaf_i]->filter->id);
                query_i -= 1;
            } else {
                clog_info(CLOG(CLOGGER_ID), "train - add %.3f noise to series %d from filter (node) %d",
                          noise_level, series_i - leaves[leaf_i]->start_id, leaves[leaf_i]->filter->id);
            }
        }
    }

    gsl_rng_free(r);
    free(combinedString);

    config->filter_query_load_size = num_query_global;
    index->filter_global_queries = (VALUE_TYPE const *) filter_query_global;
#ifdef FINE_TIMING
    clock_code = clock_gettime(CLK_ID, &stop_timestamp);
    getTimeDiff(&time_diff, start_timestamp, stop_timestamp);
    clog_info(CLOG(CLOGGER_ID), "train - generated %d global synthetic queries in %ld.%lds",
              num_query_global, time_diff.tv_sec, time_diff.tv_nsec);
#endif
    FILE *file_wb = fopen(config->filter_global_queries_filepath, "wb");
    if (file_wb == NULL) {
        clog_error(CLOG(CLOGGER_ID), "train - synthetic query output file %s failed to open",
                   config->filter_global_queries_filepath);
        exit(-1);
    }

    size_t num_write_values = fwrite(filter_query_global,
                                     sizeof(VALUE_TYPE), config->series_length * num_query_global,
                                     file_wb);
    fclose(file_wb);
    if (num_write_values != config->series_length * config->filter_query_load_size) {
        clog_error(CLOG(CLOGGER_ID), "train - %.3f/%.3fMB materialized",
                   num_write_values * sizeof(VALUE_TYPE) / 1024 / 1024,
                   (config->series_length * config->filter_query_load_size * sizeof(VALUE_TYPE)) / 1024 / 1024);
        exit(-1);
    }

    index->filter_global_query_summarizations = (VALUE_TYPE const *) filter_summarizations_global;

    SAXSymbol *filter_train_saxs = aligned_alloc(128, sizeof(SAXSymbol) * SAX_SIMD_ALIGNED_LENGTH *
                                                      config->filter_query_load_size);
    summarizations2SAX16(filter_train_saxs, filter_summarizations_global, index->breakpoints,
                         config->filter_query_load_size, index->sax_length,
                         index->sax_cardinality, config->max_threads_index);
    index->filter_global_query_saxs = (SAXSymbol const *) filter_train_saxs;

    free(leaf_cumulative_sizes);
    free(filter_i_leaf_i_map);
}


void filterQueryNode(Answer *answer, Node const *node, VALUE_TYPE const *values, unsigned int series_length,
                     SAXSymbol const *saxs, unsigned int sax_length, VALUE_TYPE const *breakpoints,
                     VALUE_TYPE scale_factor,
                     VALUE_TYPE const *query_values, VALUE_TYPE const *query_summarization,
                     VALUE_TYPE *m256_fetched_cache,
                     unsigned int train_query_id) {
    VALUE_TYPE local_l2SquareSAX8, local_l2Square, local_bsf = VALUE_MAX;
    node->filter->global_bsf_distances[train_query_id] = local_bsf;

    VALUE_TYPE const *start_value_ptr = NULL, *current_series;
    if (values) {
        start_value_ptr = values + series_length * node->start_id;
    } else if (node->values != NULL) {
        start_value_ptr = node->values;
    }

    SAXSymbol const *outer_current_sax = NULL, *current_sax;
    if (saxs) {
        outer_current_sax = saxs + SAX_SIMD_ALIGNED_LENGTH * node->start_id;
    } else if (node->saxs != NULL) {
        outer_current_sax = node->saxs;
    }

    SAXSymbol *saxs2load = NULL;
    VALUE_TYPE *values2load = NULL;
    if (start_value_ptr == NULL && outer_current_sax == NULL) {
        saxs2load = aligned_alloc(128, sizeof(SAXSymbol) * SAX_SIMD_ALIGNED_LENGTH * node->size);
        values2load = aligned_alloc(256, sizeof(VALUE_TYPE) * series_length * node->size);

        FILE *data_file = fopen(node->data_load_filepath, "rb");

        size_t nitems = fread(saxs2load, sizeof(SAXSymbol), SAX_SIMD_ALIGNED_LENGTH * node->size, data_file);
        assert(nitems == SAX_SIMD_ALIGNED_LENGTH * node->size);

        nitems = fread(values2load, sizeof(VALUE_TYPE), series_length * node->size, data_file);
        assert(nitems == series_length * node->size);

        fclose(data_file);

        outer_current_sax = (SAXSymbol const *) saxs2load;
        start_value_ptr = (VALUE_TYPE const *) values2load;
    } else {
        assert(start_value_ptr != NULL & outer_current_sax != NULL);
    }

    for (current_series = start_value_ptr, current_sax = outer_current_sax;
         current_series < start_value_ptr + series_length * node->size;
         current_series += series_length, current_sax += SAX_SIMD_ALIGNED_LENGTH) {
#ifdef ISAX_PROFILING
        sum2sax_counter_profiling += 1;
#endif
        local_l2SquareSAX8 = l2SquareSummarization2SAX8SIMD(sax_length, query_summarization, current_sax,
                                                            breakpoints, scale_factor, m256_fetched_cache);

        // either resident node or closest node, enters anyway
        if (VALUE_G(local_bsf, local_l2SquareSAX8)) {
#ifdef ISAX_PROFILING
            l2square_counter_profiling += 1;
#endif
            local_l2Square = l2SquareEarlySIMD(series_length, query_values, current_series,
                                               local_bsf, m256_fetched_cache);

            if (VALUE_G(local_bsf, local_l2Square)) {
                local_bsf = local_l2Square;
            }
        }

        current_series += series_length;
        current_sax += SAX_SIMD_ALIGNED_LENGTH;
    }

    node->filter->global_nn_distances[train_query_id] = local_bsf;

    if (values2load != NULL) {
        free(values2load);
        values2load = NULL;
    }
    if (saxs2load != NULL) {
        free(saxs2load);
        saxs2load = NULL;
    }

    if (VALUE_G(getBSF(answer), local_bsf)) {
        checkNUpdateBSF(answer, local_bsf);
    } else {
        clog_error(CLOG(CLOGGER_ID), "train query %d - bsf %f not updated at node %s, size %d",
                   query_id_profiling, local_bsf, node->sax_str, node->size);
        exit(-1);
    }
}


typedef struct FilterQueryCache {
    ID_TYPE thread_id; // debug

    Index const *index;
    Config const *config;

    Node const **leaves;
    ID_TYPE *leaf_indices;
    VALUE_TYPE *leaf_distances;
    unsigned int num_leaves;
    Node *resident_node;

    Answer *answer;
    VALUE_TYPE const *query_values;
    VALUE_TYPE const *query_summarization;

    ID_TYPE *shared_leaf_id;
    ID_TYPE leaf_block_size;
    unsigned int query_block_size;

    VALUE_TYPE *m256_fetched_cache;

    SAXSymbol const *saxs;
    VALUE_TYPE const *summarizations;
    VALUE_TYPE const *breakpoints;

    unsigned int train_query_id;

    ID_TYPE *num_active_filters_debug;
} FilterQueryCache;


void *filterCalculateLeafDistanceThread(void *cache) {
    FilterQueryCache *query_cache = (FilterQueryCache *) cache;

    VALUE_TYPE const *breakpoints = NULL;
    if (query_cache->index != NULL) {
        breakpoints = query_cache->index->breakpoints;
    }
    unsigned int sax_length = query_cache->config->sax_length;

    Node *resident_node = query_cache->resident_node;
    VALUE_TYPE *leaf_distances = query_cache->leaf_distances;

    VALUE_TYPE const *query_summarization = query_cache->query_summarization;
    VALUE_TYPE scale_factor = query_cache->config->scale_factor;
    VALUE_TYPE *m256_fetched_cache = query_cache->m256_fetched_cache;

    ID_TYPE block_size = query_cache->leaf_block_size;
    unsigned int num_leaves = query_cache->num_leaves;
    ID_TYPE *shared_leaf_id = query_cache->shared_leaf_id;

    ID_TYPE leaf_id, stop_leaf_id;
    Node const *leaf;

    unsigned int train_query_id = query_cache->train_query_id;

    while ((leaf_id = __sync_fetch_and_add(shared_leaf_id, block_size)) < num_leaves) {
        stop_leaf_id = leaf_id + block_size;
        if (stop_leaf_id > num_leaves) {
            stop_leaf_id = num_leaves;
        }

        for (unsigned int i = leaf_id; i < stop_leaf_id; ++i) {
            leaf = query_cache->leaves[i];

            if (query_cache->index == NULL) {
                breakpoints = ((Index *) leaf->index)->breakpoints;
            }

            if (resident_node != NULL && leaf == resident_node) {
                leaf_distances[i] = VALUE_MAX;

                leaf->filter->global_node_distances[train_query_id] = 0;
            } else {
                if (leaf->upper_envelops != NULL) {
                    leaf_distances[i] = l2SquareValue2EnvelopSIMD(sax_length, query_summarization,
                                                                  leaf->upper_envelops, leaf->lower_envelops,
                                                                  scale_factor, m256_fetched_cache);
                } else if (leaf->squeezed_masks != NULL) {
                    leaf_distances[i] = l2SquareValue2SAXByMaskSIMD(sax_length, query_summarization, leaf->sax,
                                                                    leaf->squeezed_masks, breakpoints, scale_factor,
                                                                    m256_fetched_cache);
                } else {
                    leaf_distances[i] = l2SquareValue2SAXByMaskSIMD(sax_length, query_summarization, leaf->sax,
                                                                    leaf->masks, breakpoints, scale_factor,
                                                                    m256_fetched_cache);
                }

                leaf->filter->global_node_distances[train_query_id] = leaf_distances[i];
            }
        }
    }

    return NULL;
}


void filterQueryNodeThreadCore(Answer *answer, Node const *node, VALUE_TYPE const *values, unsigned int series_length,
                               SAXSymbol const *saxs, unsigned int sax_length, VALUE_TYPE const *breakpoints,
                               VALUE_TYPE scale_factor,
                               VALUE_TYPE const *query_values, VALUE_TYPE const *query_summarization,
                               VALUE_TYPE *m256_fetched_cache,
                               pthread_rwlock_t *lock, unsigned int train_query_id, ID_TYPE thread_id) {
    VALUE_TYPE local_bsf = VALUE_MAX, global_bsf = getBSF(answer);

    if (VALUE_G(node->filter->global_bsf_distances[train_query_id], 0)) {
        clog_error(CLOG(CLOGGER_ID), "train query %d erroneous node %s filter %d - nn %f, bsf %f",
                   train_query_id, node->sax_str, node->filter->id,
                   node->filter->global_nn_distances[train_query_id],
                   node->filter->global_bsf_distances[train_query_id]);
        assert(VALUE_G(node->filter->global_nn_distances[train_query_id], 0));
        return;
    }

    node->filter->global_bsf_distances[train_query_id] = global_bsf;

    VALUE_TYPE const *start_value_ptr = NULL, *current_series;
    if (values) {
        start_value_ptr = values + series_length * node->start_id;
    } else if (node->values != NULL) {
        start_value_ptr = node->values;
    }

    SAXSymbol const *outer_current_sax = NULL, *current_sax;
    if (saxs) {
        outer_current_sax = saxs + SAX_SIMD_ALIGNED_LENGTH * node->start_id;
    } else if (node->saxs != NULL) {
        outer_current_sax = node->saxs;
    }

    SAXSymbol *saxs2load = NULL;
    VALUE_TYPE *values2load = NULL;
    if (start_value_ptr == NULL && outer_current_sax == NULL) {
        saxs2load = aligned_alloc(128, sizeof(SAXSymbol) * SAX_SIMD_ALIGNED_LENGTH * node->size);
        values2load = aligned_alloc(256, sizeof(VALUE_TYPE) * series_length * node->size);

        FILE *data_file = fopen(node->data_load_filepath, "rb");

        size_t nitems = fread(saxs2load, sizeof(SAXSymbol), SAX_SIMD_ALIGNED_LENGTH * node->size, data_file);
        assert(nitems == SAX_SIMD_ALIGNED_LENGTH * node->size);

        nitems = fread(values2load, sizeof(VALUE_TYPE), series_length * node->size, data_file);
        assert(nitems == series_length * node->size);

        fclose(data_file);

        outer_current_sax = (SAXSymbol const *) saxs2load;
        start_value_ptr = (VALUE_TYPE const *) values2load;
    } else {
        assert(start_value_ptr != NULL & outer_current_sax != NULL);
    }

    VALUE_TYPE local_l2SquareSAX8, local_l2Square;
    for (current_series = start_value_ptr, current_sax = outer_current_sax;
         current_series < start_value_ptr + series_length * node->size;
         current_series += series_length, current_sax += SAX_SIMD_ALIGNED_LENGTH) {
#ifdef ISAX_PROFILING
        __sync_fetch_and_add(&sum2sax_counter_profiling, 1);
#endif
        local_l2SquareSAX8 = l2SquareSummarization2SAX8SIMD(sax_length, query_summarization, current_sax,
                                                            breakpoints, scale_factor, m256_fetched_cache);

        if (VALUE_G(local_bsf, local_l2SquareSAX8)) {
#ifdef ISAX_PROFILING
            __sync_fetch_and_add(&l2square_counter_profiling, 1);
#endif
            local_l2Square = l2SquareEarlySIMD(series_length, query_values, current_series,
                                               local_bsf, m256_fetched_cache);

            if (VALUE_G(local_bsf, local_l2Square)) {
                local_bsf = local_l2Square;
            }
        }
    }

    node->filter->global_nn_distances[train_query_id] = local_bsf;

    if (values2load != NULL) {
        free(values2load);
        values2load = NULL;
    }
    if (saxs2load != NULL) {
        free(saxs2load);
        saxs2load = NULL;
    }

    if (VALUE_G(global_bsf, local_bsf)) {
        pthread_rwlock_wrlock(lock);
        checkNUpdateBSF(answer, local_bsf);
        pthread_rwlock_unlock(lock);
    }
}


void *filterQueryNodeThread(void *cache) {
    FilterQueryCache *collect_cache = (FilterQueryCache *) cache;

    VALUE_TYPE const *values = NULL;
    SAXSymbol const *saxs = NULL;
    VALUE_TYPE const *breakpoints = NULL;
    ID_TYPE *pos2id = NULL;

    assert(collect_cache->index != NULL);
    breakpoints = collect_cache->index->breakpoints;
    values = collect_cache->index->values;
    saxs = collect_cache->index->saxs;
    pos2id = collect_cache->index->pos2id;

    unsigned int series_length = collect_cache->config->series_length;
    unsigned int sax_length = collect_cache->config->sax_length;

    Node const *const *leaves = collect_cache->leaves;
    VALUE_TYPE *leaf_distances = collect_cache->leaf_distances;
    ID_TYPE *leaf_indices = collect_cache->leaf_indices;

    VALUE_TYPE const *query_summarization = collect_cache->query_summarization;
    VALUE_TYPE const *query_values = collect_cache->query_values;

    Answer *answer = collect_cache->answer;
    pthread_rwlock_t *lock = answer->lock;

    VALUE_TYPE *m256_fetched_cache = collect_cache->m256_fetched_cache;
    VALUE_TYPE scale_factor = collect_cache->config->scale_factor;

    unsigned int block_size = collect_cache->query_block_size;
    unsigned int num_leaves = collect_cache->num_leaves;
    ID_TYPE *shared_index_id = collect_cache->shared_leaf_id;

    bool sort_leaves = collect_cache->config->sort_leaves;
    bool lower_bounding = collect_cache->config->lower_bounding;

    unsigned int series_limitations = collect_cache->config->series_limitations;

    unsigned int train_query_id = collect_cache->train_query_id;
    ID_TYPE thread_id = collect_cache->thread_id;

    unsigned int sorted_leaf_pos, stop_leaf_pos;
    Node const *node2examine, *resident_node = collect_cache->resident_node;
    while ((sorted_leaf_pos = __sync_fetch_and_add(shared_index_id, block_size)) < num_leaves) {
        stop_leaf_pos = sorted_leaf_pos + block_size;
        if (stop_leaf_pos > num_leaves) {
            stop_leaf_pos = num_leaves;
        }

        for (; sorted_leaf_pos < stop_leaf_pos; ++sorted_leaf_pos) {
            node2examine = leaves[leaf_indices[sorted_leaf_pos]];

            if (node2examine != resident_node) {
                filterQueryNodeThreadCore(answer, node2examine,
                                          values, series_length, saxs, sax_length, breakpoints, scale_factor,
                                          query_values, query_summarization, m256_fetched_cache, lock,
                                          train_query_id, thread_id);
#ifdef ISAX_PROFILING
                __sync_fetch_and_add(&leaf_counter_profiling, 1);
#endif
            }
        }
    }

    return NULL;
}


void searchFilterGlobalQueryCore(Config const *config, Index *index) {
    Answer *answer = initializeAnswer(config);

    VALUE_TYPE const *breakpoints = index->breakpoints;
    VALUE_TYPE const *values = index->values;
    SAXSymbol const *saxs = index->saxs;
    Node **leaves = index->leaves;

    unsigned int series_length = config->series_length;
    unsigned int sax_length = config->sax_length;
    VALUE_TYPE scale_factor = config->scale_factor;

    ID_TYPE shared_leaf_id = 0;
    unsigned int max_threads_index = config->max_threads_index;
    FilterQueryCache collect_caches[max_threads_index];

#ifdef FINE_TIMING
    struct timespec start_timestamp, stop_timestamp;
    TimeDiff time_diff;
#endif

    unsigned int num_leaves = index->num_leaves;
    ID_TYPE *leaf_indices = malloc(sizeof(ID_TYPE) * num_leaves);
    for (ID_TYPE j = 0; j < num_leaves; ++j) {
        leaf_indices[j] = j;
    }

    VALUE_TYPE *leaf_distances = malloc(sizeof(VALUE_TYPE) * num_leaves);
    ID_TYPE leaf_block_size = 1 + num_leaves / (max_threads_index << 1u);
    unsigned int query_block_size = 2 + num_leaves / (max_threads_index << 3u);

    for (unsigned int i = 0; i < max_threads_index; ++i) {
        collect_caches[i].thread_id = i;

        collect_caches[i].answer = answer;
        collect_caches[i].index = index;
        collect_caches[i].config = config;

        collect_caches[i].num_leaves = num_leaves;
        collect_caches[i].leaves = (Node const **) leaves;
        collect_caches[i].leaf_indices = leaf_indices;
        collect_caches[i].leaf_distances = leaf_distances;

        collect_caches[i].m256_fetched_cache = aligned_alloc(256, sizeof(VALUE_TYPE) * 8);

        collect_caches[i].shared_leaf_id = &shared_leaf_id;
        collect_caches[i].query_block_size = query_block_size;
    }

    VALUE_TYPE *local_m256_fetched_cache = collect_caches[0].m256_fetched_cache;

    VALUE_TYPE const *query_values, *query_summarization;
    SAXSymbol const *query_sax;
    VALUE_TYPE local_bsf;
    Node *node;

    for (unsigned int train_query_id = 0;
         train_query_id < config->filter_num_synthetic_query_global; ++train_query_id) {
#ifdef ISAX_PROFILING
        query_id_profiling = train_query_id;
        leaf_counter_profiling = 0;
        sum2sax_counter_profiling = 0;
        l2square_counter_profiling = 0;
#endif
        resetAnswer(answer);

        query_values = index->filter_global_queries + series_length * train_query_id;
        query_summarization = index->filter_global_query_summarizations + sax_length * train_query_id;
        query_sax = index->filter_global_query_saxs + SAX_SIMD_ALIGNED_LENGTH * train_query_id;

#ifdef FINE_TIMING
        clock_code = clock_gettime(CLK_ID, &start_timestamp);
#endif
        node = index->roots[rootSAX2ID(query_sax, sax_length, 8)];
        local_bsf = getBSF(answer);

        if (node != NULL) {
            while (node->left != NULL) {
                node = route(node, query_sax, sax_length);
            }
#ifdef ISAX_PROFILING
            leaf_counter_profiling += 1;
#ifdef FINE_PROFILING
            if (config->log_leaf_visits) {
                clog_info(CLOG(CLOGGER_ID), "train query %d - BSF = %f when visit %d node %s",
                          query_id_profiling, local_bsf, leaf_counter_profiling, node->sax_str);
            }
#endif
#endif
            filterQueryNode(answer, node, values, series_length, saxs, sax_length, breakpoints, scale_factor,
                            query_values, query_summarization, local_m256_fetched_cache, train_query_id);

            local_bsf = getBSF(answer);
#ifdef ISAX_PROFILING
            clog_info(CLOG(CLOGGER_ID), "train query %d - %d l2square / %d sum2sax in resident leaf",
                      train_query_id, l2square_counter_profiling, sum2sax_counter_profiling);
#endif
#ifdef FINE_TIMING
            clock_code = clock_gettime(CLK_ID, &stop_timestamp);
            getTimeDiff(&time_diff, start_timestamp, stop_timestamp);
            clog_info(CLOG(CLOGGER_ID), "train query %d - resident-leaf search = %ld.%lds",
                      train_query_id, time_diff.tv_sec, time_diff.tv_nsec);
#endif
        } else {
            clog_info(CLOG(CLOGGER_ID), "train query %d - no resident node", train_query_id);
        }

#ifdef FINE_TIMING
        clock_code = clock_gettime(CLK_ID, &start_timestamp);
#endif
        pthread_t leaves_threads[max_threads_index];
        shared_leaf_id = 0;

        for (unsigned int j = 0; j < max_threads_index; ++j) {
            collect_caches[j].query_values = query_values;
            collect_caches[j].query_summarization = query_summarization;
            collect_caches[j].resident_node = node;
            collect_caches[j].leaf_block_size = leaf_block_size;
            collect_caches[j].train_query_id = train_query_id;

            pthread_create(&leaves_threads[j], NULL, filterCalculateLeafDistanceThread, (void *) &collect_caches[j]);
        }

        for (unsigned int j = 0; j < max_threads_index; ++j) {
            pthread_join(leaves_threads[j], NULL);
        }
#ifdef FINE_TIMING
        clock_code = clock_gettime(CLK_ID, &stop_timestamp);
        getTimeDiff(&time_diff, start_timestamp, stop_timestamp);
        clog_info(CLOG(CLOGGER_ID), "train query %d - calculate leaf distances = %ld.%lds", train_query_id,
                  time_diff.tv_sec, time_diff.tv_nsec);
        clock_code = clock_gettime(CLK_ID, &start_timestamp);
#endif
        qSortIndicesBy(leaf_indices, leaf_distances, 0, (int) (num_leaves - 1));

#ifdef FINE_TIMING
        clock_code = clock_gettime(CLK_ID, &stop_timestamp);
        getTimeDiff(&time_diff, start_timestamp, stop_timestamp);
        clog_info(CLOG(CLOGGER_ID), "train query %d - sort leaves = %ld.%lds", train_query_id,
                  time_diff.tv_sec, time_diff.tv_nsec);
#endif

        if (node == NULL) {
            node = leaves[leaf_indices[0]];
#ifdef ISAX_PROFILING
            leaf_counter_profiling += 1;
#ifdef FINE_PROFILING
            if (config->log_leaf_visits) {
                clog_info(CLOG(CLOGGER_ID), "train query %d - BSF = %f when visit %d node %s",
                          query_id_profiling, local_bsf, leaf_counter_profiling, node->sax_str);
            }
#endif
#endif
#ifdef FINE_TIMING
            clock_code = clock_gettime(CLK_ID, &start_timestamp);
#endif
            filterQueryNode(answer, node, values, series_length, saxs, sax_length, breakpoints, scale_factor,
                            query_values, query_summarization, local_m256_fetched_cache, train_query_id);
            local_bsf = getBSF(answer);

#ifdef ISAX_PROFILING
            clog_info(CLOG(CLOGGER_ID), "train query %d - %d l2square / %d sum2sax in closest leaf",
                      train_query_id, l2square_counter_profiling, sum2sax_counter_profiling);
#endif
#ifdef FINE_TIMING
            clock_code = clock_gettime(CLK_ID, &stop_timestamp);
            getTimeDiff(&time_diff, start_timestamp, stop_timestamp);
            clog_info(CLOG(CLOGGER_ID), "train query %d - closest-leaf search = %ld.%lds",
                      train_query_id, time_diff.tv_sec, time_diff.tv_nsec);
#endif
        }

#ifdef FINE_TIMING
        clock_code = clock_gettime(CLK_ID, &start_timestamp);
#endif
        pthread_t collect_threads[max_threads_index];
        shared_leaf_id = 0;

        for (unsigned int j = 0; j < max_threads_index; ++j) {
            collect_caches[j].resident_node = node;

            pthread_create(&collect_threads[j], NULL, filterQueryNodeThread, (void *) &collect_caches[j]);
        }

        for (unsigned int j = 0; j < max_threads_index; ++j) {
            pthread_join(collect_threads[j], NULL);
        }
#ifdef FINE_TIMING
        clock_code = clock_gettime(CLK_ID, &stop_timestamp);
        getTimeDiff(&time_diff, start_timestamp, stop_timestamp);
        clog_info(CLOG(CLOGGER_ID), "train query %d - exact search = %ld.%lds", train_query_id, time_diff.tv_sec,
                  time_diff.tv_nsec);
#endif
#ifdef ISAX_PROFILING
        clog_info(CLOG(CLOGGER_ID), "train query %d - %d l2square / %d sum2sax / %d entered", train_query_id,
                  l2square_counter_profiling, sum2sax_counter_profiling, leaf_counter_profiling);
#endif
        logAnswer(train_query_id, answer);
    }

    for (unsigned int i = 0; i < max_threads_index; ++i) {
        free(collect_caches[i].m256_fetched_cache);
    }

    freeAnswer(answer);

    free(leaf_distances);
    free(leaf_indices);
}


void searchFilterGlobalQueries(Config const *config, Index *index) {
#ifdef FINE_TIMING
    struct timespec start_timestamp, stop_timestamp;
    TimeDiff time_diff;
    clock_code = clock_gettime(CLK_ID, &start_timestamp);
#endif
    if (index->leaves == NULL) {
        fetchLeaves(index);
    }
    searchFilterGlobalQueryCore(config, index);
#ifdef FINE_TIMING
    clock_code = clock_gettime(CLK_ID, &stop_timestamp);
    getTimeDiff(&time_diff, start_timestamp, stop_timestamp);
    clog_info(CLOG(CLOGGER_ID), "train - search global queries = %ld.%lds", time_diff.tv_sec, time_diff.tv_nsec);
#endif
}


int compareValueF(const void *a, const void *b) {
    // Convert void* to float* and dereference to get the float values
    VALUE_TYPE fa = *(const VALUE_TYPE *) a;
    VALUE_TYPE fb = *(const VALUE_TYPE *) b;
    // Floating-point comparison logic
    if (fa < fb) return -1;
    if (fa > fb) return 1;
    return 0;
}


void *filterGenerateAndSearchNodeThread(void *cache) {
    FilterQueryCache *filter_query_cache = (FilterQueryCache *) cache;

    assert(filter_query_cache->index != NULL);
    VALUE_TYPE const *breakpoints = filter_query_cache->index->breakpoints;
    VALUE_TYPE const *values = filter_query_cache->index->values;

    unsigned int series_length = filter_query_cache->config->series_length;
    unsigned int sax_length = filter_query_cache->config->sax_length;

    ID_TYPE num_global_queries = filter_query_cache->config->filter_num_synthetic_query_global;
    ID_TYPE num_local_queries = filter_query_cache->config->filter_num_synthetic_query_local;
    ID_TYPE num_local_queries_min = filter_query_cache->config->filter_num_synthetic_query_local_min;
    ID_TYPE num_local_queries_max = filter_query_cache->config->filter_num_synthetic_query_local_max;

    bool is_dynamic = num_local_queries < 1;

    VALUE_TYPE *sorted_distances = NULL;
    if (is_dynamic) {
        ID_TYPE max_local_queries =
                num_local_queries_max > num_global_queries ? num_local_queries_max : num_global_queries;
        sorted_distances = (VALUE_TYPE *) malloc(sizeof(VALUE_TYPE) * max_local_queries);
    }

    char const *query_dump_folderpath = filter_query_cache->config->filter_local_queries_folderpath;

    VALUE_TYPE min_noise_level = filter_query_cache->config->filter_synthetic_query_min_noise_level;
    VALUE_TYPE max_noise_level = filter_query_cache->config->filter_synthetic_query_max_noise_level;
    VALUE_TYPE filter_excluding_quantile = filter_query_cache->config->filter_excluding_quantile;

    Node const **leaves = filter_query_cache->leaves;

    VALUE_TYPE *m256_fetched_cache = filter_query_cache->m256_fetched_cache;

    unsigned int block_size = filter_query_cache->query_block_size;
    unsigned int num_leaves = filter_query_cache->num_leaves;
    ID_TYPE *shared_leaf_pos = filter_query_cache->shared_leaf_id;
    ID_TYPE *num_active_filters_debug = filter_query_cache->num_active_filters_debug;

    ID_TYPE thread_id = filter_query_cache->thread_id;

    const gsl_rng_type *T;
    gsl_rng *r;
    // Create a generator chosen by the environment variable GSL_RNG_TYPE
    gsl_rng_env_setup();
    T = gsl_rng_default;
    r = gsl_rng_alloc(T);
    gsl_rng_set(r, (unsigned long) time(NULL) + thread_id);

    VALUE_TYPE *tmp_summarization = aligned_alloc(256, sizeof(VALUE_TYPE) * series_length);

    unsigned int leaf_iter, stop_leaf_pos;
    Node const *node2examine;
    while ((leaf_iter = __sync_fetch_and_add(shared_leaf_pos, block_size)) < num_leaves) {
        stop_leaf_pos = leaf_iter + block_size;
        if (stop_leaf_pos > num_leaves) {
            stop_leaf_pos = num_leaves;
        }

        clog_debug(CLOG(CLOGGER_ID), "train thread %d - leaf %d to %d / %d",
                   thread_id, leaf_iter, stop_leaf_pos, num_leaves);

        while (leaf_iter < stop_leaf_pos) {
            node2examine = leaves[leaf_iter];

            if (node2examine->filter->is_activated) {
                VALUE_TYPE mu = 0, sigma = 0;
                calculateStats(node2examine->filter->global_nn_distances, node2examine->filter->num_global_query,
                               &mu, &sigma);
                VALUE_TYPE max_legal_l2square = mu - sigma;

                if (is_dynamic) {
                    num_local_queries = 0;

                    memcpy(sorted_distances, node2examine->filter->global_nn_distances, num_global_queries);
                    qsort(sorted_distances, num_global_queries, sizeof(VALUE_TYPE), compareValueF);

                    ID_TYPE left_pos = num_global_queries * filter_excluding_quantile;
                    ID_TYPE right_pos = num_global_queries - left_pos;
                    VALUE_TYPE range_majority = sorted_distances[right_pos] - sorted_distances[left_pos];
                    VALUE_TYPE global_density = (right_pos - left_pos) / range_majority;
                    VALUE_TYPE density_margin =
                            global_density * filter_query_cache->config->filter_density_comparable_margin;
                    VALUE_TYPE local_density;
                    VALUE_TYPE exchange_cache;

                    ID_TYPE in_leaf_patience = filter_query_cache->config->filter_synthetic_query_in_leaf_patience;

                    for (ID_TYPE query_i = 0; query_i < num_local_queries_max; ++query_i) {
                        VALUE_TYPE *series_to_generate = node2examine->filter->local_queries + series_length * query_i;

                        ID_TYPE sampled_series_i = node2examine->start_id + gsl_rng_uniform_int(r, node2examine->size);
                        VALUE_TYPE const *sampled_series = values + series_length * sampled_series_i;

                        VALUE_TYPE noise_level =
                                min_noise_level + gsl_rng_uniform(r) * (max_noise_level - min_noise_level);
                        for (ID_TYPE value_i = 0; value_i < series_length; ++value_i) {
                            series_to_generate[value_i] =
                                    sampled_series[value_i] + (VALUE_TYPE) gsl_ran_gaussian(r, noise_level);
                        }

                        if (znormalizeInPlace(series_to_generate, series_length) != 0) {
                            clog_error(CLOG(CLOGGER_ID),
                                       "train thread %d filter %d query %d - broken synthetic series; regenerate",
                                       thread_id, node2examine->filter->id, query_i);
                            query_i -= 1;
                        } else {
                            piecewiseAggregate(series_to_generate, 1, series_length, tmp_summarization, sax_length);
                            bool is_skip = false;

                            if (testSeriesInNodeEnvelope(leaves[leaf_iter], tmp_summarization, sax_length,
                                                         breakpoints)) {
                                if (in_leaf_patience > 0) {
                                    clog_warn(CLOG(CLOGGER_ID),
                                              "train thread %d filter %d query %d - synthetic series falling into the same leaf node; regenerate",
                                              thread_id, node2examine->filter->id, query_i);
                                    is_skip = true;
                                } else {
                                    clog_warn(CLOG(CLOGGER_ID),
                                              "train thread %d filter %d query %d - synthetic series falling into the same leaf node but no patience",
                                              thread_id, node2examine->filter->id, query_i);
                                }

                                in_leaf_patience -= 1;
                            }

                            if (is_skip) {
                                query_i -= 1;
                            } else {
                                VALUE_TYPE local_bsf = VALUE_MAX;
                                VALUE_TYPE local_l2Square;

                                VALUE_TYPE const *start_value_ptr = values + series_length * node2examine->start_id;
                                VALUE_TYPE const *current_series;

                                for (current_series = start_value_ptr;
                                     current_series < start_value_ptr + series_length * node2examine->size;
                                     current_series += series_length) {
                                    local_l2Square = l2SquareEarlySIMD(series_length, series_to_generate,
                                                                       current_series,
                                                                       local_bsf, m256_fetched_cache);

                                    if (VALUE_G(local_bsf, local_l2Square)) {
                                        local_bsf = local_l2Square;
                                    }
                                }

                                *(node2examine->filter->local_nn_distances + query_i) = local_bsf;
                                clog_info(CLOG(CLOGGER_ID),
                                          "train thread %d filter %d (%3.f, %.3f) query %d - add %.3f noise to series %d, nn %.3f",
                                          thread_id, node2examine->filter->id, mu, sigma, query_i, noise_level,
                                          sampled_series_i - node2examine->start_id, local_bsf);

                                sorted_distances[num_local_queries] = local_bsf;
                                for (ID_TYPE sorted_i = num_local_queries;
                                     sorted_i > 0 && sorted_distances[sorted_i] < sorted_distances[sorted_i - 1];
                                     --sorted_i) {
                                    exchange_cache = sorted_distances[sorted_i];
                                    sorted_distances[sorted_i] = sorted_distances[sorted_i - 1];
                                    sorted_distances[sorted_i - 1] = sorted_distances[sorted_i];
                                }

                                num_local_queries += 1;

                                if (num_local_queries > num_local_queries_min) {
                                    left_pos = num_local_queries * filter_excluding_quantile;
                                    right_pos = num_local_queries - left_pos;
                                    range_majority = sorted_distances[right_pos] - sorted_distances[left_pos];
                                    local_density = (right_pos - left_pos) / range_majority;

                                    if (fabs(global_density - local_density) <= density_margin) {
                                        clog_info(CLOG(CLOGGER_ID),
                                                  "train thread %d filter %d - reached local density %.3f to global %.3f +- %.3f with %.d queries",
                                                  thread_id, node2examine->filter->id,
                                                  local_density, global_density, density_margin, num_local_queries);

                                        query_i = num_local_queries_max;
                                    }
                                }
                            }
                        }
                    }
                } else {
                    for (ID_TYPE query_i = 0; query_i < num_local_queries; ++query_i) {
                        VALUE_TYPE *series_to_generate = node2examine->filter->local_queries + series_length * query_i;

                        ID_TYPE sampled_series_i = node2examine->start_id + gsl_rng_uniform_int(r, node2examine->size);
                        VALUE_TYPE const *sampled_series = values + series_length * sampled_series_i;

                        VALUE_TYPE noise_level =
                                min_noise_level + gsl_rng_uniform(r) * (max_noise_level - min_noise_level);
                        for (ID_TYPE value_i = 0; value_i < series_length; ++value_i) {
                            series_to_generate[value_i] =
                                    sampled_series[value_i] + (VALUE_TYPE) gsl_ran_gaussian(r, noise_level);
                        }

                        if (znormalizeInPlace(series_to_generate, series_length) != 0) {
                            clog_error(CLOG(CLOGGER_ID),
                                       "train thread %d filter %d query %d - broken synthetic series; regenerate",
                                       thread_id, node2examine->filter->id, query_i);
                            query_i -= 1;
                        } else {
                            piecewiseAggregate(series_to_generate, 1, series_length, tmp_summarization, sax_length);

                            if (testSeriesInNodeEnvelope(leaves[leaf_iter], tmp_summarization, sax_length,
                                                         breakpoints)) {
                                clog_warn(CLOG(CLOGGER_ID),
                                          "train thread %d filter %d query %d - synthetic series falling into the same leaf node; regenerate",
                                          thread_id, node2examine->filter->id, query_i);
                                query_i -= 1;
                            } else {
                                VALUE_TYPE local_bsf = VALUE_MAX;
                                VALUE_TYPE local_l2Square;

                                VALUE_TYPE const *start_value_ptr = values + series_length * node2examine->start_id;
                                VALUE_TYPE const *current_series;

                                for (current_series = start_value_ptr;
                                     current_series < start_value_ptr + series_length * node2examine->size;
                                     current_series += series_length) {
                                    local_l2Square = l2SquareEarlySIMD(series_length, series_to_generate,
                                                                       current_series,
                                                                       local_bsf, m256_fetched_cache);

                                    if (VALUE_G(local_bsf, local_l2Square)) {
                                        local_bsf = local_l2Square;
                                    }
                                }

                                if (VALUE_G(local_bsf, max_legal_l2square)) {
                                    clog_warn(CLOG(CLOGGER_ID),
                                              "train thread %d filter %d (%.3f, %.3f) query %d - series %d +%.3f noise, nn %.3f > %.3f; regenerate",
                                              thread_id, node2examine->filter->id, mu, sigma, query_i,
                                              sampled_series_i - node2examine->start_id, noise_level,
                                              local_bsf, max_legal_l2square);
                                    query_i -= 1;
                                } else {
                                    *(node2examine->filter->local_nn_distances + query_i) = local_bsf;

                                    clog_info(CLOG(CLOGGER_ID),
                                              "train thread %d filter %d (%.3f, %.3f) query %d - series %d +%.3f noise, nn %.3f < %.3f",
                                              thread_id, node2examine->filter->id, mu, sigma, query_i,
                                              sampled_series_i - node2examine->start_id, noise_level,
                                              local_bsf, max_legal_l2square);
                                }
                            }
                        }
                    }
                }

                char *local_query_dump_filepath = concat(3, query_dump_folderpath, node2examine->sax_str, ".bin");
                FILE *file_wb = fopen(local_query_dump_filepath, "wb");
                if (file_wb == NULL) {
                    clog_error(CLOG(CLOGGER_ID), "train %d filter %d - local query output file %s failed to open",
                               thread_id, node2examine->filter->id, local_query_dump_filepath);
                    exit(-1);
                }

                size_t num_write_values = fwrite(node2examine->filter->local_queries,
                                                 sizeof(VALUE_TYPE), series_length * num_local_queries,
                                                 file_wb);
                fclose(file_wb);
                if (num_write_values != series_length * num_local_queries) {
                    clog_error(CLOG(CLOGGER_ID), "train %d filter %d - %.3f/%.3fMB materialized",
                               thread_id, node2examine->filter->id,
                               num_write_values * sizeof(VALUE_TYPE) / 1024 / 1024,
                               (series_length * num_local_queries * sizeof(VALUE_TYPE)) / 1024 / 1024);
                    exit(-1);
                }

                node2examine->filter->num_local_query = num_local_queries;

                __sync_fetch_and_add(num_active_filters_debug, 1);
            } else {
                if (node2examine->filter->local_queries != NULL) {
                    free(node2examine->filter->local_queries);
                }
                if (node2examine->filter->local_nn_distances != NULL) {
                    free(node2examine->filter->local_nn_distances);
                }

                node2examine->filter->num_local_query = 0;
            }

            leaf_iter += 1;
        }
    }

    gsl_rng_free(r);
    free(tmp_summarization);
    free(sorted_distances);

    pthread_exit(NULL);
}


void generateAndSearchLocalQueryCore(Config const *config, Index *index, ID_TYPE num_active_filters) {
    unsigned int max_threads_index = config->max_threads_index;
    FilterQueryCache filter_query_caches[max_threads_index];
    pthread_t generate_and_search_threads[max_threads_index];

    ID_TYPE shared_leaf_id = 0, num_active_filters_debug = 0;
    ID_TYPE leaf_block_size = (index->num_leaves + max_threads_index - 1) / (max_threads_index << 1);
    clog_debug(CLOG(CLOGGER_ID), "train - local generation leaf_block_size %d", leaf_block_size);

    for (unsigned int i = 0; i < max_threads_index; ++i) {
        filter_query_caches[i].thread_id = i;

        filter_query_caches[i].answer = NULL; // modified
        filter_query_caches[i].index = index;
        filter_query_caches[i].config = config;

        filter_query_caches[i].num_leaves = index->num_leaves;
        filter_query_caches[i].leaves = (Node const **) index->leaves;
        filter_query_caches[i].leaf_indices = NULL; // modified
        filter_query_caches[i].leaf_distances = NULL; // modified

        filter_query_caches[i].m256_fetched_cache = aligned_alloc(256, sizeof(VALUE_TYPE) * 8);

        filter_query_caches[i].shared_leaf_id = &shared_leaf_id;
        filter_query_caches[i].num_active_filters_debug = &num_active_filters_debug;

        filter_query_caches[i].query_block_size = leaf_block_size;

        pthread_create(&generate_and_search_threads[i], NULL, filterGenerateAndSearchNodeThread,
                       (void *) &filter_query_caches[i]);
    }

    for (unsigned int i = 0; i < max_threads_index; ++i) {
        pthread_join(generate_and_search_threads[i], NULL);
        free(filter_query_caches[i].m256_fetched_cache);
    }

    clog_debug(CLOG(CLOGGER_ID), "train - generated local queries for %d / %d pre-activated filters",
               num_active_filters_debug, num_active_filters);
}


void generateAndSearchFilterLocalQueries(Config const *config, Index *index, ID_TYPE num_active_filters) {
#ifdef FINE_TIMING
    struct timespec start_timestamp, stop_timestamp;
    TimeDiff time_diff;
    clock_code = clock_gettime(CLK_ID, &start_timestamp);
#endif
    if (index->leaves == NULL) {
        fetchLeaves(index);
    }

    generateAndSearchLocalQueryCore(config, index, num_active_filters);
#ifdef FINE_TIMING
    clock_code = clock_gettime(CLK_ID, &stop_timestamp);
    getTimeDiff(&time_diff, start_timestamp, stop_timestamp);
    clog_info(CLOG(CLOGGER_ID), "train - generate and search local queries = %ld.%lds",
              time_diff.tv_sec, time_diff.tv_nsec);
#endif
}


typedef struct TrainCache {
    Config const *config;
    Index const *index;
    Node const *const *leaves;

    ID_TYPE *shared_leaf_id;
    ID_TYPE leaf_block_size;

    unsigned int stream_id;
} TrainCache;


void *trainThread(void *cache) {
    TrainCache *train_cache = (TrainCache *) cache;

    Config const *config = train_cache->config;

    ID_TYPE block_size = train_cache->leaf_block_size;
    unsigned int num_leaves = train_cache->index->num_leaves;
    ID_TYPE *shared_leaf_id = train_cache->shared_leaf_id;

    ID_TYPE leaf_id, stop_leaf_id;
    Node const *leaf;
    unsigned int stream_id = train_cache->stream_id;

    while ((leaf_id = __sync_fetch_and_add(shared_leaf_id, block_size)) < num_leaves) {
        stop_leaf_id = leaf_id + block_size;
        if (stop_leaf_id > num_leaves) {
            stop_leaf_id = num_leaves;
        }
#ifdef ISAX_PROFILING
        unsigned int local_num_filters_learned = 0;
#endif
        for (unsigned int i = leaf_id; i < stop_leaf_id; ++i) {
            leaf = train_cache->leaves[i];

            if (leaf->filter->is_activated) {
                int return_code = trainNeuralFilter(config, leaf->filter, stream_id, leaf->sax_str);
                if (return_code > 0) {
#ifdef ISAX_PROFILING
                    local_num_filters_learned += 1;
#endif
                } else {
                    clog_error(CLOG(CLOGGER_ID), "train stream %d node %s - failed to train filter %d, size %d",
                               stream_id, leaf->sax_str, leaf->filter->id, leaf->size);
                }
            } else {
#ifdef ISAX_PROFILING
                logNeuralFilter(config, leaf->filter, stream_id, leaf->sax_str);
#endif
            }
        }
#ifdef ISAX_PROFILING
        pthread_mutex_lock(log_lock_profiling);
        num_filters_learned += local_num_filters_learned;
        pthread_mutex_unlock(log_lock_profiling);
#endif
    }

    return NULL;
}


void trainNeuralFilters(Config const *config, Index *index, ID_TYPE num_active_filters) {
    if (index->leaves == NULL) {
        fetchLeaves(index);
    }

#ifdef FINE_TIMING
    struct timespec start_timestamp, stop_timestamp;
    TimeDiff time_diff;
    clock_code = clock_gettime(CLK_ID, &start_timestamp);
#endif
    unsigned int max_threads_train = config->max_threads_train;
    TrainCache train_caches[max_threads_train];
    pthread_t train_threads[max_threads_train];

    ID_TYPE shared_leaf_id = 0;
    ID_TYPE leaf_block_size = (index->num_leaves + max_threads_train - 1) / max_threads_train;

    for (unsigned int j = 0; j < max_threads_train; ++j) {
        train_caches[j].config = config;
        train_caches[j].index = index;
        train_caches[j].leaves = (const Node *const *) index->leaves;

        train_caches[j].leaf_block_size = leaf_block_size;
        train_caches[j].shared_leaf_id = &shared_leaf_id;

        train_caches[j].stream_id = j;

        pthread_create(&train_threads[j], NULL, trainThread, (void *) &train_caches[j]);
    }

    for (unsigned int j = 0; j < max_threads_train; ++j) {
        pthread_join(train_threads[j], NULL);
    }
#ifdef FINE_TIMING
    clock_code = clock_gettime(CLK_ID, &stop_timestamp);
    getTimeDiff(&time_diff, start_timestamp, stop_timestamp);
    clog_info(CLOG(CLOGGER_ID), "train - trained %d / %d filters = %ld.%lds",
              num_filters_learned, num_active_filters,
              time_diff.tv_sec, time_diff.tv_nsec);
#endif
}
