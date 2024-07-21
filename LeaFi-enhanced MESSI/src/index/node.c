/******************************************************************************
 * Copyright (c) 2024, the LeaFi author(s).
 ******************************************************************************/

#include "node.h"

#include <math.h>

#include "sax.h"
#include "file.h"


Node *initializeNode(SAXSymbol *sax, SAXMask *masks, unsigned int sax_length, unsigned int sax_cardinality) {
    Node *node = initializeNode4Load(sax2str(sax, masks, sax_length, sax_cardinality));

    node->sax = sax;
    node->masks = masks;

    return node;
}


Node *initializeNode4Load(char *sax_str) {
    Node *node = malloc(sizeof(Node));

    node->index = NULL;

    node->sax = NULL;
    node->masks = NULL;
    node->sax_str = sax_str;

    node->squeezed_masks = NULL;

    node->upper_envelops = NULL;
    node->lower_envelops = NULL;

    node->ids = NULL;
    node->start_id = 0;

    node->saxs = NULL;
    node->values = NULL;

    node->size = 0;
    node->capacity = 0;

//    node->num_synthetic_queries = 0;

    node->left = NULL;
    node->right = NULL;

    node->lock = malloc(sizeof(pthread_mutex_t));
    assert(pthread_mutex_init(node->lock, NULL) == 0);

    node->compactness = -1;

    node->filter = NULL;
    node->filter_id = -1;

    node->node_dump_filepath = NULL;
    node->data_dump_filepath = NULL;
    node->filter_dump_filepath = NULL;
//    node->query_dump_filepath = NULL;

    node->node_load_filepath = NULL;
    node->data_load_filepath = NULL;
    node->filter_load_filepath = NULL;

    return node;
}


void insertNode(Node *leaf, ID_TYPE id, unsigned int initial_leaf_size, unsigned int leaf_size) {
    if (leaf->capacity == 0) {
        leaf->ids = malloc(sizeof(ID_TYPE) * (leaf->capacity = initial_leaf_size));
    } else if (leaf->size == leaf->capacity) {
        if ((leaf->capacity *= 2) > leaf_size) {
            leaf->capacity = leaf_size;
        }
        leaf->ids = realloc(leaf->ids, sizeof(ID_TYPE) * leaf->capacity);
    }

    leaf->ids[leaf->size++] = id;
}


void initFilterInfoRecursive(Config const *config, Node *node, int *filter_id) {
    if (node != NULL) {
        if (node->size == 0) {
            assert(node->left != NULL && node->right != NULL);

            initFilterInfoRecursive(config, node->left, filter_id);
            initFilterInfoRecursive(config, node->right, filter_id);
        } else if (node->size > 0) {
            assert(node->left == NULL && node->right == NULL);

            node->filter = initFilterInfo(config, node->size, *filter_id);
            *filter_id += 1;
        }
    }
}


void addFilterTrainQueryRecursive(Config const *config, Node *node, VALUE_TYPE const *filter_global_queries,
                                  bool if_check_activate) {
    if (node != NULL) {
        if (node->left != NULL && node->right != NULL) {
            assert(node->size == 0);

            addFilterTrainQueryRecursive(config, node->left, filter_global_queries, if_check_activate);
            addFilterTrainQueryRecursive(config, node->right, filter_global_queries, if_check_activate);
        } else if (node->size > 0) {
            assert(node->left == NULL && node->right == NULL);

            addFilterTrainQuery(node->filter, filter_global_queries,
                                config->filter_query_load_size > 0 ? config->filter_query_load_size
                                                                   : config->filter_num_synthetic_query_global,
                                config->filter_num_synthetic_query_local > 0 ? config->filter_num_synthetic_query_local
                                                                             : config->filter_num_synthetic_query_local_max);
        }
    }
}


void pushFilters(FilterAllocator *allocator, Node *node) {
    if (node != NULL) {
        if (node->size > 0) {
            assert(node->left == NULL && node->right == NULL);

            pushFilter(allocator, node->filter);
        } else { // node->size == 0
            assert(node->left != NULL && node->right != NULL);

            pushFilters(allocator, node->left);
            pushFilters(allocator, node->right);
        }
    }
}


void inspectNode(Node *node, unsigned int *num_series, unsigned int *num_leaves, unsigned int *num_roots,
                 unsigned int *num_filters, unsigned int *num_series_filter, bool print_leaf_size) {
    if (node != NULL) {
        if (num_roots != NULL) {
            *num_roots += 1;
        }

        if (node->size != 0) {
#ifdef FINE_PROFILING
            if (print_leaf_size) {
                clog_info(CLOG(CLOGGER_ID), "index - node %s = %lu", node->sax_str, node->size);
            }
#endif
            if (node->filter && node->filter->is_activated) {
                *num_filters += 1;
//                node->filter_id = node->filter->id; // unused

                *num_series_filter += node->size;
            }

            *num_leaves += 1;
            *num_series += node->size;
        } else if (node->left != NULL) {
            inspectNode(node->left, num_series, num_leaves, NULL, num_filters, num_series_filter, print_leaf_size);
            inspectNode(node->right, num_series, num_leaves, NULL, num_filters, num_series_filter, print_leaf_size);
        }
    }
}


bool testSeriesInNodeEnvelope(Node const *node, VALUE_TYPE const *summarizations, unsigned int sax_length,
                              VALUE_TYPE const *breakpoints) {
    VALUE_TYPE const *current_breakpoints;
    ID_TYPE hit_counter = 0;

    for (ID_TYPE segment_i = 0; segment_i < sax_length; ++segment_i) {
//        if (node->masks[segment_i] > 1) { // cardinality == 1 does not further split besides the root splits
        current_breakpoints = breakpoints + OFFSETS_BY_SEGMENTS[segment_i] +
                              OFFSETS_BY_MASK[node->masks[segment_i]] +
                              ((unsigned int) node->sax[segment_i] >> SHIFTS_BY_MASK[node->masks[segment_i]]);

        if (VALUE_GEQ(summarizations[segment_i], *current_breakpoints) &&
            VALUE_LEQ(summarizations[segment_i], *(current_breakpoints + 1))) {
            hit_counter += 1;
        }
//        }
    }

    clog_debug(CLOG(CLOGGER_ID), "filter %d - %d/%d segments matched", node->filter->id, hit_counter, sax_length);

    return hit_counter == sax_length;
}


VALUE_TYPE getCompactness(Node *leaf_node, VALUE_TYPE const *values, unsigned int series_length) {
    if (leaf_node->size == 0) {
        return -1;
    } else if (leaf_node->size == 1) {
        return 0;
    } else if (leaf_node->compactness > 0) {
        return leaf_node->compactness;
    }

    double sum = 0;
    VALUE_TYPE *local_m256_fetched_cache = aligned_alloc(256, sizeof(VALUE_TYPE) * 8);

    VALUE_TYPE const *outer_current_series = values + series_length * leaf_node->start_id;
    VALUE_TYPE const *inner_current_series;
    VALUE_TYPE const *stop = values + series_length * (leaf_node->start_id + leaf_node->size);

    while (outer_current_series < stop) {
        inner_current_series = outer_current_series + series_length;

        while (inner_current_series < stop) {
            sum += sqrt(l2SquareSIMD(
                    series_length, outer_current_series, inner_current_series, local_m256_fetched_cache));

            inner_current_series += series_length;
        }

        outer_current_series += series_length;
    }

    leaf_node->compactness = (VALUE_TYPE) (sum / (double) (leaf_node->size * (leaf_node->size - 1) / 2.));

    free(local_m256_fetched_cache);
    return leaf_node->compactness;
}


void dumpNode(Config const *config, Node *node, VALUE_TYPE const *values, SAXSymbol const *saxs) {
    node->node_dump_filepath = concat(3, config->nodes_dump_folderpath, node->sax_str, config->dump_filename_postfix);
    FILE *node_file = fopen(node->node_dump_filepath, "wb");

    fwrite(node->sax, sizeof(SAXSymbol), SAX_SIMD_ALIGNED_LENGTH, node_file);
    fwrite(node->masks, sizeof(SAXMask), config->sax_length, node_file);
    fwrite(&node->size, sizeof(unsigned int), 1, node_file);

    if (node->size == 0) {
        assert(node->left != NULL && node->right != NULL);

        size_t str_len = strlen(node->left->sax_str);
        fwrite(&str_len, sizeof(size_t), 1, node_file);
        fwrite(node->left->sax_str, sizeof(char), str_len, node_file);

        str_len = strlen(node->right->sax_str);
        fwrite(&str_len, sizeof(size_t), 1, node_file);
        fwrite(node->right->sax_str, sizeof(char), str_len, node_file);
    }

    fwrite(&node->filter_id, sizeof(int), 1, node_file); // should not be used
    fclose(node_file);
    node->node_load_filepath = node->node_dump_filepath;

    if (node->size > 0) {
        assert(node->left == NULL && node->right == NULL);

        node->data_dump_filepath = concat(3, config->data_dump_folderpath, node->sax_str,
                                          config->dump_filename_postfix);
        FILE *data_file = fopen(node->data_dump_filepath, "wb");

        fwrite(saxs + SAX_SIMD_ALIGNED_LENGTH * node->start_id,
               sizeof(SAXSymbol), SAX_SIMD_ALIGNED_LENGTH * node->size,
               data_file);
        fwrite(values + config->series_length * node->start_id,
               sizeof(VALUE_TYPE), config->series_length * node->size,
               data_file);

        fclose(data_file);
        node->data_load_filepath = node->data_dump_filepath;
    } else {
        assert(node->left != NULL && node->right != NULL);

        dumpNode(config, node->left, values, saxs);
        dumpNode(config, node->right, values, saxs);
    }

    if (config->on_disk) {
        node->values = NULL;
        node->saxs = NULL;
    }
}


Node *loadNode(Config const *config, Node *node, bool free_mask, bool free_sax) {
    node->node_load_filepath = concat(3, config->nodes_load_folderpath, node->sax_str, config->dump_filename_postfix);

    if (!checkFileExists(node->node_load_filepath)) {
        if (strlen(node->sax_str) > 2 * config->sax_length - 1) {
            // only first-layer nodes might be freed
            clog_error(CLOG(CLOGGER_ID), "index - miss %s; expected at %s", node->sax_str, node->node_load_filepath);
        }

        freeNode(node, free_mask, free_sax);
        return NULL;
    }

    if (strlen(node->sax_str) == 2 * config->sax_length - 1) {
        assert(node->sax_str != NULL);
    }

    FILE *node_file = fopen(node->node_load_filepath, "rb");

    node->sax = aligned_alloc(128, sizeof(SAXSymbol) * SAX_SIMD_ALIGNED_LENGTH);
    size_t nitems = fread(node->sax, sizeof(SAXSymbol), config->sax_length, node_file);
    assert(nitems == config->sax_length);

    node->masks = aligned_alloc(256, sizeof(SAXMask) * config->sax_length);
    nitems = fread(node->masks, sizeof(SAXMask), config->sax_length, node_file);
    assert(nitems == config->sax_length);

    nitems = fread(&node->size, sizeof(unsigned int), 1, node_file);
    assert(nitems == 1);

    if (node->size == 0) {
        size_t str_len;

        // load left child
        nitems = fread(&str_len, sizeof(size_t), 1, node_file);
        assert(nitems == 1);

        char *left_sax_str = malloc(sizeof(char) * (str_len + 1));
        nitems = fread(left_sax_str, sizeof(char), str_len, node_file);
        assert(nitems == str_len);

        left_sax_str[str_len] = '\0';
        node->left = initializeNode4Load(left_sax_str);

        // load right child
        nitems = fread(&str_len, sizeof(size_t), 1, node_file);
        assert(nitems == 1);

        char *right_sax_str = malloc(sizeof(char) * (str_len + 1));
        nitems = fread(right_sax_str, sizeof(char), str_len, node_file);
        assert(nitems == str_len);

        right_sax_str[str_len] = '\0';
        node->right = initializeNode4Load(right_sax_str);
    }

    nitems = fread(&node->filter_id, sizeof(int), 1, node_file);  // should not be used
    assert(nitems == 1);

    fclose(node_file);

    if (node->size > 0) {
        assert(node->left == NULL && node->right == NULL);

        node->data_load_filepath = concat(3, config->data_load_folderpath, node->sax_str,
                                          config->dump_filename_postfix);

        if (!config->on_disk) { // in-memory
            FILE *data_file = fopen(node->data_load_filepath, "rb");

            assert(node->saxs == NULL);
            node->saxs = aligned_alloc(128, sizeof(SAXSymbol) * SAX_SIMD_ALIGNED_LENGTH * node->size);
            nitems = fread(node->saxs, sizeof(SAXSymbol), SAX_SIMD_ALIGNED_LENGTH * node->size, data_file);
            assert(nitems == SAX_SIMD_ALIGNED_LENGTH * node->size);

            assert(node->values == NULL);
            node->values = aligned_alloc(256, sizeof(VALUE_TYPE) * config->series_length * node->size);
            nitems = fread(node->values, sizeof(VALUE_TYPE), config->series_length * node->size, data_file);
            assert(nitems == config->series_length * node->size);

            fclose(data_file);
        } else {
            FILE *data_file = fopen(node->data_load_filepath, "rb");
            if (data_file == NULL) {
                clog_error(CLOG(CLOGGER_ID), "index - failed to find the data of node %s, expected at %s",
                           node->sax_str, node->data_load_filepath);
                exit(-1);
            }

            fseek(data_file, 0L, SEEK_END);
            ID_TYPE num_bytes_found = ftell(data_file);
            ID_TYPE num_bytes_expected = node->size * (
                    sizeof(SAXSymbol) * SAX_SIMD_ALIGNED_LENGTH + sizeof(VALUE_TYPE) * config->series_length);
            if (num_bytes_found != num_bytes_expected) {
                clog_error(CLOG(CLOGGER_ID), "index - data of node %s corrupted, expected %d but found %d bytes",
                           node->sax_str, num_bytes_expected, num_bytes_found);
                exit(-1);
            }

            fclose(data_file);
        }
    } else {
        assert(node->left != NULL && node->right != NULL);

        loadNode(config, node->left, false, false);
        loadNode(config, node->right, false, false);
    }

//    clog_debug(CLOG(CLOGGER_ID), "index - loaded node %s, size %d",
//               node->sax_str, node->size);
    return node;
}


void dumpFilters(const Config *config, Node *node) {
    if (node != NULL) {
        if (node->size == 0) {
            assert(node->left != NULL && node->right != NULL);

            dumpFilters(config, node->left);
            dumpFilters(config, node->right);
        } else {
            assert(node->left == NULL && node->right == NULL);

            char *filter_dump_prefix = concat(2, config->filters_dump_folderpath, node->sax_str);
            dumpFilter(config, node->filter, filter_dump_prefix);
//            clog_debug(CLOG(CLOGGER_ID), "index - dump node %s %d filter %d, activated %d, trained %d",
//                       node->sax_str, node->filter_id, node->neural_filter->id,
//                       node->neural_filter->is_activated, node->neural_filter->is_trained);
        }
    }
}


void loadFilterRecursive(Config const *config, Node *node) {
    if (node != NULL) {
        if (node->size == 0) {
            assert(node->left != NULL && node->right != NULL);

            loadFilterRecursive(config, node->left);
            loadFilterRecursive(config, node->right);
        } else {
            assert(node->left == NULL && node->right == NULL);

            char *filter_load_prefix = concat(2, config->filters_load_folderpath, node->sax_str);
            loadFilter(config, filter_load_prefix, node->filter);
//            clog_debug(CLOG(CLOGGER_ID), "index - load node %s %d filter %d, activated %d, trained %d",
//                       node->sax_str, node->filter_id, node->neural_filter->id,
//                       node->neural_filter->is_activated, node->neural_filter->is_trained);
        }
    }
}


void freeNode(Node *node, bool free_mask, bool free_sax) {
    if (node != NULL) {
        if (node->left != NULL) {
            freeNode(node->left, false, false);
            freeNode(node->right, true, true);
        }

        free(node->sax_str);
        node->sax_str = NULL;

        bool is_free_load_filepaths = node->node_dump_filepath != node->node_load_filepath;
        if (node->node_dump_filepath != NULL) {
            free(node->node_dump_filepath);
            node->node_dump_filepath = NULL;
        }
        if (node->data_dump_filepath != NULL) {
            free(node->data_dump_filepath);
            node->data_dump_filepath = NULL;
        }
        if (node->filter_dump_filepath != NULL) {
            free(node->filter_dump_filepath);
            node->filter_dump_filepath = NULL;
        }
//        if (node->query_dump_filepath != NULL) {
//            free(node->query_dump_filepath);
//            node->query_dump_filepath = NULL;
//        }

        if (is_free_load_filepaths) {
            if (node->node_load_filepath != NULL) {
                free(node->node_load_filepath);
                node->node_load_filepath = NULL;
            }
            if (node->data_load_filepath != NULL) {
                free(node->data_load_filepath);
                node->data_load_filepath = NULL;
            }
            if (node->filter_load_filepath != NULL) {
                free(node->filter_load_filepath);
                node->filter_load_filepath = NULL;
            }
        }

        if (node->saxs != NULL) {
            free(node->saxs);
            node->saxs = NULL;
        }

        if (node->values != NULL) {
            free(node->values);
            node->values = NULL;
        }

        if (free_mask && node->masks != NULL) {
            free(node->masks);
            node->masks = NULL;
        }

        if (free_sax && node->sax != NULL) {
            free(node->sax);
            node->sax = NULL;
        }

//        free(node->index);
        node->index = NULL;

        if (node->squeezed_masks != NULL) {
            free(node->squeezed_masks);
            node->squeezed_masks = NULL;
        }
        if (node->upper_envelops != NULL) {
            free(node->upper_envelops);
            node->upper_envelops = NULL;
        }
        if (node->lower_envelops != NULL) {
            free(node->lower_envelops);
            node->upper_envelops = NULL;
        }

        if (node->ids) {
            free(node->ids);
            node->ids = NULL;
        }

        if (node->filter) {
            freeFilter(node->filter);
            node->filter = NULL;
        }

        pthread_mutex_destroy(node->lock);
        free(node->lock);

        free(node);
    }
}


void cleanNode(Node *node) {
    node->size = 0;
    node->capacity = 0;

    if (node->ids) {
        free(node->ids);
        node->ids = NULL;
    }

    if (node->filter) {
        freeFilter(node->filter);
        node->filter = NULL;
    }
}
