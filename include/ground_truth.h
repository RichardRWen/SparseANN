#ifndef _SPARSE_GROUND_TRUTH_H_
#define _SPARSE_GROUND_TRUTH_H_

#include <parlay/sequence.h>

#include "forward_index.h"
#include "linscan.h"

template <typename id_type>
parlay::sequence<parlay::sequence<id_type>> ground_truth(char *inserts_file, char *queries_file, int k = 10);

template <typename id_type>
parlay::sequence<parlay::sequence<id_type>> ground_truth(forward_index<id_type>& inserts, forward_index<id_type>& queries, int k = 10);

template <typename id_type>
double recall(parlay::sequence<parlay::sequence<id_type>>& ground_truth, parlay::sequence<parlay::sequence<id_type>>& neighbors, int k = 10);

#endif
