#ifndef _SPARSE_GROUND_TRUTH_H_
#define _SPARSE_GROUND_TRUTH_H_

#include <numeric>

#include <parlay/sequence.h>

#include "forward_index.h"
#include "inverted_index.h"


template <typename val_type = float, typename id_type = uint32_t>
parlay::sequence<parlay::sequence<id_type>> ground_truth(char *inserts_file, char *queries_file, int k = 10) { // Does returning a sequence in this way produce copying overheads?
	// create forward index from file
	// TODO: assert this is actually a .csr file
	forward_index<val_type> fwd_index(inserts_file, "csr");
	
	// create reverse index from forward index
	inverted_index<val_type> inv_index(fwd_index);

	forward_index<val_type> queries(queries_file, "csr");
	// call neighbors() on every query in parallel
	auto gt = parlay::sequence<parlay::sequence<id_type>>::from_function(fwd_index.points.size(),
		[&inv_index, &queries, &k] (size_t i) -> parlay::sequence<id_type> {
			auto neighbors_dist = inv_index.neighbors(queries.points[i], k);
			auto neighbors = parlay::sequence<id_type>::from_function(neighbors_dist.size(), 
				[&neighbors_dist] (size_t i) -> id_type {
					return neighbors_dist[i].first;
				}
			);
			return neighbors;
		}
	);
	return gt;
}


template <typename val_type, typename id_type = uint32_t>
parlay::sequence<parlay::sequence<id_type>> ground_truth(forward_index<val_type>& inserts, forward_index<val_type>& queries, int k = 10) {
	inverted_index<val_type> inv_index(inserts);

	auto gt = parlay::sequence<parlay::sequence<id_type>>::from_function(queries.points.size(),
		[&inv_index, &queries, &k] (size_t i) -> parlay::sequence<id_type> {
			auto neighbors = inv_index.neighbors(queries.points[i], k);
			return parlay::sequence<id_type>::from_function(k,
				[&neighbors] (size_t i) -> id_type {
					return neighbors[i].first;
				}
			);
		}
	);
	return gt;
}


template <typename id_type>
double get_recall(parlay::sequence<parlay::sequence<id_type>>& ground_truth, parlay::sequence<parlay::sequence<id_type>>& neighbors, int k1 = 10, int k2 = 10) {
	double recall = 0;
	int num_queries = std::min(ground_truth.size(), neighbors.size());
	int i;
	for (i = 0; i < num_queries; i++) {
		for (int j = 0; j < k1; j++) {
			for (int l = 0; l < k2; l++) {
				if (neighbors[i][l] == ground_truth[i][j]) {
					recall++;
					break;
				}
			}
		}
	}
	return recall / (i * k1);
}

#endif
