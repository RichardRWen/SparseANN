#include "ground_truth.h"

template <typename id_type>
parlay::sequence<parlay::sequence<id_type>> ground_truth(char *inserts_file, char *queries_file, int k) { // Does returning a sequence in this way result in copying overheads?
	// create forward index from file
	// TODO: assert this is actually a .csr file
	forward_index<float> fwd_index(inserts_file, "csr");
	
	// create reverse index from forward index
	inverted_index<float> inv_index(fwd_index);

	forward_index<float> queries(queries_file, "csr");
	// call neighbors() on every query in parallel
	parlay::sequence<parlay::sequence<id_type>> ground_truth(fwd_index.points.size(),
		[&inv_index, &queries, &k] (size_t i) -> parlay::sequence<id_type> {
			auto neighbors_dist = inv_index.neighbors(queries.points[i], k);
			parlay::sequence<id_type> neighbors(neighbors_dist.size(), 
				[&neighbors_dist] (size_t i) -> id_type {
					return neighbors_dist[i].first;
				}
			);
			return neighbors;
		}
	);
	return ground_truth;
}

template <typename id_type>
parlay::sequence<parlay::sequence<id_type>> ground_truth(forward_index<id_type>& inserts, forward_index<id_type>& queries, int k) {
	inverted_index<float> inv_index(inserts);

	parlay::sequence<parlay::sequence<id_type>> ground_truth(queries.points.size(),
		[&inv_index, &queries, &k] (size_t i) -> parlay::sequence<id_type> {
			return inv_index.neighbors(queries.points[i], k);
		}
	);
	return ground_truth;
}

template <typename id_type>
double recall(parlay::sequence<parlay::sequence<id_type>>& ground_truth, parlay::sequence<parlay::sequence<id_type>>& neighbors, int k) {
	double recall = 0;
	int i;
	for (i = 0; i < ground_truth.size() && i < neighbors.size(); i++) {
		for (int j = 0; j < k; j++) {
			for (int l = 0; l < k; l++) {
				if (neighbors[i][l] == ground_truth[i][j]) {
					recall++;
					break;
				}
			}
		}
	}
	return recall / (i * k);
}
