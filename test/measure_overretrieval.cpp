#include <iostream>
#include <cstring>

#include "forward_index.h"
#include "ground_truth.h"
#include "coord_order.h"
#include "test_util.h"

#include <parlay/sequence.h>
#include <parlay/delayed_sequence.h>
#include <parlay/primitives.h>
#include <openssl/rand.h>

#define TEST 5

int main(int argc, char **argv) {
	int k = 10;
	int overretrievals[10] = {1, 2, 5, 10, 20, 50, 100, 200, 500, 1000};
	for (int i = 0; i < sizeof(overretrievals) / sizeof(overretrievals[0]); i++) {
		overretrievals[i] *= k;
	}

	forward_index<float> inserts, queries;
	time_function("Reading input files", [&] () {
		inserts = forward_index<float>("data/base_small.csr", "csr");
		queries = forward_index<float>("data/queries.dev.csr", "csr");
	});

	parlay::sequence<parlay::sequence<std::pair<uint32_t, float>>> gtd;
	time_function("Computing ground truth of inputs", [&] () {
		gtd = ground_truth_with_distances(inserts, queries, k);
	});

	switch (TEST) {
	case 0: { // no transformation (sanity check)
		break;
	}
	case 1: { // max compress randomly
		int comp_dims = 400;//1400;
		time_function("Applying transformation - random max sketch with " + std::to_string(comp_dims) + " dims", [&] () {
			auto weights = parlay::sequence<uint32_t>::uninitialized(inserts.dims);
			RAND_bytes((unsigned char*)(&weights[0]), weights.size() * sizeof(weights[0]));
			auto iota = parlay::delayed_tabulate(weights.size(), [] (uint32_t i) { return i; });
			auto perm = parlay::integer_sort(iota,
				[&] (uint32_t a) -> uint32_t {
					return weights[a];
				}
			);
			inserts.reorder_dims(perm);
			inserts = forward_index<float>::group_and_max(inserts, comp_dims);
			queries.reorder_dims(perm);
			queries = forward_index<float>::group_and_max(queries, comp_dims);
		});
		
		break;
	}
	case 2: { // reorder and max compress
		coord_order order("data/processed/reorder/base_small_100000.ord", "ord");
		int comp_dims = 120;//1400;
		time_function("Applying transformation - reorder + max sketch with " + std::to_string(comp_dims) + " dims", [&] () {
			inserts.reorder_dims(order.order_map);
			inserts = forward_index<float>::group_and_max(inserts, comp_dims);
			queries.reorder_dims(order.order_map);
			queries = forward_index<float>::group_and_max(queries, comp_dims);
		});
        int count = 0;
        for (int i = 0; i < inserts.size(); i++) {
            count += inserts.points[i].size();
        }
        std::cout << "avg nonzeros: " << (float)count / inserts.size() << std::endl;
		break;
	}
	case 3: { // max compress randomly
		int comp_dims = 400;//1400;
		time_function("Applying transformation - random sum sketch with " + std::to_string(comp_dims) + " dims", [&] () {
			auto weights = parlay::sequence<uint32_t>::uninitialized(inserts.dims);
			RAND_bytes((unsigned char*)(&weights[0]), weights.size() * sizeof(weights[0]));
			auto iota = parlay::delayed_tabulate(weights.size(), [] (uint32_t i) { return i; });
			auto perm = parlay::integer_sort(iota,
				[&] (uint32_t a) -> uint32_t {
					return weights[a];
				}
			);
			inserts.reorder_dims(perm);
			inserts = forward_index<float>::group_and_sum(inserts, comp_dims);
			queries.reorder_dims(perm);
			queries = forward_index<float>::group_and_sum(queries, comp_dims);
		});
		
		break;
	}
	case 4: { // reorder and max compress
		coord_order order("data/processed/reorder/base_small_100000.ord", "ord");
		int comp_dims = 400;
		time_function("Applying transformation - reorder + sum sketch with " + std::to_string(comp_dims) + " dims", [&] () {
			inserts.reorder_dims(order.order_map);
			inserts = forward_index<float>::group_and_sum(inserts, comp_dims);
			queries.reorder_dims(order.order_map);
			queries = forward_index<float>::group_and_sum(queries, comp_dims);
		});
		break;
	}
	case 5: { // marking only nonzeros
		time_function("Applying transformation - only marking nonzeros", [&] () {
            parlay::parallel_for(0, inserts.points.size(), [&] (size_t i) {
                for (size_t j = 0; j < inserts.points[i].size(); j++) {
                    inserts.points[i][j].second = 1;
                }
            });
            parlay::parallel_for(0, queries.points.size(), [&] (size_t i) {
                for (size_t j = 0; j < queries.points[i].size(); j++) {
                    queries.points[i][j].second = 1;
                }
            });
		});
		break;
	}
	default:
		exit(0);
	}

	parlay::sequence<parlay::sequence<std::pair<uint32_t, float>>> tgtd;
	time_function("Computing ground truth of transformed inputs", [&] () {
		tgtd = ground_truth_with_distances(inserts, queries, overretrievals[(sizeof(overretrievals) / sizeof(overretrievals[0])) - 1]);
	});

	auto gt = parlay::sequence<parlay::sequence<uint32_t>>::from_function(gtd.size(), [&] (size_t i) {
		return parlay::sequence<uint32_t>::from_function(gtd[i].size(), [&] (size_t j) {
			return gtd[i][j].first;
		});
	});
	auto tgt = parlay::sequence<parlay::sequence<uint32_t>>::from_function(tgtd.size(), [&] (size_t i) {
		return parlay::sequence<uint32_t>::from_function(tgtd[i].size(), [&] (size_t j) {
			return tgtd[i][j].first;
		});
	});

	for (int overret : overretrievals) {
		double recall = get_recall(gt, tgt, k, overret);
		std::cout << "Recall " << overret << "@" << k << ":   \t" << recall << std::endl;
	}

	/*exit(0);

	double target_recall = 0.9;
	std::cout << "Target recall:\t" << target_recall << std::endl;
	uint64_t target_found = queries.size() * k * target_recall, total_found = 0;
	std::vector<int> num_not_found(queries.size(), k);*/
	
}
