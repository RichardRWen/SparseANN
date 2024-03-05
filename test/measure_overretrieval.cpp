#include <iostream>

#include "forward_index.h"
#include "ground_truth.h"
#include "coord_order.h"
#include "test_util.h"

#include <parlay/sequence.h>
#include <parlay/primitives.h>
#include <openssl/rand.h>

#define TEST 0

int main(int argc, char **argv) {
	int k = 10;
	int overretrievals[10] = {1, 2, 5, 10, 20, 50, 100, 200, 500, 1000};
	for (int i = 0; i < sizeof(overretrievals) / sizeof(overretrievals[0]); i++) {
		overretrievals[i] *= k;
	}

	forward_index<float> inserts, queries;
	time_function("Reading input files", [&] () {
		inserts = forward_index<float>("data/base_small.csr", "csr");
		queries = forward_index<float>("data/queries.dev.csr", "csr", 100);
	});

	parlay::sequence<parlay::sequence<uint32_t>> gt;
	time_function("Computing ground truth of inputs", [&] () {
		gt = ground_truth(inserts, queries, k);
	});

	switch (TEST) {
	case 0: { // no transformation (sanity check)
		break;
	}
	case 1: { // compress randomly
		int comp_dims = 1400;
		time_function("Applying transformation - random max sketch", [&] () {
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
	case 2: { // reorder and compress
		coord_order order("data/processed/reorder/base_small_100000.ord", "ord");
		int comp_dims = 200;
		time_function("Applying transformation - reorder + max sketch", [&] () {
			inserts.reorder_dims(order.order_map);
			inserts = forward_index<float>::group_and_max(inserts, comp_dims);
			queries.reorder_dims(order.order_map);
			queries = forward_index<float>::group_and_max(queries, comp_dims);
		});
		break;
	}
	default:
		exit(0);
	}

	parlay::sequence<parlay::sequence<uint32_t>> transformed_gt;
	time_function("Computing ground truth of transformed inputs", [&] () {
		transformed_gt = ground_truth(inserts, queries, overretrievals[(sizeof(overretrievals) / sizeof(overretrievals[0])) - 1]);
	});

	for (int overret : overretrievals) {
		double recall = get_recall(gt, transformed_gt, k, overret);
		std::cout << "Recall " << overret << "@" << k << ":   \t" << recall << std::endl;
	}

	/*exit(0);

	double target_recall = 0.9;
	std::cout << "Target recall:\t" << target_recall << std::endl;
	uint64_t target_found = queries.size() * k * target_recall, total_found = 0;
	std::vector<int> num_not_found(queries.size(), k);*/
	
}
