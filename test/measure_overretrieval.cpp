#include <iostream>

#include "forward_index.h"
#include "ground_truth.h"
#include "coord_order.h"
#include "test_util.h"

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

	parlay::sequence<parlay::sequence<uint32_t>> gt;
	time_function("Computing ground truth of inputs", [&] () {
		gt = ground_truth(inserts, queries, k);
	});

	coord_order order("data/processed/reorder/base_small_100000.ord", "ord");
	int comp_dims = 200;
	forward_index<float> transformed_inserts;
	forward_index<float> transformed_queries;
	time_function("Applying transformation", [&] () {
		inserts.reorder_dims(order.order_map);
		inserts = forward_index<float>::group_and_max(inserts, comp_dims);
		queries.reorder_dims(order.order_map);
		queries = forward_index<float>::group_and_max(queries, comp_dims);
	});

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
