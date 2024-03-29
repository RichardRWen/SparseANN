#include <iostream>
#include <cstdio>
#include <fstream>
#include <strings.h>
#include <string>
#include <vector>
#include <unordered_set>
#include <algorithm>
#include <math.h>
#include <time.h>
#include <numeric>
#include <cassert>

#include "../include/coord_order.h"

#define RANDOM_INPUT 0
#define MANUAL_TEST 0
#define SEQUENTIAL 0
#if SEQUENTIAL
	#define SHINGLE_ORDER shingle_order_seq
	#define ITERATED_SWAP iterated_swap_seq
	#define REORDER reorder_seq
#else
	#define SHINGLE_ORDER shingle_order
	#define ITERATED_SWAP iterated_swap
	#define REORDER reorder
#endif

void print_level(int level) {
	std::cout << "===========================" << std::endl
			  << "====      Level " << level << "      ====" << std::endl
			  << "===========================" << std::endl << std::endl;
}

void print_state(coord_order &order) {
	std::cout << "Order:   \t";
	for (size_t i = 0; i < order.order.size(); i++) {
		std::cout << order.order[i] << " ";
	}
	std::cout << std::endl;
	std::cout << "Order Map:\t";
	for (size_t i = 0; i < order.order.size(); i++) {
		std::cout << order.order_map[i] << " ";
	}
	std::cout << std::endl;
	for (size_t i = 0; i < order.fwd_index.points.size(); i++) {
		std::cout << "Vector " << i << ":\t";
		std::vector<uint32_t> indices(order.fwd_index.points[i].size());
		std::iota(indices.begin(), indices.end(), uint32_t(0));
		std::sort(indices.begin(), indices.end(),
			[&order, i] (uint32_t a, uint32_t b) -> bool {
				return order.order_map[order.fwd_index.points[i][a].first] < order.order_map[order.fwd_index.points[i][b].first];
			});

		size_t index = 0;
		for (size_t j = 0; j < indices.size(); j++) {
			for (; index < order.order_map[order.fwd_index.points[i][indices[j]].first]; index++) std::cout << ". ";
			std::cout << "O ";
			index++;
		}
		for (; index < order.order.size(); index++) std::cout << ". ";
		std::cout << "\t" << order._log_gap_cost(order.fwd_index.points[i]) << std::endl;
	}
	std::cout << "Avg gap cost:\t" << order.log_gap_cost() << std::endl;
}

void print_move_gains(coord_order &order, uint32_t start, uint32_t end) {
	std::vector<double> gains(end - start, 0);
	
	uint32_t set_size = (end - start) / 2;
	uint32_t mid = start + set_size;
	std::unordered_set<uint32_t> set1;
	std::unordered_set<uint32_t> set2;
	for (int i = start; i < mid; i++) set1.insert(order.order[i]);
	for (int i = mid;   i < end; i++) set2.insert(order.order[i]);

	for (size_t i = start; i < end; i++) {
		if (i < mid) gains[i] = order.move_gain_seq(order.order[i], set1, set2);
		else gains[i] = order.move_gain_seq(order.order[i], set2, set1);
	}

	std::vector<uint32_t> indices(end - start);
	std::iota(indices.begin(), indices.end(), (uint32_t)0);
	std::sort(indices.begin(), indices.begin() + set_size,
		[&gains] (uint32_t a, uint32_t b) -> bool {
			return gains[a] < gains[b];
		});
	std::sort(indices.begin() + set_size, indices.end(),
		[&gains] (uint32_t a, uint32_t b) -> bool {
			return gains[a] < gains[b];
		});
	
	std::cout << "Set 1 move gains:\t";
	for (int i = 0; i < set_size; i++) {
		std::cout << "(" << order.order[start + indices[i]] << ", " << gains[indices[i]] << ") ";
	}
	std::cout << std::endl;
	std::cout << "Set 2 move gains:\t";
	for (int i = set_size; i < end - start; i++) {
		std::cout << "(" << order.order[start + indices[i]] << ", " << gains[indices[i]] << ") ";
	}
	std::cout << std::endl;
}

double get_partition_cost(coord_order &order, uint32_t start, uint32_t end) {
	uint32_t set_size = (end - start) / 2;
	uint32_t mid = start + set_size;

	std::unordered_set<uint32_t> set1;
	std::unordered_set<uint32_t> set2;
	for (int i = start; i < mid; i++) set1.insert(order.order[i]);
	for (int i = mid;   i < end; i++) set2.insert(order.order[i]);

	double partition_cost = 0;
	for (auto point : order.fwd_index.points) {
		uint32_t deg1 = 0, deg2 = 0;
		for (auto coord : point) {
			if (set1.find(coord.first) != set1.end()) deg1++;
			else if (set2.find(coord.first) != set2.end()) deg2++;
		}
		partition_cost += deg1 * log((double)(set1.size() + 1) / (deg1 + 1))
			+ deg2 * log((double)(set2.size() + 1) / (deg2 + 1));
		// I decided to add 1 to the numerators here so that the log can't be negative
	}

	/*double partition_cost = 0;
	for (int i = start; i < end; i++) {
		for (auto vector : order.inv_index.posting_lists[order.order[i]]) {
			uint32_t deg1 = 0, deg2 = 0;
			for (auto coord : order.fwd_index.points[vector.id]) {
				if (set1.find(coord.index) != set1.end()) deg1++;
				else if (set2.find(coord.index) != set2.end()) deg2++;
			}
			partition_cost += deg1 * log((double)(set1.size()) / (deg1 + 1))
							+ deg2 * log((double)(set2.size()) / (deg2 + 1)); // This could end up being negative. Is that intentional? Maybe add one to each of the set sizes as well?
		}
	}*/

	return partition_cost;
}

int main(int argc, char **argv) {
#if RANDOM_INPUT
	#define MANUAL_TEST 0
	uint32_t num_dims = 100;
	uint32_t num_points = 1000;
	float coord_frequency = 0.1;
	srand(time(NULL));
	coord_order order(num_dims);

	std::cout << "Generating random point set" << std::endl;
	for (uint32_t i = 0; i < num_points; i++) {
		order.fwd_index.points.push_back(parlay::sequence<std::pair<uint32_t, float>>(0));
		for (uint32_t j = 0; j < num_dims; j++) {
			if ((float)rand() / RAND_MAX < coord_frequency) {
				order.fwd_index.points[i].push_back(std::make_pair(j, (float)1));
			}
		}
	}
	order.inv_index = inverted_index<float, uint32_t>(order.fwd_index);
	std::cout << "Generated " << num_points << " points of dim " << num_dims << " at nonzero frequency " << coord_frequency << std::endl;
#else
	srand(0);
	size_t sample_size = -1ull;
	if (argc < 2) {
		std::cout << "Usage: " << argv[0] << " [csr infile] [optional sample size]" << std::endl;
		exit(0);
	}
	if (argc >= 3) {
		sample_size = std::stoi(argv[2]);
	}

	if (sample_size == -1ull) std::cout << "Reading all lines from " << argv[1] << std::endl;
	else std::cout << "Reading " << sample_size << " lines from " << argv[1] << std::endl;
	coord_order order(argv[1], "csr", sample_size);
	if (!order.inv_index.num_lists) exit(0);
#endif
	
#if MANUAL_TEST
	std::cout << "===Initial State===" << std::endl;
	print_state(order);
	std::cout << "Partition cost:\t" << get_partition_cost(order, 0, order.order.size()) << std::endl;


	std::cout << std::endl;
	print_level(1);

	std::cout << "Treating sets of size 100\%" << std::endl;
	std::cout << "Applying min hash" << std::endl;
	order.SHINGLE_ORDER(0, order.order.size());
	std::cout << "===State after min hash===" << std::endl;
	print_state(order);
	std::cout << "Partition cost:\t" << get_partition_cost(order, 0, order.order.size()) << std::endl;

	std::cout << std::endl;
	print_move_gains(order, 0, order.order.size());
	std::cout << "Applying 1 swap round" << std::endl;
	order.ITERATED_SWAP(0, order.order.size());
	std::cout << "===State after 1 swap round===" << std::endl;
	print_state(order);
	std::cout << "Partition cost:\t" << get_partition_cost(order, 0, order.order.size()) << std::endl;

	std::cout << std::endl;
	print_move_gains(order, 0, order.order.size());
	std::cout << "Applying 1 swap round" << std::endl;
	order.ITERATED_SWAP(0, order.order.size());
	std::cout << "===State after 2 swap rounds===" << std::endl;
	print_state(order);
	std::cout << "Partition cost:\t" << get_partition_cost(order, 0, order.order.size()) << std::endl;

	std::cout << std::endl;
	print_move_gains(order, 0, order.order.size());
	std::cout << "Applying 18 swap rounds" << std::endl;
	for (int i = 0; i < 18; i++) {
		if (!order.ITERATED_SWAP(0, order.order.size())) break;
	}
	std::cout << "===State after 20 swap rounds===" << std::endl;
	print_state(order);
	std::cout << "Partition cost:\t" << get_partition_cost(order, 0, order.order.size()) << std::endl;


	std::cout << std::endl;
	print_level(2);

	uint32_t mid = order.order.size() / 2;
	std::cout << "Treating sets of size 50\%" << std::endl;
	std::cout << "Applying min hash" << std::endl;
	order.SHINGLE_ORDER(0, mid);
	order.SHINGLE_ORDER(mid, order.order.size());
	print_state(order);
	std::cout << "Partition cost:\t" << get_partition_cost(order, 0, mid) + get_partition_cost(order, mid, order.order.size()) << std::endl;

	std::cout << std::endl;
	print_move_gains(order, 0, mid);
	print_move_gains(order, mid, order.order.size());
	std::cout << "Applying 1 swap round" << std::endl;
	order.ITERATED_SWAP(0, mid);
	order.ITERATED_SWAP(mid, order.order.size());
	std::cout << "===State after 1 swap round===" << std::endl;
	print_state(order);
	std::cout << "Partition cost:\t" << get_partition_cost(order, 0, mid) + get_partition_cost(order, mid, order.order.size()) << std::endl;

	std::cout << std::endl;
	print_move_gains(order, 0, mid);
	print_move_gains(order, mid, order.order.size());
	std::cout << "Applying 1 swap round" << std::endl;
	order.ITERATED_SWAP(0, mid);
	order.ITERATED_SWAP(mid, order.order.size());
	std::cout << "===State after 2 swap rounds===" << std::endl;
	print_state(order);
	std::cout << "Partition cost:\t" << get_partition_cost(order, 0, mid) + get_partition_cost(order, mid, order.order.size()) << std::endl;

	std::cout << std::endl;
	print_move_gains(order, 0, mid);
	print_move_gains(order, mid, order.order.size());
	std::cout << "Applying 18 swap rounds" << std::endl;
	for (int i = 0; i < 18; i++) {
		if (!order.ITERATED_SWAP(0, mid)) break;
	}
	for (int i = 0; i < 18; i++) {
		if (!order.ITERATED_SWAP(mid, order.order.size())) break;
	}
	std::cout << "===State after 20 swap rounds===" << std::endl;
	print_state(order);
	std::cout << "Partition cost:\t" << get_partition_cost(order, 0, mid) + get_partition_cost(order, mid, order.order.size()) << std::endl;

#else
	#if !(RANDOM_INPUT)
	if (order.fwd_index.dims <= 40 && order.fwd_index.points.size() <= 40) {
		std::cout << "===Initial State===" << std::endl;
		print_state(order);
		std::cout << "Partition cost:\t" << get_partition_cost(order, 0, order.order.size()) << std::endl;

		order.REORDER(20);

		std::cout << "===Final State===" << std::endl;
		print_state(order);
	}
	else order.REORDER(20, true);
	#else
	order.REORDER(20, true);
	#endif

#endif

	return 0;
}
