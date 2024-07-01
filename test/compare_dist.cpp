#include <iostream>
#include <fstream>
#include <cstring>
#include <algorithm>
#include <vector>

#include "forward_index.h"
#include "ground_truth.h"
#include "coord_order.h"
#include "test_util.h"

#include <parlay/sequence.h>
#include <parlay/delayed_sequence.h>
#include <parlay/primitives.h>
#include <openssl/rand.h>

int main(int argc, char **argv) {
	forward_index<float> inserts, queries;
	time_function("Reading input files", [&] () {
		inserts = forward_index<float>("data/base_1M.csr", "csr");
		queries = forward_index<float>("data/queries.dev.csr", "csr");
	});

	int *ins_counts = new int[inserts.dims];
	int *que_counts = new int[queries.dims];
	assert(inserts.dims == queries.dims);

	for (int i = 0; i < inserts.dims; i++) {
		ins_counts[i] = que_counts[i] = 0;
	}

	int total_ins_coords = 0, total_que_coords = 0;
	for(auto& point : inserts.points) {
		for (auto& coord : point) {
			ins_counts[coord.first]++;
			total_ins_coords++;
		}
	}

	for(auto& point : queries.points) {
		for (auto& coord : point) {
			que_counts[coord.first]++;
			total_que_coords++;
		}
	}

	std::vector<int> indexes(inserts.dims);
	for (int i = 0; i < indexes.size(); i++) indexes[i] = i;
	std::sort(indexes.begin(), indexes.end(), [&] (int a, int b) {
		return ins_counts[a] > ins_counts[b];
	});

	std::ofstream writer("logs/dist.csv");
	writer << "dimension,inserts,queries\n";
	for (int i = 0; i < indexes.size(); i++) {
		writer << indexes[i] << "," << (float)ins_counts[indexes[i]] / total_ins_coords << "," << (float)que_counts[indexes[i]] / total_que_coords << "\n";
	}
	writer.close();
}
