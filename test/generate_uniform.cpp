#include <iostream>
#include <cstring>

#include <set>

#include "forward_index.h"

#include <parlay/sequence.h>
#include <parlay/primitives.h>
#include <openssl/rand.h>

int main(int argc, char **argv) {
	forward_index<float> dataset;
	dataset.dims = 30109;

	//uint64_t target_num_points = 100000;
	//uint64_t target_nonzero = 150;
	uint64_t target_num_points = 10000;
	uint64_t target_nonzero = 50;
	
	uint16_t *rand = new uint16_t[target_nonzero * target_num_points];
	RAND_bytes((unsigned char*)rand, target_nonzero * target_num_points * sizeof(uint16_t));

	dataset.points = parlay::sequence<parlay::sequence<std::pair<uint32_t, float>>>(target_num_points);
	parlay::parallel_for(0, target_num_points, [&] (size_t i) {
		std::set<uint16_t> nonzeroes;
		for (int j = 0; j < target_nonzero; j++) {
			nonzeroes.insert(rand[target_nonzero * i + j] % dataset.dims);
		}
		size_t j = target_nonzero * i;
		for (uint16_t nonzero : nonzeroes) {
			dataset.points[i].push_back(std::make_pair<uint32_t, float>(nonzero, 3.5 * rand[j++] / (1ul << 16)));
		}
	});

	//dataset.write_to_file("data/unif_30109_small.csr", "csr");
	dataset.write_to_file("data/unif_30109_queries.csr", "csr");

	delete[] rand;
}
