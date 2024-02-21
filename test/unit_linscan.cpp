#include <stdio.h>
#include <iostream>
#include <cassert>

#include "../include/inverted_index.h"

int main(int argc, char **argv) {
	inverted_index<float> inv_index(5);

	float insert_vectors[][5] = {
		{0, 4, 0, 0, 1},
		{3, 2, 0, 0, 0},
		{0, 0, 2, 0, 2}
	};
	for (int i = 0; i < sizeof(insert_vectors)/sizeof(insert_vectors[0]); i++) {
		int ret = inv_index.insert(i, insert_vectors[i]);
		assert(ret);
	}

	float query_vector[5] = {1, 2, 0, 0, 0};
	auto k_top = inv_index.neighbors(query_vector, 3);

	assert(k_top.size() == 2);
	assert(k_top[0].first == 1 && k_top[0].second == 7);
	assert(k_top[1].first == 0 && k_top[1].second == 8);
	
	for (int i = 0; i < k_top.size(); i++) {
		std::cout << k_top[i].first << "\t" << k_top[i].second << std::endl;
	}

	std::cout << "Test completed" << std::endl;
	return 0;
}
