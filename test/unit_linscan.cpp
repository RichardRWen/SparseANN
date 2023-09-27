#include <stdio.h>

#include <cassert>

#include "../include/linscan.h"

int main(int argc, char **argv) {
	inverted_index inv_index;
	inverted_index_init(&inv_index, 5);

	int insert_vectors[][5] = {
		{0, 4, 0, 0, 1},
		{3, 2, 0, 0, 0},
		{0, 0, 2, 0, 2}
	};
	for (int i = 0; i < sizeof(insert_vectors)/sizeof(insert_vectors[0]); i++) {
		int ret = inverted_index_insert(&inv_index, insert_vectors[i], i);
		assert(ret);
	}

	int query_vector[5] = {1, 2, 0, 0, 0};
	std::vector<inverted_value> k_top = inverted_index_query(&inv_index, query_vector, 3);

	assert(k_top.size() == 2);
	assert(k_top[0].id == 1 && k_top[0].value == 7);
	assert(k_top[1].id == 0 && k_top[1].value == 8);
	
	for (int i = 0; i < k_top.size(); i++) {
		printf("%lu\t%lu\n", k_top[i].id, k_top[i].value);
	}

	inverted_index_free(&inv_index);

	printf("Test completed\n");
	return 0;
}
