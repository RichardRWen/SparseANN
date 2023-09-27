#include <stdlib.h>
#include <stdio.h>
#include <fcntl.h>

#include <cassert>
#include <set>
#include <unordered_map>
#include <algorithm>

#include "../include/linscan.h"

// Initialize an inverted index with the given number of inverted lists
int inverted_index_init(inverted_index *inv_index, unsigned int num_lists) {
	inv_index->lists = new inverted_list[num_lists];
	if (inv_index->lists == NULL) return 0;

	inv_index->num_lists = num_lists;
	for (int i = 0; i < num_lists; i++) {
		inv_index->lists[i].values = new std::vector<inverted_value>();
	}

	return 1;
}

void inverted_index_free(inverted_index *inv_index) {
	if (inv_index == NULL || inv_index->lists == NULL) return;
	for (int i = 0; i < inv_index->num_lists; i++) {
		if (inv_index->lists[i].values != NULL) {
			delete inv_index->lists[i].values;
			inv_index->lists[i].values = NULL;
		}
	}
	delete[] inv_index->lists;
	inv_index->lists = NULL;
	inv_index->num_lists = 0;
}

// For each nonzero coordinate in the input vector, add the vector's identifier into the corresponding inverted list
// Return the number of nonzero coordinates found
int inverted_index_insert(inverted_index *inv_index, int *vector, uint64_t vector_id) {
	int num_coords = 0;
	for (int i = 0; i < inv_index->num_lists; i++) {
		if (vector[i]) {
			inverted_value new_value;
			new_value.id = vector_id;
			new_value.value = vector[i];
			inv_index->lists[i].values->push_back(new_value);
			num_coords++;
		}
	}
	return num_coords;
}

struct greater_inverted_value {
	bool operator()(const inverted_value a, const inverted_value b) const {
		return a.value > b.value;
	}
};

std::vector<inverted_value> inverted_index_query(inverted_index *inv_index, int *vector, int k) {
	// Calculate the inner products for all documents that share nonzero coordinates with the query vector
	std::unordered_map<uint64_t, int> inner_products;
	std::pair<std::unordered_map<uint64_t, int>::iterator, bool> insert_result;
	for (int i = 0; i < inv_index->num_lists; i++) {
		if (vector[i]) {
			for (int j = 0; j < inv_index->lists[i].values->size(); j++) {
				int partial_inner_product = inv_index->lists[i].values[0][j].value * vector[i];
				std::pair<uint64_t, int> temp_pair(inv_index->lists[i].values[0][j].id, partial_inner_product);
				insert_result = inner_products.insert(temp_pair);
				if (!insert_result.second) {
					insert_result.first->second += partial_inner_product;
				}
			}
		}
	}

	// Use heap to find top k inner products
	std::vector<inverted_value> heap;
	for (auto it = inner_products.begin(); it != inner_products.end(); it++) {
		inverted_value next_value;
		next_value.id = it->first;
		next_value.value = it->second;
		
		heap.push_back(next_value);
		std::push_heap(heap.begin(), heap.end(), greater_inverted_value());
		if (heap.size() > k) {
			std::pop_heap(heap.begin(), heap.end(), greater_inverted_value());
		}
	}

	return heap;
}
