#ifndef _SPARSE_INVERTED_INDEX_H_
#define _SPARSE_INVERTED_INDEX_H_

#include <stdlib.h>
#include <cstdint>
#include <vector>
#include <unordered_map>
#include <cstring>
#include <fstream>
#include <iostream>
#include <algorithm>

#include <parlay/sequence.h>

#include "forward_index.h"

template <typename val_type = float, typename id_type = uint32_t>
class inverted_index {
public:
	uint32_t num_lists;
	parlay::sequence<parlay::sequence<std::pair<id_type, val_type>>> posting_lists;

	inverted_index(const unsigned int num_lists) : num_lists(num_lists), posting_lists(num_lists) {}
	inverted_index(const char *filename, const char *filetype, const size_t _num_to_read = -1ULL) {
		if (strcmp(filetype, "csr") == 0) {
			std::ifstream indptr_reader(filename);
			if (!indptr_reader.is_open()) {
				std::cout << "Unable to open file " << filename << " for reading" << std::endl;
				num_lists = 0;
				return;
			}
			std::ifstream index_reader(filename);
			std::ifstream data_reader(filename);

			uint64_t num_vecs, num_dims, num_vals;
			indptr_reader.read((char*)(&num_vecs), sizeof(uint64_t));
			indptr_reader.read((char*)(&num_dims), sizeof(uint64_t));
			indptr_reader.read((char*)(&num_vals), sizeof(uint64_t));
			index_reader.seekg((num_vecs + 4) * sizeof(uint64_t));
			data_reader.seekg((num_vecs + 4) * sizeof(uint64_t) + num_vals * sizeof(unsigned int));

			num_lists = num_dims;
			posting_lists.resize(num_dims);

			size_t num_to_read = (_num_to_read < num_vecs ? _num_to_read : num_vecs);

			uint64_t indptr_start, indptr_end;
			unsigned int index;
			float data;
			indptr_reader.read((char*)(&indptr_end), sizeof(uint64_t));
			for (size_t i_vecs = 0; i_vecs < num_to_read; i_vecs++) {
				indptr_start = indptr_end;
				indptr_reader.read((char*)(&indptr_end), sizeof(uint64_t));
				for (; indptr_start < indptr_end; indptr_start++) {
					index_reader.read((char*)(&index), sizeof(unsigned int));
					data_reader.read((char*)(&data), sizeof(float));
					posting_lists[index].emplace_back(i_vecs, data);
				}
			}
		}
		else {
			num_lists = 0;
		}
	}
	inverted_index(const forward_index<val_type>& fwd_index) {
		num_lists = fwd_index.dims;
		posting_lists.resize(num_lists);
		for (id_type i = 0; i < fwd_index.points.size(); i++) {
			for (auto coord : fwd_index.points[i]) {
				posting_lists[coord.first].push_back(std::make_pair(i, coord.second));
			}
		}
	}
	
	int insert(id_type vector_id, val_type *vector) {
		int num_coords = 0;
		for (int i = 0; i < num_lists; i++) {
			if (vector[i]) {
				posting_lists[i].emplace_back(vector_id, vector[i]);
				num_coords++;
			}
		}
		return num_coords;
	}

	parlay::sequence<std::pair<id_type, val_type>> neighbors(val_type *vector, int k) {
		// Calculate the inner products for all documents that share nonzero coordinates with the query vector
		std::unordered_map<id_type, val_type> inner_products;
		std::pair<typename std::unordered_map<id_type, val_type>::iterator, bool> insert_result;
		for (int i = 0; i < num_lists; i++) {
			if (vector[i]) {
				for (int j = 0; j < posting_lists[i].size(); j++) {
					val_type partial_inner_product = posting_lists[i][j].second * vector[i];
					std::pair<id_type, val_type> temp_pair(posting_lists[i][j].first, partial_inner_product);
					insert_result = inner_products.insert(temp_pair);
					if (!insert_result.second) {
						insert_result.first->second += partial_inner_product;
					}
				}
			}
		}

		// Use heap to find top k inner products
		parlay::sequence<std::pair<id_type, val_type>> heap;
		for (auto it = inner_products.begin(); it != inner_products.end(); it++) {
			heap.push_back(std::make_pair((id_type)it->first, (val_type)it->second));
			std::push_heap(heap.begin(), heap.end(),
				[] (std::pair<id_type, val_type>& a, std::pair<id_type, val_type>& b) -> bool {
					return a.second > b.second;
				}
			);
			if (heap.size() > k) {
				std::pop_heap(heap.begin(), heap.end(),
					[] (std::pair<id_type, val_type>& a, std::pair<id_type, val_type>& b) -> bool {
						return a.second > b.second;
					}
				);
			}
		}

		return heap;
	}

	parlay::sequence<std::pair<id_type, val_type>> neighbors(parlay::sequence<std::pair<uint32_t, val_type>>& vector, int k) {
		// Calculate the inner products for all documents that share nonzero coordinates with the query vector
		std::unordered_map<id_type, val_type> inner_products;
		std::pair<typename std::unordered_map<id_type, val_type>::iterator, bool> insert_result;
		for (int i = 0; i < vector.size(); i++) {
			for (int j = 0; j < posting_lists[vector[i].first].size(); j++) {
				val_type partial_inner_product = posting_lists[vector[i].first][j].second * vector[i].second;
				std::pair<id_type, val_type> temp_pair(posting_lists[vector[i].first][j].first, partial_inner_product);
				insert_result = inner_products.insert(temp_pair);
				if (!insert_result.second) {
					insert_result.first->second += partial_inner_product;
				}
			}
		}

		// Use heap to find top k inner products
		parlay::sequence<std::pair<id_type, val_type>> heap;
		for (auto it = inner_products.begin(); it != inner_products.end(); it++) {
			heap.push_back(std::make_pair((id_type)it->first, (val_type)it->second));
			std::push_heap(heap.begin(), heap.end(),
				[] (std::pair<id_type, val_type>& a, std::pair<id_type, val_type>& b) -> bool {
					return a.second > b.second;
				}
			);
			if (heap.size() > k) {
				std::pop_heap(heap.begin(), heap.end(),
					[] (std::pair<id_type, val_type>& a, std::pair<id_type, val_type>& b) -> bool {
						return a.second > b.second;
					}
				);
			}
		}

		// TODO: pad ou if the heap doesn't have k items
		return heap;
	}
};

#endif
