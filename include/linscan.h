#include <stdlib.h>
#include <cstdint>
#include <vector>
#include <unordered_map>
#include <cstring>
#include <fstream>
#include <algorithm>

template <typename val_type, typename id_type = uint64_t>
class inverted_index {
	public:
	struct posting_list {
		struct posted_value {
			id_type id;
			val_type value;

			posted_value() : id(0), value(0) {}
			posted_value(id_type id, val_type value) : id(id), value(value) {}

		static inline bool greater(const posted_value a, const posted_value b) {
			return a.value > b.value;
		}
		};

		std::vector<posted_value> values;
	};

	unsigned int num_lists;
	posting_list *lists;

	inverted_index(const unsigned int num_lists) : num_lists(num_lists) {}
	inverted_index(const char *filename, const char *filetype, const size_t _num_to_read = -1ULL) {
		if (strcmp(filetype, "csr") == 0) {
			std::ifstream indptr_reader(filename);
			if (!indptr_reader.is_open()) {
				num_lists = 0;
				lists = NULL;
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
			lists = new posting_list[num_dims];

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
					typename posting_list::posted_value new_value(i_vecs, data);
					lists[index].values.push_back(new_value);
				}
			}
		}
		else {
			num_lists = 0;
			lists = NULL;
		}
	}
	
	int insert(id_type vector_id, val_type *vector) {
		int num_coords = 0;
		for (int i = 0; i < num_lists; i++) {
			if (vector[i]) {
				typename posting_list::posted_value new_value(vector_id, vector[i]);
				lists[i].values.push_back(new_value);
				num_coords++;
			}
		}
		return num_coords;
	}

	std::vector<typename posting_list::posted_value> neighbors(val_type *vector, int k) {
		// Calculate the inner products for all documents that share nonzero coordinates with the query vector
		std::unordered_map<id_type, val_type> inner_products;
		std::pair<typename std::unordered_map<id_type, val_type>::iterator, bool> insert_result;
		for (int i = 0; i < num_lists; i++) {
			if (vector[i]) {
				for (int j = 0; j < lists[i].values.size(); j++) {
					val_type partial_inner_product = lists[i].values[j].value * vector[i];
					std::pair<id_type, val_type> temp_pair(lists[i].values[j].id, partial_inner_product);
					insert_result = inner_products.insert(temp_pair);
					if (!insert_result.second) {
						insert_result.first->second += partial_inner_product;
					}
				}
			}
		}

		// Use heap to find top k inner products
		std::vector<typename posting_list::posted_value> heap;
		for (auto it = inner_products.begin(); it != inner_products.end(); it++) {
			typename posting_list::posted_value next_value;
			next_value.id = it->first;
			next_value.value = it->second;

			heap.push_back(next_value);
			std::push_heap(heap.begin(), heap.end(), posting_list::posted_value::greater);
			if (heap.size() > k) {
				std::pop_heap(heap.begin(), heap.end(), posting_list::posted_value::greater);
			}
		}

		return heap;
	}
};
