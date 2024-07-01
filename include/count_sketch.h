#include <fstream>
#include <vector>
#include <string.h>

#include "hashutil.h"

template <typename val_t>
class count_sketch {
public:
	size_t full_len;
	size_t comp_len;
    parlay::sequence<parlay::sequence<val_t>> vectors;

	max_sketch(const size_t _full_len, const size_t _comp_len, const size_t _num_hash_fcns) {
		full_len = _full_len;
		comp_len = _comp_len;
		num_hash_fcns = (_num_hash_fcns == 0 ? 1 : _num_hash_fcns);
		hash_fcn_seeds = new unsigned int[num_hash_fcns];
		for (int i = 0; i < num_hash_fcns; i++) hash_fcn_seeds[i] = rand();
	}
	max_sketch(const char *filename, const char *filetype, const size_t _num_to_read, const size_t _comp_len, const size_t _num_hash_fcns) {
		comp_len = _comp_len;
		num_hash_fcns = (_num_hash_fcns == 0 ? 1 : _num_hash_fcns);
		hash_fcn_seeds = new unsigned int[num_hash_fcns];
		for (int i = 0; i < num_hash_fcns; i++) hash_fcn_seeds[i] = rand();
		if (strcmp(filetype, "csr") == 0) {
			std::ifstream reader_indptrs(filename);

			uint64_t num_vecs, num_dims, num_vals;
			reader_indptrs.read((char*)(&num_vecs), sizeof(num_vecs));
			reader_indptrs.read((char*)(&num_dims), sizeof(num_dims));
			reader_indptrs.read((char*)(&num_vals), sizeof(num_vals));
			size_t startof_indptrs = sizeof(num_vecs) + sizeof(num_dims) + sizeof(num_vals);
			size_t startof_indices = startof_indptrs + (num_vecs + 1) * sizeof(uint64_t);
			size_t startof_data    = startof_indices + num_vals * sizeof(unsigned int);

			std::ifstream reader_indices(filename);
			reader_indices.seekg(startof_indices);
			std::ifstream reader_data(filename);
			reader_data.seekg(startof_data);

			uint64_t indptr_end, indptr_curr = 0;
			reader_indptrs.read((char*)(&indptr_end), sizeof(indptr_end));
			
			size_t num_to_read = (_num_to_read < num_vecs ? _num_to_read : num_vecs);
			unsigned int index_temp;
			value_type data_temp;
			size_t comp_index_temp;
			for (size_t i = 0; i < num_to_read; i++) {
				reader_indptrs.read((char*)(&indptr_end), sizeof(indptr_end));
				value_type *compressed = new value_type[comp_len];
				for (; indptr_curr < indptr_end; indptr_curr++) {
					reader_indices.read((char*)(&index_temp), sizeof(index_temp));
					reader_data.read((char*)(&data_temp), sizeof(data_temp));
					for (int j = 0; j < num_hash_fcns; j++) {
						comp_index_temp = hash(index_temp, hash_fcn_seeds[j]) % comp_len;
						if (data_temp > compressed[comp_index_temp]) compressed[comp_index_temp] = data_temp;
					}
				}
				comp_vectors.push_back(compressed);
			}

			reader_indptrs.close();
			reader_indices.close();
			reader_data.close();
		}
		else {
			full_len = comp_len;
			num_hash_fcns = 1;
			hash_fcn_seeds = new unsigned int[1];
			hash_fcn_seeds[0] = rand();
		}
	}
	max_sketch(const char *filename, const char *filetype, const size_t _comp_len, const size_t _num_hash_fcns) : max_sketch(filename, filetype, -1ull, _comp_len, _num_hash_fcns) {}

	static inline uint64_t hash(int key, unsigned int seed) {
		return MurmurHash64A(&key, sizeof(key), seed);
	}

	void insert(value_type *vector) {
		value_type *compressed = new value_type[comp_len];
		for (int i = 0; i < full_len; i++) {
			if (vector[i] <= 0) continue;
			for (int j = 0; j < num_hash_fcns; j++) {
				size_t index = hash(i, hash_fcn_seeds[j]) % comp_len;
				if (vector[i] > compressed[index]) compressed[index] = vector[i];
			}
		}
		comp_vectors.push_back(compressed);
	}

	value_type get(size_t vector_index, size_t coord_index) {
		value_type min = comp_vectors[vector_index][hash(coord_index, hash_fcn_seeds[0]) % comp_len];
		for (int i = 1; i < num_hash_fcns; i++) {
			size_t comp_index = hash(coord_index, hash_fcn_seeds[i]) % comp_len;
			if (comp_vectors[vector_index][comp_index] < min) min = comp_vectors[vector_index][comp_index];
		}
		return min;
	}

	void free() {
		for (int i = 0; i < comp_vectors.size(); i++) {
			if (comp_vectors[i] != NULL) delete[] comp_vectors[i];
			comp_vectors[i] = NULL;
		}
		delete[] hash_fcn_seeds;
	}

	bool write_to_file(char *filename) {
		std::ofstream writer(filename);
		if (!writer.is_open()) return false;
		
		unsigned int num_vecs = comp_vectors.size();
		writer.write((char*)(&num_vecs), sizeof(unsigned int));
		writer.write((char*)(&comp_len), sizeof(unsigned int));

		for (int i = 0; i < comp_vectors.size(); i++) {
			for (int j = 0; j < comp_len; j++) {
				writer.write((char*)(&comp_vectors[i][j]), sizeof(value_type));
			}
		}

		writer.close();
		return true;
	}
};
