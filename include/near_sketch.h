#include <fstream>
#include <vector>
#include <string.h>

template <typename value_type>
class near_sketch {
public:
	size_t full_len;
	size_t comp_len;
	double scale;
	std::vector<value_type*> comp_vectors;

	near_sketch(const size_t _full_len, const size_t _comp_len) {
		full_len = _full_len;
		scale = comp_len = _comp_len;
		scale /= full_len;
	}
	near_sketch(const char *filename, const char *filetype, const size_t _num_to_read, const size_t _comp_len) {
		comp_len = _comp_len;
		if (strcmp(filetype, "csr") == 0) {
			std::ifstream indptr_reader(filename);

			uint64_t num_vecs, num_dims, num_vals;
			indptr_reader.read((char*)(&num_vecs), sizeof(num_vecs));
			indptr_reader.read((char*)(&num_dims), sizeof(num_dims));
			indptr_reader.read((char*)(&num_vals), sizeof(num_vals));
			size_t startof_indptrs = sizeof(num_vecs) + sizeof(num_dims) + sizeof(num_vals);
			size_t startof_indices = startof_indptrs + (num_vecs + 1) * sizeof(uint64_t);
			size_t startof_data    = startof_indices + num_vals * sizeof(uint32_t);

			std::ifstream index_reader(filename);
			index_reader.seekg(startof_indices);
			std::ifstream value_reader(filename);
			value_reader.seekg(startof_data);

			uint64_t indptr_end, indptr_curr = 0;
			indptr_reader.read((char*)(&indptr_end), sizeof(indptr_end));
			
			size_t num_to_read = (_num_to_read < num_vecs ? _num_to_read : num_vecs);
			unsigned int index_temp;
			value_type data_temp;
			uint64_t comp_index_temp;
			for (size_t i = 0; i < num_to_read; i++) {
				indptr_reader.read((char*)(&indptr_end), sizeof(indptr_end));
				value_type *compressed = new value_type[comp_len];
				for (; indptr_curr < indptr_end; indptr_curr++) {
					index_reader.read((char*)(&index_temp), sizeof(index_temp));
					value_reader.read((char*)(&data_temp), sizeof(data_temp));
					comp_index_temp = (uint64_t)(scale * index_temp);
					if (data_temp > compressed[comp_index_temp]) compressed[comp_index_temp] = data_temp;
				}
				comp_vectors.push_back(compressed);
			}

			indptr_reader.close();
			index_reader.close();
			value_reader.close();
		}
		else {
			full_len = comp_len;
		}
	}
	near_sketch(const char *filename, const char *filetype, const size_t _comp_len) : near_sketch(filename, filetype, -1ull, _comp_len) {}

	void insert(value_type *vector) {
		value_type *compressed = new value_type[comp_len];
		for (int i = 0; i < full_len; i++) {
			uint64_t index = (uint64_t)(scale * i);
			if (vector[i] > compressed[index]) compressed[index] = vector[i];
		}
		comp_vectors.push_back(compressed);
	}

	value_type get(size_t vector_index, size_t coord_index) {
		return comp_vectors[vector_index][(uint64_t)(scale * coord_index)];
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
