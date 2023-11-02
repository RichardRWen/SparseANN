#include <iostream>
#include <fstream>
#include <cstdint>
#include <cassert>

#include "../include/linscan.h"

int k = 100;

int main(int argc, char **argv) {
	if (argc < 4) {
		std::cout << "Usage: " << argv[0] << " [path to inserts .csr] [path to queries .csr] [path to outfile]" << std::endl;
		exit(0);
	}

	std::ifstream query_reader(argv[2]);
	if (!query_reader.is_open()) {
		std::cout << "Could not read queries\n" << std::endl;
		exit(0);
	}
	std::ifstream index_reader(argv[2]);
	std::ifstream value_reader(argv[2]);

	std::ofstream writer(argv[3]);
	if (!writer.is_open()) {
		std::cout << "Could not open outfile\n" << std::endl;
		exit(0);
	}

	inverted_index<float, uint64_t> inv_index(argv[1], "csr");
	if (inv_index.num_lists == 0) {
		std::cout << "Could not read inserts\n" << std::endl;
	}

	uint64_t num_vecs, num_dims, num_vals;
	query_reader.read((char*)(&num_vecs), sizeof(uint64_t));
	query_reader.read((char*)(&num_dims), sizeof(uint64_t));
	query_reader.read((char*)(&num_vals), sizeof(uint64_t));
	
	index_reader.seekg((num_vecs + 4) * sizeof(uint64_t));
	value_reader.seekg((num_vecs + 4) * sizeof(uint64_t) + num_vals * sizeof(unsigned int));

	writer.write((char*)(&num_vecs), sizeof(uint64_t));
	writer.write((char*)(&k), sizeof(uint64_t));

	uint64_t indptr_start, indptr_end;
	query_reader.read((char*)(&indptr_end), sizeof(uint64_t));
	float *vector = new float[num_dims];
	unsigned int temp_index;
	float temp_value;
	for (int i = 0; i < num_vecs; i++) {
		indptr_start = indptr_end;
		query_reader.read((char*)(&indptr_end), sizeof(uint64_t));
		bzero(vector, num_dims * sizeof(float));
		for (; indptr_start < indptr_end; indptr_start++) {
			index_reader.read((char*)(&temp_index), sizeof(unsigned int));
			value_reader.read((char*)(&temp_value), sizeof(float));
			vector[temp_index] = temp_value;
		}
		auto neighbors = inv_index.neighbors(vector, k);
		for (int j = 0; j < neighbors.size(); j++) {
			writer.write((char*)(&neighbors[j].id), sizeof(uint64_t));
			writer.write((char*)(&neighbors[j].value), sizeof(float));
		}
		if (i < 10) {
			std::cout << "Query " << i << std::endl;
			for (int j = 0; j < neighbors.size(); j++) {
				std::cout << "(" << neighbors[j].id << ", " << neighbors[j].value << ")\t";
			}
			std::cout << std::endl;
		}
	}

	writer.close();
	query_reader.close();
	index_reader.close();
	value_reader.close();

	return 0;
}
