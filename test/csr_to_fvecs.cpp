#include <iostream>
#include <cstdio>
#include <fstream>
#include <strings.h>

int main(int argc, char **argv) {
	if (argc < 3) {
		printf("Usage: [csr infile] [fvecs outfile] [optional sample size]\n");
		exit(0);
	}
	size_t vecs_to_read = -1ull;
	if (argc > 3) {
		vecs_to_read = strtoull(argv[3], NULL, 10);
	}

	std::ifstream indptr_reader(argv[1]);
	std::ifstream index_reader(argv[2]);
	std::ifstream value_reader(argv[3]);
	if (!indptr_reader.is_open()) {
		std::cout << "Could not open file " << argv[1] << " for reading" << std::endl;
		exit(0);
	}

	std::ofstream writer(argv[2]);
	if (!writer.is_open()) {
		std::cout << "Could not open file " << argv[2] << " for writing" << std::endl;
		exit(0);
	}

	// READ METADATA
	uint64_t num_vecs, num_dims, num_vals;
	indptr_reader.read((char*)(&num_vecs), sizeof(num_vecs));
	indptr_reader.read((char*)(&num_dims), sizeof(num_dims));
	indptr_reader.read((char*)(&num_vals), sizeof(num_vals));
	index_reader.seekg((num_vecs + 4) * sizeof(uint64_t));
	value_reader.seekg((num_vecs + 4) * sizeof(uint64_t) + num_vals * sizeof(uint32_t));

	// READ VECTOR
	writer.write((char*)(&num_vecs), sizeof(uint32_t));
	writer.write((char*)(&num_dims), sizeof(uint32_t));

	float *buffer = new float[num_dims];
	if (vecs_to_read > num_vecs) vecs_to_read = num_vecs;
	uint64_t indptr_start, indptr_end;
	indptr_reader.read((char*)(&indptr_end), sizeof(uint64_t));
	for (int i = 0; i < vecs_to_read; i++) {
		indptr_start = indptr_end;
		bzero(buffer, num_dims * sizeof(float));
		for (int j = indptr_start; j < indptr_end; j++) {
			uint32_t index;
			index_reader.read((char*)(&index), sizeof(uint32_t));
			value_reader.read((char*)(&buffer[index]), sizeof(float));
		}
		writer.write((char*)buffer, num_dims * sizeof(float));
	}

	return 0;
}
