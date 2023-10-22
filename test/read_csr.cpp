#include <iostream>
#include <cstdio>
#include <fstream>
#include <strings.h>

int main(int argc, char **argv) {
	if (argc <= 1) {
		printf("Please provide csr file to read\n");
		exit(0);
	}
	size_t vec_to_read = 0;
	if (argc > 2) {
		vec_to_read = strtoull(argv[2], NULL, 10);
	}

	std::ifstream reader(argv[1]);
	if (!reader.is_open()) {
		printf("Could not open file\n");
		exit(0);
	}

	// READ METADATA
	uint64_t num_vecs, num_dims, num_vals;
	reader.read((char*)(&num_vecs), sizeof(num_vecs));
	reader.read((char*)(&num_dims), sizeof(num_dims));
	reader.read((char*)(&num_vals), sizeof(num_vals));
	size_t startof_indptrs = sizeof(num_vecs) + sizeof(num_dims) + sizeof(num_vals);
	size_t startof_indices = startof_indptrs + (num_vecs + 1) * sizeof(uint64_t);
	size_t startof_data    = startof_indices + num_vals * sizeof(unsigned int);

	std::cout << "Num points: " << num_vecs << std::endl;
	std::cout << "Num dims:   " << num_dims << std::endl;
	std::cout << "Num vals:   " << num_vals << std::endl;

	// READ VECTOR
	unsigned int *index_buffer = new unsigned int[num_dims];
	float *data_buffer = new float[num_dims];
	size_t vec_num_nonzeros = 0;

	reader.seekg(startof_indptrs + vec_to_read * sizeof(uint64_t));
	uint64_t indptr_start, indptr_end;
	reader.read((char*)(&indptr_start), sizeof(uint64_t));
	reader.read((char*)(&indptr_end), sizeof(uint64_t));
	vec_num_nonzeros = indptr_end - indptr_start;

	reader.seekg(startof_indices + indptr_start * sizeof(unsigned int));
	for (int j = 0; j < vec_num_nonzeros; j++) {
			reader.read((char*)(&index_buffer[j]), sizeof(unsigned int));
	}

	bzero(data_buffer, num_dims * sizeof(float));
	reader.seekg(startof_data + indptr_start * sizeof(float));
	for (int j = 0; j < vec_num_nonzeros; j++) {
			reader.read((char*)(&data_buffer[index_buffer[j]]), sizeof(float));
	}

	// PRINT RESULTS
	std::cout << "Vector " << vec_to_read << ":" << std::endl;
	for (int i = 0; i < vec_num_nonzeros; i++) {
		std::cout << index_buffer[i] << ":\t" << data_buffer[index_buffer[i]] << std::endl;
	}
	std::cout << vec_num_nonzeros << " nonzero coordinates" << std::endl;

	// CLEANUP
	reader.close();
	delete[] index_buffer;
	delete[] data_buffer;

	return 0;
}
