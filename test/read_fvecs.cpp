#include <iostream>
#include <cstdio>
#include <fstream>
#include <strings.h>

int main(int argc, char **argv) {
	if (argc <= 1) {
		printf("Please provide fvecs file to read\n");
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
	uint64_t num_vecs, num_dims;
	reader.read((char*)(&num_vecs), sizeof(num_vecs));
	reader.read((char*)(&num_dims), sizeof(num_dims));

	if (num_dims == 0) {
		std::cout << "Error: num_dims = 0" << std::endl;
		reader.close();
		exit(0);
	}

	std::cout << "Num vecs: " << num_vecs << std::endl;
	std::cout << "Num dims: " << num_dims << std::endl;

	// READ AND PRINT VECTOR
	float temp_data;
	for (int i = 0; i < num_vecs; i++) {
		std::cout << i << ":\t";
		for (int j = 1; j < num_dims; j++) {
			reader.read((char*)(&temp_data), sizeof(temp_data));
			std::cout << temp_data << ",";
		}
		reader.read((char*)(&temp_data), sizeof(temp_data));
		std::cout << temp_data << std::endl;
	}

	reader.close();

	return 0;
}
