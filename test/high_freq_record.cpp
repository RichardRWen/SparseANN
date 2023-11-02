#include <iostream>
#include <cstdio>
#include <fstream>
#include <strings.h>
#include <string>
#include <vector>
#include <algorithm>

struct coord_info {
	unsigned int id;
	unsigned int val;
};

int main(int argc, char **argv) {
	if (argc < 4) {
		std::cout << "Usage: " << argv[0] << " [.csr file] [outfile] [num coords]" << std::endl;
		exit(0);
	}
	unsigned int num_coords = std::stoi(argv[3]);

	std::ifstream indptr_reader(argv[1]);
	if (!indptr_reader.is_open()) {
		std::cout << "Could not open csr file" << std::endl;
		exit(0);
	}

	// READ METADATA
	uint64_t num_vecs, num_dims, num_vals;
	indptr_reader.read((char*)(&num_vecs), sizeof(num_vecs));
	indptr_reader.read((char*)(&num_dims), sizeof(num_dims));
	indptr_reader.read((char*)(&num_vals), sizeof(num_vals));
	std::ifstream index_reader(argv[1]);
	std::ifstream value_reader(argv[1]);
	index_reader.seekg((num_vecs + 4) * sizeof(uint64_t));
	value_reader.seekg((num_vecs + 4) * sizeof(uint64_t) + num_vals * sizeof(unsigned int));
	if (num_dims < num_coords) num_coords = num_dims;

	// READ VECTOR
	std::vector<coord_info> coord_counts(num_dims, {0, 0});
	for (int i = 0; i < num_dims; i++) {
		coord_counts[i].id = i;
	}

	uint64_t indptr_start, indptr_end;
	indptr_reader.read((char*)(&indptr_end), sizeof(uint64_t));
	for (int i = 0; i < num_vals; i++) {
		unsigned int temp_index;
		index_reader.read((char*)(&temp_index), sizeof(unsigned int));
		coord_counts[temp_index].val++;
	}
	std::sort(coord_counts.begin(), coord_counts.end(), [](coord_info a, coord_info b) {
		return a.val > b.val;
	});

	indptr_reader.close();
	index_reader.close();
	value_reader.close();

	std::ofstream writer(argv[2]);
	if (!writer.is_open()) {
		std::cout << "Could not open outfile" << std::endl;
		exit(0);
	}

	writer.write((char*)(&num_coords), sizeof(unsigned int));
	writer.write((char*)(&num_dims), sizeof(unsigned int));
	for (int i = 0; i < num_coords; i++) {
		writer.write((char*)(&coord_counts[i].id), sizeof(unsigned int));
	}
	std::cout << "Wrote to " << argv[2] << std::endl;

	std::cout << "Head:" << std::endl;
	for (int i = 0; i < 10; i++) {
		std::cout << coord_counts[i].id << "\t" << coord_counts[i].val << std::endl;
	}

	return 0;
}
