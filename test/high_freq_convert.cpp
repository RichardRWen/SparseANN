#include <iostream>
#include <cstdio>
#include <fstream>
#include <strings.h>
#include <string>
#include <vector>
#include <algorithm>
#include <cassert>

int main(int argc, char **argv) {
	if (argc < 4) {
		std::cout << "Usage: " << argv[0] << " [high freq recording] [any number of .csr files and their outfiles]" << std::endl;
		exit(0);
	}
	if (argc % 2 == 1) {
		std::cout << "Every infile must be followed by an outfile" << std::endl;
		exit(0);
	}

	std::ifstream reader(argv[1]);
	if (!reader.is_open()) {
		printf("Could not open file\n");
		exit(0);
	}

	// READ COORD MAP
	uint64_t num_coords, num_dims;
	reader.read((char*)(&num_coords), sizeof(unsigned int));
	reader.read((char*)(&num_dims), sizeof(unsigned int));

	std::vector<unsigned int> coord_map(num_dims, num_coords);
	unsigned int index = 0;
	for (int i = 0; i < num_coords; i++) {
		unsigned int temp_coord;
		reader.read((char*)(&temp_coord), sizeof(unsigned int));
		coord_map[temp_coord] = index++;
	}
	reader.close();
	std::cout << "Loaded filter of " << num_coords << " out of " << num_dims << " coords" << std::endl;

	// CONVERT CSRs
	for (int i = 2; i < argc; i += 2) {
		std::ifstream csr_reader(argv[i]);
		std::ofstream csr_writer(argv[i + 1]);
		if (!csr_reader.is_open() || !csr_writer.is_open()) {
			std::cout << "Could not open file pair " << i / 2 << std::endl;
			exit(0);
		}

		std::cout << "Applying filter to " << argv[i] << std::endl;
		uint64_t vecs, dims, vals;
		csr_reader.read((char*)(&vecs), sizeof(uint64_t));
		csr_reader.read((char*)(&dims), sizeof(uint64_t));
		csr_reader.read((char*)(&vals), sizeof(uint64_t));

		csr_writer.write((char*)(&vecs), sizeof(uint64_t));
		csr_writer.write((char*)(&num_coords), sizeof(uint64_t));
		csr_writer.write((char*)(&vals), sizeof(uint64_t));

		std::ifstream index_reader(argv[i]);
		index_reader.seekg((vecs + 4) * sizeof(uint64_t));
		std::ifstream value_reader(argv[i]);
		value_reader.seekg((vecs + 4) * sizeof(uint64_t) + vals * sizeof(unsigned int));

		std::vector<uint64_t> endptrs;
		std::vector<unsigned int> indices;
		std::vector<float> values;
		endptrs.push_back(0);

		uint64_t indptr_start, indptr_end, new_endptr = 0;
		csr_reader.read((char*)(&indptr_end), sizeof(uint64_t));
		for (int i = 0; i < vecs; i++) {
			indptr_start = indptr_end;
			csr_reader.read((char*)(&indptr_end), sizeof(uint64_t));
			unsigned int temp_index;
			float temp_value;
			for (; indptr_start < indptr_end; indptr_start++) {
				index_reader.read((char*)(&temp_index), sizeof(unsigned int));
				value_reader.read((char*)(&temp_value), sizeof(float));
				if (coord_map[temp_index] < num_coords) {
					indices.push_back(coord_map[temp_index]);
					values.push_back(temp_value);
					new_endptr++;
				}
			}
			endptrs.push_back(new_endptr);
		}

		std::cout << "Writing results to " << argv[i + 1] << std::endl;
		assert(endptrs.size() == vecs + 1);
		assert(indices.size() == values.size());
		for (int i = 0; i < vecs + 1; i++) {
			csr_writer.write((char*)(&endptrs[i]), sizeof(uint64_t));
		}
		for (int i = 0; i < indices.size(); i++) {
			csr_writer.write((char*)(&indices[i]), sizeof(unsigned int));
		}
		for (int i = 0; i < values.size(); i++) {
			csr_writer.write((char*)(&values[i]), sizeof(float));
		}
		
		csr_writer.close();
		csr_reader.close();
		index_reader.close();
		value_reader.close();
	}

	return 0;
}
