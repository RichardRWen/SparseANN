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
		std::cout << "Usage: " << argv[0] << " [csr infile] [trec outfile] [sample size]" << std::endl;
		exit(0);
	}

	std::ifstream reader(argv[1]);
	if (!reader.is_open()) {
		printf("Could not open file\n");
		exit(0);
	}

	uint64_t num_vecs, num_dims, num_vals;
	reader.read((char*)(&num_vecs), sizeof(uint64_t));
	reader.read((char*)(&num_dims), sizeof(uint64_t));
	reader.read((char*)(&num_vals), sizeof(uint64_t));

	std::vector<unsigned int> sample(num_vecs, 0);
	for (int i = 0; i < num_vecs; i++) sample[i] = i;
	std::random_shuffle(sample.begin(), sample.end());

	unsigned int sample_size = std::stoi(argv[3]);
	if (sample_size > num_vecs) sample_size = num_vecs;

	std::ofstream writer(argv[2]);
	for (int i = 0; i < sample_size; i++) {
		reader.seekg((sample[i] + 3) * sizeof(uint64_t));
		uint64_t indptr_start, indptr_end;
		reader.read((char*)(&indptr_start), sizeof(uint64_t));
		reader.read((char*)(&indptr_end), sizeof(uint64_t));

		reader.seekg((num_vecs + 4) * sizeof(uint64_t) + indptr_start * sizeof(unsigned int));
		writer << "<top>\n" << "<num> Number: " << sample[i] << "\n" << "<desc>\n";
		unsigned int temp_index;
		for (; indptr_start < indptr_end; indptr_start++) {
			reader.read((char*)(&temp_index), sizeof(unsigned int));
			writer << temp_index << " ";
		}
		writer << "\n" << "</top>\n\n";
	}

	reader.close();
	writer.close();

	return 0;
}
