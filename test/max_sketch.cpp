#include <iostream>
#include <cassert>

#include "../include/max_sketch.h"

int main(int argc, char **argv) {
	if (argc < 5) {
		std::cout << "Usage: " << argv[0] << " [path to .csr] [path to outfile] [comp dims (eg 200)] [num hash fcns (eg 3)] [optional seed]" << std::endl;
		exit(0);
	}
	uint64_t seed = time(NULL);
	if (argc > 5) {
		seed = strtoull(argv[5], NULL, 10);
	}
	std::cout << "Generating hash functions with seed " << seed << std::endl;
	srand(seed);

	uint64_t comp_len = strtoull(argv[3], NULL, 10);
	uint64_t num_hash_fcns = strtoull(argv[4], NULL, 10);

	max_sketch<float> sketch(argv[1], "csr", comp_len, num_hash_fcns);
	
	if (sketch.write_to_file(argv[2])) std::cout << "Wrote to file " << argv[2] << std::endl;
	else std::cout << "Unable to write to file" << std::endl;

	sketch.free();

	return 0;
}
