#include <iostream>
#include <cassert>

#include "../include/near_sketch.h"

int main(int argc, char **argv) {
	if (argc < 4) {
		std::cout << "Usage: " << argv[0] << " [path to .csr] [path to outfile] [comp dims (eg 200)]" << std::endl;
		exit(0);
	}

	uint64_t comp_len = strtoull(argv[3], NULL, 10);

	near_sketch<float> sketch(argv[1], "csr", comp_len);
	
	if (sketch.write_to_file(argv[2])) std::cout << "Wrote to file " << argv[2] << std::endl;
	else std::cout << "Unable to write to file" << std::endl;

	return 0;
}
