#include <iostream>
#include <cassert>

#include "../include/max_sketch.h"

int main(int argc, char **argv) {
	if (argc < 3) {
		std::cout << "Usage: " << argv[0] << " [path to .csr] [path to outfile]" << std::endl;
		exit(0);
	}

	max_sketch<float> sketch(argv[1], "csr", 100, 200, 3);
	assert(sketch.comp_vectors.size() == 100);
	
	sketch.write_to_file(argv[2]);

	sketch.free();

	return 0;
}
