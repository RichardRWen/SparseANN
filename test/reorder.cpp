#include <iostream>
#include <cstdio>
#include <fstream>
#include <strings.h>
#include <string>
#include <vector>
#include <algorithm>
#include <cassert>

#include "../include/coord_order.h"

int main(int argc, char **argv) {
	if (argc < 4) {
		std::cout << "Usage: " << argv[0] << " [csr infile] [sample size] [outfile]" << std::endl;
		exit(0);
	}

	coord_order order(argv[1], "csr", std::stoull(argv[2]));
	order.reorder(20, true);
	
	order.write_to_file(argv[3]);

	return 0;
}
