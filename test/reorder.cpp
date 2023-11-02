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
	if (argc < 3) {
		std::cout << "Usage: " << argv[0] << " [csr infile] [sample size]" << std::endl;
		exit(0);
	}

	coord_order order(argv[1], "csr", std::stoi(argv[2]));
	order.reorder(20, true);

	return 0;
}
