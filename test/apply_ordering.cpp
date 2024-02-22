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
		std::cout << "Usage: " << argv[0] << " [order infile] [csr infile] [csr outfile]" << std::endl;
		exit(0);
	}

	coord_order order(argv[1], "ord");
	
	order.apply_ordering(argv[2], argv[3]);

	//forward_index fwd_index<float>(argv[2], "csr");
	//fwd_index.reorder(order.order_map);

	return 0;
}
