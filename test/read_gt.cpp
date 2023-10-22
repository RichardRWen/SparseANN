#include <iostream>
#include <cstdio>
#include <fstream>
#include <strings.h>

int main(int argc, char **argv) {
	if (argc <= 1) {
		printf("Please provide gt file to read\n");
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

	unsigned int n, d;
	reader.read((char*)(&n), sizeof(unsigned int));
	reader.read((char*)(&d), sizeof(unsigned int));

	std::cout << "n: " << n << std::endl;
	std::cout << "d: " << d << std::endl;

	// CLEANUP
	reader.close();

	return 0;
}
