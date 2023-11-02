#include <iostream>
#include <cstdio>
#include <fstream>
#include <vector>
#include <strings.h>
#include <utility>

int main(int argc, char **argv) {
	if (argc <= 1) {
		std::cout << "Please provide csr file to write" << std::endl;
		exit(0);
	}

	std::ofstream writer(argv[1]);
	if (!writer.is_open()) {
		printf("Could not open file\n");
		exit(0);
	}

	uint64_t num_vecs, num_dims, num_vals = 0;
	int input_type = 0;
	std::cout << "Number of vectors: " << std::flush;
	std::cin >> num_vecs;
	std::cout << "Number of dimensions: " << std::flush;
	std::cin >> num_dims;
	while (input_type <= 0) {
		if (input_type < 0) std::cout << "Supported input formats are: vec" << std::endl;
		std::cout << "Input format (vec/csr): " << std::flush;
		std::string s;
		std::cin >> s;
		if (!s.compare("vec")) {
			input_type = 1;
		}
		else {
			input_type = -1;
		}
	}

	std::vector<std::vector<std::pair<uint32_t, float>>> vecs(num_vecs);
	for (int i = 0; i < num_vecs; i++) {
		uint32_t pos, j = 0;
		std::string s, token;
		std::cout << "Vector " << i << ":\t" << std::flush;
		do {
			getline(std::cin, s);
		} while (s.size() == 0);
		while ((pos = s.find(" ")) != std::string::npos && j < num_dims - 1) {
			token = s.substr(0, pos);
			float val = std::stof(token);
			s.erase(0, pos + 1);
			if (val != 0) vecs[i].push_back(std::make_pair(j, val));
			j++;
		}
		if (pos == std::string::npos) {
			float val = std::stof(s);
			if (val != 0) vecs[i].push_back(std::make_pair(j, val));
		}
		else {
			token = s.substr(0, pos);
			float val = std::stof(token);
			if (val != 0) vecs[i].push_back(std::make_pair(j, val));
		}
		std::cout << "Wrote " << vecs[i].size() << " values" << std::endl;
	}

	for (int i = 0; i < num_vecs; i++) num_vals += vecs[i].size();
	
	writer.write((char*)(&num_vecs), sizeof(uint64_t));
	writer.write((char*)(&num_dims), sizeof(uint64_t));
	writer.write((char*)(&num_vals), sizeof(uint64_t));
	
	uint64_t indptr = 0;
	writer.write((char*)(&indptr), sizeof(uint64_t));
	for (int i = 0; i < num_vecs; i++) {
		indptr += vecs[i].size();
		writer.write((char*)(&indptr), sizeof(uint64_t));
	}
	for (int i = 0; i < num_vecs; i++) {
		for (int j = 0; j < vecs[i].size(); j++) {
			writer.write((char*)(&vecs[i][j].first), sizeof(uint32_t));
		}
	}
	for (int i = 0; i < num_vecs; i++) {
		for (int j = 0; j < vecs[i].size(); j++) {
			writer.write((char*)(&vecs[i][j].second), sizeof(float));
		}
	}

	writer.close();

	return 0;
}
