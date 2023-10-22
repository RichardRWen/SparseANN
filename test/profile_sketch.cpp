#include <iostream>
#include <cstdio>
#include <fstream>
#include <strings.h>
#include <set>

int main(int argc, char **argv) {
	if (argc < 3) {
		std::cout << "Usage: " << argv[0] << " [Original .csr file] [Compressed .fvecs file] [Optional size of head]" << std::endl;
		exit(0);
	}
	size_t vecs_to_read = -1;
	if (argc > 3) {
		vecs_to_read = strtoull(argv[3], NULL, 10);
	}

	std::ifstream csr_reader(argv[1]);
	if (!csr_reader.is_open()) {
		printf("Could not open csr file\n");
		exit(0);
	}
	std::ifstream csr_data_reader(argv[1]);

	std::ifstream vec_reader(argv[2]);
	if (!vec_reader.is_open()) {
		printf("Could not open fvecs file\n");
		exit(0);
	}

	// READ CSR METADATA
	uint64_t num_full_vecs, num_full_dims, num_full_vals;
	csr_reader.read((char*)(&num_full_vecs), sizeof(num_full_vecs));
	csr_reader.read((char*)(&num_full_dims), sizeof(num_full_dims));
	csr_reader.read((char*)(&num_full_vals), sizeof(num_full_vals));

	if (num_full_vecs < vecs_to_read) vecs_to_read = num_full_vecs;

	std::cout << "Num full vecs: " << num_full_vecs << std::endl;
	std::cout << "Num full dims: " << num_full_dims << std::endl;

	// READ VECS METADATA
	unsigned int num_comp_vecs, num_comp_dims;
	vec_reader.read((char*)(&num_comp_vecs), sizeof(num_comp_vecs));
	vec_reader.read((char*)(&num_comp_dims), sizeof(num_comp_dims));

	if (num_comp_vecs < vecs_to_read) vecs_to_read = num_comp_vecs;

	std::cout << "Num comp vecs: " << num_comp_vecs << std::endl;
	std::cout << "Num comp dims: " << num_comp_dims << std::endl;

	// READ AND PROFILE VECTORS
	double preservation = 0; // The portion of unique values preserved in the compressed vectors
	double weight_pres = 0;
	double redundancy = 0; // The portion of coordinates that are actually used (nonzeros)
	double zeros = 0;

	float temp_data, temp_weight, temp_acc_weight, temp_real_weight;
	std::set<float> unique_set;
	uint64_t temp_indptr, temp_indptr_prev;
	csr_reader.read((char*)(&temp_indptr), sizeof(temp_indptr));
	csr_data_reader.seekg(3 * sizeof(uint64_t) + (num_full_vecs + 1) * sizeof(uint64_t) + num_full_vals * sizeof(unsigned int));
	for (int i = 0; i < vecs_to_read; i++) {
		unique_set.clear();
		temp_acc_weight = temp_real_weight = 0;
		temp_indptr_prev = temp_indptr;
		csr_reader.read((char*)(&temp_indptr), sizeof(temp_indptr));
		for (int j = 0; j < num_comp_dims; j++) {
			vec_reader.read((char*)(&temp_data), sizeof(temp_data));
			if (temp_data != 0 && unique_set.find(temp_data) == unique_set.end()) {
				unique_set.insert(temp_data);
			}
			else if (temp_data == 0) {
				zeros++;
			}
			else {
				redundancy++;
			}
		}
		preservation += (double)unique_set.size() / (temp_indptr - temp_indptr_prev);
		for (int j = temp_indptr_prev; j < temp_indptr; j++) {
			csr_data_reader.read((char*)(&temp_weight), sizeof(float));
			if (unique_set.find(temp_weight) != unique_set.end()) temp_acc_weight += temp_weight;
			temp_real_weight += temp_weight;
		}
		weight_pres += temp_acc_weight / temp_real_weight;
	}
	preservation /= vecs_to_read;
	weight_pres /= vecs_to_read;
	redundancy /= (vecs_to_read * num_comp_dims);
	zeros /= (vecs_to_read * num_comp_dims);

	csr_reader.close();
	vec_reader.close();

	std::cout << "Value Preservation:  " << preservation << std::endl;
	std::cout << "Weight Preservation: " << weight_pres << std::endl;
	std::cout << "Redundancy:          " << redundancy << std::endl;
	std::cout << "Unused dims:         " << zeros << std::endl;

	return 0;
}
