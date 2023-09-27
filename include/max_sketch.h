#include <vector>

struct max_sketch {
	size_t vector_len;
	size_t sketch_len;
	std::vector<int*> *quantized_vectors;
	size_t num_hash_fcns;
	unsigned int *hash_fcn_seeds;
};
