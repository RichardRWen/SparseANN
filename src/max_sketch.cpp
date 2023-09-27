#include <stdlib.h>

#include "../include/max_sketch.h"
#include "../include/hashutil.h"

static inline uint64_t max_sketch_hash(int key, unsigned int seed) {
	return MurmurHash64A(&key, sizeof(key), seed);
}

void max_sketch_init(max_sketch *sketch, size_t vector_len, size_t sketch_len, size_t num_hash_fcns) {
	sketch->vector_len = vector_len;
	sketch->sketch_len = sketch_len;
	sketch->quantized_vectors = new std::vector<int*>();
	sketch->num_hash_fcns = num_hash_fcns;
	sketch->hash_fcn_seeds = new unsigned int[num_hash_fcns];
	for (int i = 0; i < num_hash_fcns; i++) {
		sketch->hash_fcn_seeds[i] = rand();
	}
}

void max_sketch_free(max_sketch *sketch) {
	for (int i = 0; i < sketch->quantized_vectors->size(); i++) {
		if ((*sketch->quantized_vectors)[i] != NULL) delete[] (*sketch->quantized_vectors)[i];
		(*sketch->quantized_vectors)[i] = NULL;
	}
	delete sketch->quantized_vectors;
}

void max_sketch_insert(max_sketch *sketch, int *vector) {
	int *quantized = new int[sketch->vector_len];
	bzero(quantized, sizeof(int) * sketch->sketch_len);
	for (int i = 0; i < sketch->vector_len; i++) {
		for (int j = 0; j < sketch->num_hash_fcns; j++) {
			size_t index = max_sketch_hash(i, sketch->hash_fcn_seeds[j]) % sketch->sketch_len;
			if (vector[i] > quantized[index]) quantized[index] = vector[i];
		}
	}
	sketch->quantized_vectors->push_back(quantized);
}
