#include <fstream>
#include <cstdint>
#include <math.h>

struct coord_order {
	size_t dims;
	unsigned int *order;

	coord_order(const unsigned int _dims) : dims(_dims) {
		order = new unsigned int[dims];
		for (unsigned int i = 0; i < dims; i++) {
			order[i] = i;
		}
	}

	double log_gap_cost(unsigned int *ordered_indices, size_t num_nonzeros) { // Note: assumes input indices are sorted by new order
		double cost = 0;
		for (int i = 0; i < num_nonzeros - 1; i++) {
			cost += log(ordered_indices[i + 1] - ordered_indices[i]);
		}
		return cost;
	}
	double total_log_gap_cost(uint64_t *indptrs, unsigned int *indices, size_t num_vecs) {
		double total_cost = 0;
		for (int i = 0; i < num_vecs; i++) {
			total_cost += log_gap_cost(&indices[indptrs[i]], indptrs[i + 1] - indptrs[i]);
		}
		return total_cost;
	}

	void reorder(const char *filename) {
		std::ifstream reader(filename);
		if (!reader.is_open()) return;

		uint64_t num_vecs, num_dims, num_vals;
		reader.read((char*)(&num_vecs), sizeof(uint64_t));
		reader.read((char*)(&num_dims), sizeof(uint64_t));
		reader.read((char*)(&num_vals), sizeof(uint64_t));

		uint64_t     *indptrs = malloc((num_vecs + 1) * sizeof(uint64_t));
		unsigned int *indices = malloc(num_vals * sizeof(unsigned int));
		float        *values  = malloc(num_vals * sizeof(float));
		reader.read((char*)indptrs, (num_vecs + 1) * sizeof(uint64_t));
		reader.read((char*)indices, num_vals * sizeof(unsigned int));
		reader.read((char*)values, num_vals * sizeof(float));

		// loop here, rearranging coords and to minimize the return value of total_log_gap_cost
	}

	unsigned int operator [] (const size_t i) {
		return order[i];
	}
};
