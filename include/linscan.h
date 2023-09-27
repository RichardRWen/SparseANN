#include <stdlib.h>
#include <cstdint>
#include <vector>

struct inverted_value {
	uint64_t id;
	uint64_t value;
};

struct inverted_list {
	uint64_t list_id;
	std::vector<inverted_value> *values;
};

struct inverted_index {
	inverted_list *lists;
	unsigned int num_lists;
};

int inverted_index_init(inverted_index *inv_index, unsigned int num_lists);
void inverted_index_free(inverted_index *inv_index);

int inverted_index_insert(inverted_index *inv_index, int *vector, uint64_t vector_id);
std::vector<inverted_value> inverted_index_query(inverted_index *inv_index, int *vector, int k);
