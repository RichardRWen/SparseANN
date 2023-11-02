#include <cstdlib>
#include <cstdint>

uint64_t MurmurHash64A(const void *key, const int len, const unsigned int seed);

struct hasher_murmur64a {
	unsigned int seed;
	uint64_t operator () (const void *key, const int len) const {
		return MurmurHash64A(key, len, seed);
	}
	uint64_t operator () (const uint64_t key) const {
		return MurmurHash64A(&key, sizeof(uint64_t), seed);
	}
	uint64_t operator () (const uint32_t key) const {
		return MurmurHash64A(&key, sizeof(uint32_t), seed);
	}

	hasher_murmur64a() {
		seed = rand();
	}
	hasher_murmur64a(const unsigned int seed_in) {
		seed = seed_in;
	}
};

struct hash_table_node;
struct hash_table;
