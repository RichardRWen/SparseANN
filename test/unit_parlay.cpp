#include <fstream>
#include <cstdint>
#include <cstdlib>
#include <math.h>
#include <numeric>
#include <vector>
#include <unordered_set>
#include <string>
#include <queue>
#include <utility>
#include <cassert>

#include <parlay/parallel.h>
#include <parlay/primitives.h>
#include <parlay/sequence.h>
#include <parlay/slice.h>

int main(int argc, char **argv) {
	parlay::sequence<uint32_t> seq;
	for (int i = 0; i < 10; i++) {
		seq.push_back((uint32_t)(rand() % 100));
		std::cout << seq[i] << " ";
	}
	std::cout << std::endl;

	uint64_t sum = parlay::reduce(parlay::make_slice(seq.begin(), seq.end()), parlay::binary_op([] (uint64_t acc, uint32_t x) -> uint64_t {
				return (acc + ((uint64_t)x << 32));
			}, (uint64_t)0));
	std::cout << sum << std::endl;
}
