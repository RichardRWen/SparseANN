#include <fstream>
#include <cstdint>
#include <cstdlib>
#include <math.h>
#include <numeric>
#include <vector>
#include <unordered_set>
#include <atomic>
#include <string>
#include <queue>
#include <utility>
#include <cassert>
#include <thread> // for sleep_for
#include <chrono> // for ms

#include <parlay/parallel.h>
#include <parlay/primitives.h>
#include <parlay/sequence.h>
#include <parlay/slice.h>
#include <parlay/alloc.h>

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

	std::vector<std::atomic<int>> atomic_list;
	//atomic_list.emplace_back(1);

	auto uninit_seq = parlay::sequence<uint32_t>::uninitialized(10);

	// SAMPLE: having one output thread and one work thread
	parlay::sequence<uint32_t> tracking_seq(5, 0);
	parlay::par_do(
		[&tracking_seq] () {
			bool all_complete = false;
			while (!all_complete) {
				all_complete = true;
				std::cout << "\r";
				for (uint32_t complete : tracking_seq) {
					if (complete) std::cout << "1 ";
					else {
						std::cout << "0 ";
						all_complete = false;
					}
				}
				std::cout << std::flush;
				std::this_thread::sleep_for(std::chrono::milliseconds(100));
			}
			std::cout << std::endl;
		},
		[&tracking_seq] () {
			parlay::random_generator gen;
			std::uniform_real_distribution<double> dis(0, 1);
			auto random_shuffle = parlay::tabulate(tracking_seq.size(), [&] (size_t i) {
					auto r = gen[i];
					return dis(r);
				});
			auto completion_order = parlay::sort(
				parlay::tabulate(tracking_seq.size(),
					[&] (size_t i) -> uint32_t {
						return i;
					}),
				[&] (uint32_t a, uint32_t b) -> bool {
					return random_shuffle[a] < random_shuffle[b];
				});

			for (uint32_t rank : completion_order) {
				std::this_thread::sleep_for(std::chrono::milliseconds(500));
				tracking_seq[rank] = 1;
			}
		});
}
