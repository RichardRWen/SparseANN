#include <fstream>
#include <cstdint>
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

#include "hashutil.h"
#include "linscan.h"
#include "forward_index.h"

struct coord_order {
	std::vector<uint32_t> order;
	inverted_index<float, uint32_t> inv_index;
	forward_index<float, uint32_t> fwd_index;

	coord_order(const uint32_t dims) : order(dims, 0), inv_index(dims), fwd_index(dims) {
		std::iota(order.begin(), order.end(), (uint32_t)0);
	}
	coord_order(const char *filename, const char *filetype, const size_t _num_to_read = -1ULL) : inv_index(filename, filetype, _num_to_read), fwd_index(filename, filetype, _num_to_read) {
		order.resize(inv_index.num_lists);
		std::iota(order.begin(), order.end(), (uint32_t)0);
	}

	void initial_partition(const uint32_t start, const uint32_t end) {
		hasher_murmur64a hasher;
		std::vector<uint32_t> indices(end - start);
		std::iota(indices.begin(), indices.end(), 0);

		std::vector<uint32_t> shingles(end - start, -1);
		for (int i = 0; i < shingles.size(); i++) {
			// sort by shingles
			for (uint32_t j = 0; j < inv_index.lists[order[start + i]].values.size(); j++) {
				uint32_t hash = (uint32_t)hasher(inv_index.lists[order[start + i]].values[j].id);
				if (hash < shingles[i]) shingles[i] = hash;
			}
		}

		std::sort(indices.begin(), indices.end(), [shingles] (uint32_t a, uint32_t b) -> bool {
				return shingles[a] < shingles[b];
				});
		for (int i = 0; i < shingles.size(); i++) {
			shingles[i] = order[start + i];
		}
		for (int i = 0; i < shingles.size(); i++) {
			order[start + i] = shingles[indices[i]];
		}
	}

	void iterated_swap(const uint32_t start, const uint32_t end) {
		if (end - start < 2) return;
		uint32_t set_size = (end - start) / 2;
		uint32_t mid = (end + start) / 2;

		// split coords in half to form two sets
		std::unordered_set<uint32_t> set1;
		std::unordered_set<uint32_t> set2;
		for (int i = start; i < mid; i++) set1.insert(order[i]);
		for (int i = mid;   i < end; i++) set2.insert(order[i]);

		// calculate move gains for every coord
		std::vector<float> gains(end - start, 0);
		for (int i = start; i < end; i++) {
			// for each point that would be affected by this move (each point with this coord)
			for (int j = 0; j < inv_index.lists[order[i]].values.size(); j++) {
				uint32_t deg1 = 0, deg2 = 0;
				uint32_t point_id = inv_index.lists[order[i]].values[j].id;
				for (int k = 0; k < fwd_index.points[point_id].size(); k++) {
					if (set1.find(fwd_index.points[point_id][k].index) != set1.end()) deg1++;
					else if (set2.find(fwd_index.points[point_id][k].index) != set2.end()) deg2++;
				}

				gains[i - start] -= log((double)(set1.size()) / (deg1 + 1)) * (deg1 + 1);
				gains[i - start] -= log((double)(set2.size()) / (deg2 + 1)) * (deg2 + 1);
				if (i < mid) {
					assert(deg1 > 0);
					gains[i - start] += log((double)(set1.size()) / (deg1)) * (deg1);
					gains[i - start] += log((double)(set2.size()) / (deg2 + 2)) * (deg2 + 2);
				}
				else {
					assert(deg2 > 0);
					gains[i - start] += log((double)(set1.size()) / (deg1 + 2)) * (deg1 + 2);
					gains[i - start] += log((double)(set2.size()) / (deg2)) * (deg2);
				}
			}
		}

		// sort the two sets according to move gain
		std::vector<uint32_t> indices(end - start);
		std::iota(indices.begin(), indices.end(), (uint32_t)0);
		std::sort(indices.begin(), indices.begin() + set_size, [gains] (uint32_t a, uint32_t b) -> bool {
				return gains[a] < gains[b];
				});
		std::sort(indices.begin() + set_size, indices.end(),   [gains] (uint32_t a, uint32_t b) -> bool {
				return gains[a] < gains[b];
				});

		// perform swaps between the sets so long as they are profitable
		for (int i = 0; i < set_size; i++) {
			if (gains[indices[i]] + gains[indices[set_size + i]] > 0) {
				uint32_t temp = indices[i];
				indices[i] = indices[set_size + i];
				indices[set_size + i] = temp;
			}
			else break;
		}

		// rearrange by newly calculated order
		std::vector<uint32_t> clone(end - start);
		for (int i = start; i < end; i++) clone[i - start] = order[i];
		for (int i = 0; i < indices.size(); i++) order[start + i] = clone[indices[i]];
	}

	void reorder(const uint32_t max_iters = 20, const bool verbose = false) {
		if (verbose) std::cout << "Initial log gap cost:\t" << log_gap_cost(0, order.size()) << std::endl;
		std::queue<std::pair<uint32_t, uint32_t>> queue;
		queue.push(std::make_pair(0, order.size()));
		
		while (!queue.empty()) {
			initial_partition(queue.front().first, queue.front().second);
			for (int i = 0; i < max_iters; i++) {
				iterated_swap(queue.front().first, queue.front().second);
			}

			uint32_t mid = (queue.front().first + queue.front().second) / 2;
			if (mid - queue.front().first > 1) queue.push(std::make_pair(queue.front().first, mid));
			if (queue.front().second - mid > 1) queue.push(std::make_pair(mid, queue.front().second));

			if (verbose && queue.front().second == order.size()) std::cout << "Gap cost after level size " << (queue.front().second - queue.front().first) << ":\t" << log_gap_cost(0, order.size()) << std::endl;
			queue.pop();
		}
	}

	void write_to_file(const char *filename) {
		std::ofstream writer(filename);
		if (!writer.is_open()) return;

		uint32_t temp = order.size();
		writer.write((char*)(&temp), sizeof(uint32_t));
		for (size_t i = 0; i < order.size(); i++) {
			writer.write((char*)(&order[i]), sizeof(uint32_t));
		}
	}

	std::string range_to_string(const uint32_t start, const uint32_t end) {
		std::string str = "";
		for (uint32_t i = start; i < end; i++) {
			str += std::to_string(order[i]) + " ";
		}
		return str;
	}

	double log_gap_cost(const uint32_t start, const uint32_t end) {
		double total_log_gap_cost = 0;
		// we're going to add the individual normalized gap costs of each vector, then average them at the end
		for (int i = 0; i < fwd_index.points.size(); i++) {
			// preempt cases with no or all nonzeros, because they'll cause calculation issues
			if (fwd_index.points[i].size() == 0 || fwd_index.points[i].size() == fwd_index.dims) {
				total_log_gap_cost += 1;
				continue;
			}

			// apply order to nonzeros and then sort by new indices
			std::vector<uint32_t> indices(fwd_index.points[i].size());
			std::iota(indices.begin(), indices.end(), 0);
			std::sort(indices.begin(), indices.end(), [this, i] (uint32_t a, uint32_t b) -> bool {
					return this->order[this->fwd_index.points[i][a].index] < order[this->fwd_index.points[i][b].index];
					});

			// calculate log gaps
			double log_gap_cost = 0;
			for (int j = 1; j < fwd_index.points[i].size(); j++) {
				log_gap_cost += log(order[fwd_index.points[i][indices[j]].index] - order[fwd_index.points[i][indices[j - 1]].index]);
			}
			
			// normalize by optimal cost
			if (fwd_index.points[i].size() > 1) log_gap_cost /= (fwd_index.points[i].size() - 1);
			total_log_gap_cost += log_gap_cost;
		}

		return total_log_gap_cost / (fwd_index.points.size());
	}

	unsigned int operator [] (const size_t i) {
		return order[i];
	}
};

std::string range_to_string(coord_order order, uint32_t start, uint32_t end) {
	return order.range_to_string(start, end);
}
double log_gap_cost(coord_order order, const uint32_t start, const uint32_t end) {
	return order.log_gap_cost(start, end);
}
