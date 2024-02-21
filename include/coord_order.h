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
#include "parlay_util.h"

/* TODO: Things I've noticed
	- reduce doesn't accept more complicated values, either as accumulator or as input
	- reduce omits the start of the slice
*/

struct coord_order {
	std::vector<uint32_t> order; // what coordinate is in each position
	std::vector<uint32_t> order_map; // what position each coordinate is in
	inverted_index<float, uint32_t> inv_index;
	forward_index<float> fwd_index;

	coord_order(const uint32_t dims) : order(dims, 0), order_map(dims, 0), inv_index(dims), fwd_index(dims) {
		std::iota(order.begin(), order.end(), (uint32_t)0);
		std::iota(order_map.begin(), order_map.end(), (uint32_t)0);
	}
	coord_order(const char *filename, const char *filetype, const size_t _num_to_read = -1ULL) : inv_index(filename, filetype, _num_to_read), fwd_index(filename, filetype, _num_to_read) {
		if (strcmp(filetype, "csr") == 0) {
			order.resize(inv_index.num_lists);
			std::iota(order.begin(), order.end(), (uint32_t)0);
			order_map.resize(inv_index.num_lists);
			std::iota(order_map.begin(), order_map.end(), (uint32_t)0);
			assert(inv_index.posting_lists.size() == order.size() && fwd_index.dims == order.size());
		}
		else if (strcmp(filetype, "ord") == 0) {
			std::ifstream reader(filename);
			if (!reader.is_open()) {
				std::cout << "Unable to open " << filename << std::endl;
				return;
			}

			uint32_t num_dims;
			reader.read((char*)(&num_dims), sizeof(uint32_t));
			order.resize(num_dims);
			order_map.resize(num_dims);

			reader.read((char*)(&order[0]), num_dims * sizeof(uint32_t));
			for (uint32_t i = 0; i < num_dims; i++) {
				order_map[order[i]] = i;
			}
			
			reader.close();
		}
	}

	void shingle_order_seq(const uint32_t start, const uint32_t end) {
		hasher_murmur64a hasher;
		std::vector<uint32_t> indices(end - start);
		std::iota(indices.begin(), indices.end(), 0);

		// Find the shingle of each dimension ie. the lowest hash among vectors nonzero in that dimension
		std::vector<uint32_t> shingles(end - start, -1);
		for (int i = 0; i < shingles.size(); i++) {
			for (uint32_t j = 0; j < inv_index.posting_lists[order[start + i]].size(); j++) {
				uint32_t hash = (uint32_t)hasher(inv_index.posting_lists[order[start + i]][j].first);
				if (hash < shingles[i]) shingles[i] = hash;
			}
		}

		// Sort dimensions by their shingles
		std::sort(indices.begin(), indices.end(), [shingles] (uint32_t a, uint32_t b) -> bool {
				return shingles[a] < shingles[b];
			});
		// Use the shingles list as a temp storage for the coordinate order
		for (int i = 0; i < shingles.size(); i++) {
			shingles[i] = order[start + i];
		}
		// Reorder dimensions by the sorted shingle ordering
		for (int i = 0; i < shingles.size(); i++) {
			order[start + i] = shingles[indices[i]];
			order_map[order[start + i]] = start + i;
		}
	}

	void shingle_order(const uint32_t start, const uint32_t end) {
		hasher_murmur64a hasher;

		auto shingles = parlay::sequence<uint32_t>::from_function(end - start,
			[&] (size_t i) -> uint32_t {
				uint32_t shingle = -1;
				parlay::parallel_for(0, inv_index.posting_lists[order[start + i]].size(),
					[&] (size_t j) {
						uint32_t hash = (uint32_t)hasher(inv_index.posting_lists[order[start + i]][j].first);
						if (hash < shingle) shingle = hash;
					}
				);
				return shingle;
			}
		);

		auto indices = parlay::sort(
			parlay::tabulate(end - start, [] (size_t i) {return i;}), 
			[&shingles] (uint32_t a, uint32_t b) -> bool {
				return shingles[a] < shingles[b];
			});
		parlay::copy(parlay::make_slice(&order[start], &order[start + shingles.size()]), parlay::make_slice(shingles));
		parlay::parallel_for(0, shingles.size(), [&] (size_t i) {
			order[start + i] = shingles[indices[i]];
			order_map[order[start + i]] = start + i;
		}, 100);
	}

	double partial_move_gain_seq(const uint32_t vector_id, std::unordered_set<uint32_t> &set1, std::unordered_set<uint32_t> &set2) {
		uint32_t deg1 = 0, deg2 = 0;
		for (auto coord : fwd_index.points[vector_id]) {
			if (set1.find(coord.first) != set1.end()) deg1++;
			else if (set2.find(coord.first) != set2.end()) deg2++;
		}

		double partial_gain = 0; // Might want to ask about whether we should account for the set sizes changing
		partial_gain -= deg1       * log((double)(set1.size()) / (deg1 + 1));
		partial_gain -= deg2       * log((double)(set2.size()) / (deg2 + 1));
		partial_gain += (deg1 - 1) * log((double)(set1.size()) / (deg1));
		partial_gain += (deg2 + 1) * log((double)(set2.size()) / (deg2 + 2));
		return partial_gain;
	}
	double move_gain_seq(const uint32_t coord_id, std::unordered_set<uint32_t> &set1, std::unordered_set<uint32_t> &set2) {
		double move_gain = 0;
		for (auto vector : inv_index.posting_lists[coord_id]) {
			move_gain += partial_move_gain_seq(vector.first, set1, set2);
		}
		return move_gain;
	}

	bool iterated_swap_seq(const uint32_t start, const uint32_t end) {
		if (end - start <= 2) return false;
		uint32_t set_size = (end - start) / 2;
		uint32_t mid = start + set_size;

		// split coords in half to form two sets
		std::unordered_set<uint32_t> set1;
		std::unordered_set<uint32_t> set2;
		for (int i = start; i < mid; i++) set1.insert(order[i]);
		for (int i = mid;   i < end; i++) set2.insert(order[i]);

		// calculate move gains for every coord
		std::vector<double> gains(end - start);
		for (int i = start; i < end; i++) {
			if (i < mid) gains[i - start] = move_gain_seq(order[i], set1, set2);
			else gains[i - start] = move_gain_seq(order[i], set2, set1);
		}

		// sort the two sets according to move gain
		std::vector<uint32_t> indices(end - start);
		std::iota(indices.begin(), indices.end(), (uint32_t)0);
		std::sort(indices.begin(), indices.begin() + set_size, [&gains] (uint32_t a, uint32_t b) -> bool {
				return gains[a] < gains[b];
				});
		std::sort(indices.begin() + set_size, indices.end(),   [&gains] (uint32_t a, uint32_t b) -> bool {
				return gains[a] < gains[b];
				});

		// perform swaps between the sets so long as they are profitable
		for (int i = 0; i < set_size; i++) {
			if (gains[indices[i]] + gains[indices[set_size + i]] < 0) {
				uint32_t temp = indices[i];
				indices[i] = indices[set_size + i];
				indices[set_size + i] = temp;
			}
			else if (i == 0) return false;
			else break;
		}

		// rearrange by newly calculated order
		// NOTE: This rearrangement schema causes unwapped coordinates to shift to the back of their half. May want to consider if this is better than directly swapping coordinates where they are. The current schema would cause the actual log gap cost to shift around a bit, but it may make subsequent sorts cheaper because it maintains some semblence of sorted order.
		std::vector<uint32_t> clone(end - start);
		for (int i = start; i < end; i++) clone[i - start] = order[i];
		for (int i = 0; i < indices.size(); i++) {
			order[start + i] = clone[indices[i]];
			order_map[order[start + i]] = start + i;
		}
		return true;
	}

	std::pair<uint32_t, uint32_t> _move_gain_set_count(parlay::sequence<std::pair<uint32_t, float>> &list, size_t start, size_t end, std::unordered_set<uint32_t> &set1, std::unordered_set<uint32_t> &set2) {
		if (end <= start) return std::make_pair<uint32_t, uint32_t>(0, 0);
		if (end - start == 1) {
			if (set1.find(list[start].first) != set1.end()) return std::make_pair<uint32_t, uint32_t>(1, 0); // TODO: CHECK IF THE INDICES IN LIST ARE IN REORDERED FORM OR NOT
			else if (set2.find(list[start].first) != set2.end()) return std::make_pair<uint32_t, uint32_t>(0, 1);
			else return std::make_pair<uint32_t, uint32_t>(0, 0);
		}
		
		std::pair<uint32_t, uint32_t> gain1, gain2;
		uint32_t mid = (start + end) / 2;
		parlay::par_do(
		[&]() {
			gain1 = _move_gain_set_count(list, start, mid, set1, set2);
		},
		[&]() {
			gain2 = _move_gain_set_count(list, mid, end, set1, set2);
		});

		return std::make_pair<uint32_t, uint32_t>(gain1.first + gain2.first, gain1.second + gain2.second);
	}
	double _move_gain(parlay::sequence<std::pair<uint32_t, float>> &p_list, uint32_t start, uint32_t end, std::unordered_set<uint32_t> &set1, std::unordered_set<uint32_t> &set2) {
		if (end <= start) return 0;
		if (end - start == 1) {
			std::pair<uint32_t, uint32_t> deg = _move_gain_set_count(fwd_index.points[p_list[start].first], 0, fwd_index.points[p_list[start].first].size(), set1, set2);

			double partial_gain = 0; // NOTE: I think these calcs may be wrong - check the seq version
			partial_gain -= log((double)(set1.size()) / (deg.first + 1)) * (deg.first + 1);
			partial_gain -= log((double)(set2.size()) / (deg.second + 1)) * (deg.second + 1);
			partial_gain += log((double)(set1.size()) / (deg.first)) * (deg.first);
			partial_gain += log((double)(set2.size()) / (deg.second + 2)) * (deg.second + 2);
			return partial_gain;
		}

		double gain1, gain2;
		uint32_t mid = (start + end) / 2;
		parlay::par_do(
		[&]() {
			gain1 = _move_gain(p_list, start, mid, set1, set2);
		},
		[&]() {
			gain2 = _move_gain(p_list, mid, end, set1, set2);
		});

		return gain1 + gain2;
	}

	/* Helper function for _move_gain()
	 * Given a vector and two sets of dimensions, counts how many
	 * of the nonzero dimensions of the vector are in each set.
	 */
	std::pair<uint32_t, uint32_t> _get_degrees(parlay::sequence<std::pair<uint32_t, float>> &vec, std::unordered_set<uint32_t> &set1, std::unordered_set<uint32_t> &set2) {
		auto partial_degrees = parlay::delayed_tabulate(vec.size(),
			[&vec, &set1, &set2] (size_t i) -> std::pair<uint32_t, uint32_t> {
				if (set1.find(vec[i].first) != set1.end()) return std::make_pair<uint32_t, uint32_t>(1, 0);
				else if (set2.find(vec[i].first) != set2.end()) return std::make_pair<uint32_t, uint32_t>(0, 1);
				else return std::make_pair<uint32_t, uint32_t>(0, 0);
			});
		std::pair<uint32_t, uint32_t> degrees = parlay::reduce(parlay::make_slice(partial_degrees.begin(), partial_degrees.end()),
			parlay::binary_op([] (std::pair<uint32_t, uint32_t> a, std::pair<uint32_t, uint32_t> b) -> std::pair<uint32_t, uint32_t> {
				return std::make_pair<uint32_t, uint32_t>(a.first + b.first, a.second + b.second);
			}, std::make_pair<uint32_t, uint32_t>(0, 0)));
		return degrees;
	}
	/* Helper function for iterated_swap()
	 * Calculates the gain from moving a particular dimension from
	 * one set to the other.
	 */
	double _move_gain(const uint32_t index, std::unordered_set<uint32_t> &set1, std::unordered_set<uint32_t> &set2) {
		auto partial_move_gains = parlay::delayed_tabulate(inv_index.posting_lists[index].size(),
			[this, index, &set1, &set2] (size_t i) -> double {
				std::pair<uint32_t, uint32_t> deg = _get_degrees(fwd_index.points[inv_index.posting_lists[index][i].first], set1, set2);

				double partial_gain = 0;
				partial_gain -= log((double)(set1.size()) / (deg.first + 1)) * (deg.first + 1);
				partial_gain -= log((double)(set2.size()) / (deg.second + 1)) * (deg.second + 1);
				partial_gain += log((double)(set1.size()) / (deg.first)) * (deg.first);
				partial_gain += log((double)(set2.size()) / (deg.second + 2)) * (deg.second + 2);
				return partial_gain;
			});
		double move_gain = parlay::reduce(parlay::make_slice(partial_move_gains.begin(), partial_move_gains.end()),
			parlay::binary_op([] (double a, double b) -> double {
				return a + b;
			}, (double)0));
		return move_gain;
	}
	/* Wrapper function for _move_gain()
	 */
	double move_gain(const uint32_t index, std::unordered_set<uint32_t> &set1, std::unordered_set<uint32_t> &set2) {
		return _move_gain(inv_index.posting_lists[index], 0, inv_index.posting_lists[index].size(), set1, set2);
	}
	/* Given a range of dimensions in the coordinate order,
	 * partition them into two sets, then greedily swap dimensions
	 * across the two partitions in order to maximize correlation
	 * between the dimensions in each partition.
	 */
	bool iterated_swap(const uint32_t start, const uint32_t end) {
		if (end - start < 2) return false;
		uint32_t set_size = (end - start) / 2;
		uint32_t mid = (end + start) / 2;

		// TODO: profile if parlay hashtable improves performance
		// split dims in half to form two sets
		std::unordered_set<uint32_t> set1;
		std::unordered_set<uint32_t> set2;
		for (int i = start; i < mid; i++) set1.insert(order[i]);
		for (int i = mid;   i < end; i++) set2.insert(order[i]);

		// calculate move gains for every dim
		auto move_gains = parlay::sequence<double>::from_function(end - start,
			[&] (size_t i) -> double {
				if (i < set_size) return move_gain(order[start + i], set1, set2);
				else return move_gain(order[start + i], set2, set1);
			}
		);

		// sort the dims according to move gain
		auto indices = parlay::sequence<uint32_t>::from_function(end - start, 
			[] (size_t i) -> uint32_t {
				return (uint32_t)i;
			}
		);
		parlay::par_do(
			[&indices, &move_gains, set_size] () {
				parlay::stable_sort_inplace(parlay::make_slice(indices.begin(), indices.begin() + set_size),
					[&move_gains] (uint32_t a, uint32_t b) -> bool {
						return move_gains[a] < move_gains[b];
					}
				);
			},
			[&indices, &move_gains, set_size] () {
				parlay::stable_sort_inplace(parlay::make_slice(indices.begin() + set_size, indices.end()),
					[&move_gains] (uint32_t a, uint32_t b) -> bool {
						return move_gains[a] < move_gains[b];
					}
				);
			}
		);
		
		// swap dims between the sets as long as it's profitable
		for (int i = 0; i < set_size; i++) {
			if (move_gains[indices[i]] + move_gains[indices[set_size + i]] < 0) {
				uint32_t temp = indices[i];
				indices[i] = indices[set_size + i];
				indices[set_size + i] = temp;
			}
			else if (i == 0) return false;
			else break;
		}

		// rearrange by newly calculated order
		// TODO: see if order_map can be set first to remove need for clone
		auto clone = parlay::sequence<uint32_t>::from_function(end - start, [&] (size_t i) {return order[start + i];});
		for (int i = 0; i < indices.size(); i++) {
			order[start + i] = clone[indices[i]];
			order_map[order[start + i]] = start + i;
		}

		return true;
	}

	void reorder_seq(const uint32_t max_iters = 20, const bool verbose = false) {
		if (verbose) std::cout << "Initial log gap cost:\t" << log_gap_cost_seq() << std::endl;
		auto start = std::chrono::high_resolution_clock::now();

		std::queue<std::pair<uint32_t, uint32_t>> queue;
		queue.push(std::make_pair(0, order.size()));
		
		while (!queue.empty()) {
			shingle_order_seq(queue.front().first, queue.front().second);
			for (int i = 0; i < max_iters; i++) {
				if (!iterated_swap_seq(queue.front().first, queue.front().second)) break;
			}

			uint32_t mid = (queue.front().first + queue.front().second) / 2;
			if (mid - queue.front().first > 1) queue.push(std::make_pair(queue.front().first, mid));
			if (queue.front().second - mid > 1) queue.push(std::make_pair(mid, queue.front().second));

			if (verbose && queue.front().second == order.size()) std::cout << "Gap cost after level size " << (queue.front().second - queue.front().first) << ":\t" << log_gap_cost_seq() << std::endl;
			queue.pop();
		}

		if (verbose) {
			auto stop = std::chrono::high_resolution_clock::now();
			auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
			std::cout << "Final log gap cost:\t" << log_gap_cost_seq() << std::endl;
			std::cout << "Execution time: " << duration.count() / 1000.0 << " sec" << std::endl;
		}
	}

	void _reorder(const uint32_t start, const uint32_t end, const uint32_t max_iters) {
		if (end - start <= 2) return;

		shingle_order(start, end);
		for (int i = 0; i < max_iters; i++) {
			if (!iterated_swap(start, end)) break;
		}

		uint32_t mid = (start + end) / 2;
		parlay::par_do(
			[&] () {
				_reorder(start, mid, max_iters);
			},
			[&] () {
				_reorder(mid, end, max_iters);
			}
		);
	}
	// TODO: make sure iter_swaps terminates early if no swaps done
	void reorder(const uint32_t max_iters = 20, const bool verbose = false) {
		if (verbose) {
			std::cout << "Initial log gap cost:\t" << log_gap_cost() << std::endl;
			auto start = std::chrono::high_resolution_clock::now();

			_reorder(0, order.size(), max_iters);
		
			auto stop = std::chrono::high_resolution_clock::now();
			auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
			std::cout << "Final log gap cost:\t" << log_gap_cost() << std::endl;
			std::cout << "Execution time: " << duration.count() / 1000.0 << " sec" << std::endl;
		}
		else _reorder(0, order.size(), max_iters);
	}

	void write_to_file(const char *filename) {
		std::ofstream writer(filename);
		if (!writer.is_open()) {
			std::cout << "Couldn't open file " << filename << " for writing" << std::endl;
			std::ofstream retry("failsafe_record.ord");
			if (retry.is_open()) {
				uint32_t temp = order.size();
				retry.write((char*)(&temp), sizeof(uint32_t));
				for (size_t i = 0; i < order.size(); i++) {
					retry.write((char*)(&order[i]), sizeof(uint32_t));
				}
				retry.close();
				return;
			}
			else return;
		}

		uint32_t temp = order.size();
		writer.write((char*)(&temp), sizeof(uint32_t));
		for (size_t i = 0; i < order.size(); i++) {
			writer.write((char*)(&order[i]), sizeof(uint32_t));
		}
		writer.close();
	}

	void apply_ordering(const char *infile, const char *outfile) {
		std::ifstream indptr_reader(infile);
		std::ifstream index_reader(infile);
		std::ifstream value_reader(infile);
		if (!indptr_reader.is_open() || !index_reader.is_open() || !value_reader.is_open()) {
			std::cout << "Could not open " << infile << std::endl;
			return;
		}
		
		std::ofstream writer(outfile);
		if (!writer.is_open()) {
			std::cout << "Could not open " << outfile << std::endl;
			return;
		}

		uint64_t num_vecs, num_dims, num_vals, indptr = 0, indptr_prev = indptr;
		indptr_reader.read((char*)(&num_vecs), sizeof(uint64_t));
		indptr_reader.read((char*)(&num_dims), sizeof(uint64_t));
		indptr_reader.read((char*)(&num_vals), sizeof(uint64_t));

		writer.write((char*)(&num_vecs), sizeof(uint64_t));
		writer.write((char*)(&num_dims), sizeof(uint64_t));
		writer.write((char*)(&num_vals), sizeof(uint64_t));

		for (uint64_t i = 0; i < num_vecs + 1; i++) {
			indptr_reader.read((char*)(&indptr), sizeof(uint64_t));
			writer.write((char*)(&indptr), sizeof(uint64_t));
		}

		indptr_reader.seekg(4 * sizeof(uint64_t));
		index_reader.seekg((num_vecs + 4) * sizeof(uint64_t));
		indptr = 0;
		std::vector<uint32_t> index_buffer(num_dims);
		for (uint64_t i = 0; i < num_vecs; i++) {
			indptr_prev = indptr;
			indptr_reader.read((char*)(&indptr), sizeof(uint64_t));
			index_reader.read((char*)(&index_buffer[0]), (indptr - indptr_prev) * sizeof(uint32_t));

			std::sort(index_buffer.begin(), index_buffer.begin() + (indptr - indptr_prev), [this] (uint32_t a, uint32_t b) -> bool {
					//std::cout << a << " " << b << std::endl;
					return order_map[a] < order_map[b];
				});

			for (uint64_t j = 0; j < indptr - indptr_prev; j++) {
				index_buffer[j] = order_map[index_buffer[j]];
			}
			writer.write((char*)(&index_buffer[0]), (indptr - indptr_prev) * sizeof(uint32_t));
		}

		indptr_reader.seekg(4 * sizeof(uint64_t));
		index_reader.seekg((num_vecs + 4) * sizeof(uint64_t));
		value_reader.seekg((num_vecs + 4) * sizeof(uint64_t) + num_vals * sizeof(uint32_t));
		indptr = 0;
		std::vector<float> value_buffer(num_dims);
		std::vector<uint32_t> sort_buffer(num_dims);
		for (uint64_t i = 0; i < num_vecs; i++) {
			indptr_prev = indptr;
			indptr_reader.read((char*)(&indptr), sizeof(uint64_t));
			index_reader.read((char*)(&index_buffer[0]), (indptr - indptr_prev) * sizeof(uint32_t));
			value_reader.read((char*)(&value_buffer[0]), (indptr - indptr_prev) * sizeof(float));
			std::iota(sort_buffer.begin(), sort_buffer.begin() + (indptr - indptr_prev), (uint32_t)0);

			//std::cout << "sorting" << std::endl;
			std::sort(sort_buffer.begin(), sort_buffer.begin() + (indptr - indptr_prev), [this, &index_buffer] (uint32_t a, uint32_t b) -> bool {
					//std::cout << a << " " << b << " " << index_buffer[a] << " " << index_buffer[b] << std::endl;
					return order_map[index_buffer[a]] < order_map[index_buffer[b]];
				});

			for (uint64_t j = 0; j < indptr - indptr_prev; j++) {
				index_buffer[j] = value_buffer[sort_buffer[j]];
			}
			writer.write((char*)(&index_buffer[0]), (indptr - indptr_prev) * sizeof(uint32_t));
		}

		indptr_reader.close();
		index_reader.close();
		value_reader.close();
		writer.close();
	}

	std::string range_to_string(const uint32_t start, const uint32_t end) {
		std::string str = "";
		for (uint32_t i = start; i < end; i++) {
			str += std::to_string(order[i]) + " ";
		}
		return str;
	}

	double log_gap_cost_seq() {
		double total_log_gap_cost = 0;
		for (auto coords : fwd_index.points) {
			//std::cout << "fwd_index.points[2][1].value = " << fwd_index.points[2][1].value << std::endl;
			if (coords.size() <= 1) continue;

			std::vector<uint32_t> indices(coords.size());
			std::iota(indices.begin(), indices.end(), (uint32_t)0);

			/*std::cout << "available:";
			for (int i = 0; i < coords.size(); i++) {
				std::cout << " " << coords[i].index;
			}
			std::cout << std::endl;*/
			std::sort(indices.begin(), indices.end(), [this, &coords] (const uint32_t a, const uint32_t b) -> bool {
					//std::cout << a << " " << coords[a].index << " " << b << " " << coords[b].index << std::endl;
					return order_map[coords[a].first] < order_map[coords[b].first];
				});

			double partial_log_gap_cost = 0;
			for (int i = 1; i < coords.size(); i++) {
				partial_log_gap_cost += log(order_map[coords[indices[i]].first] - order_map[coords[indices[i - 1]].first]);
			}
			total_log_gap_cost += partial_log_gap_cost / (coords.size() - 1);
		}

		return total_log_gap_cost / fwd_index.points.size();
	}

	double _log_gap_cost(const parlay::sequence<std::pair<uint32_t, float>> &coords) {
		if (coords.size() <= 1) {
			return 0;
		}

		auto indices = parlay::sequence<uint32_t>::from_function(coords.size(),
			[] (size_t i) {
				return i;
			}
		);

		parlay::sort_inplace(parlay::make_slice(indices.begin(), indices.end()),
			[this, &coords] (const uint32_t a, const uint32_t b) -> bool {
				return order_map[coords[a].first] < order_map[coords[b].first];
			});

		double log_gap_cost = parlay::reduce(parlay::iota<uint32_t>(coords.size() - 1),
			parlay::binary_op([this, &coords, &indices] (double acc, uint32_t x) -> double {
				assert(order_map[coords[indices[x]].first] < order_map[coords[indices[x + 1]].first]);
				return acc + log(order_map[coords[indices[x + 1]].first] - order_map[coords[indices[x]].first]);
			}, (double)0));

		return log_gap_cost / (coords.size() - 1);
	}
	double log_gap_cost() {
		auto partial_log_gap_costs = parlay::delayed_tabulate(fwd_index.points.size(), [this](size_t i) {
				return _log_gap_cost(fwd_index.points[i]);
			});
		double total_log_gap_cost = parlay::reduce(parlay::make_slice(partial_log_gap_costs.begin(), partial_log_gap_costs.end()));
		return total_log_gap_cost / fwd_index.points.size();
	}
	/*double _log_gap_cost(const uint32_t start, const uint32_t end) {
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
	}*/

	unsigned int operator [] (const size_t i) {
		return order[i];
	}
};

std::string range_to_string(coord_order order, uint32_t start, uint32_t end) {
	return order.range_to_string(start, end);
}
double log_gap_cost(coord_order order) {
	return order.log_gap_cost();
	//return order.log_gap_cost(start, end) + order._log_gap_cost(start, end);
}
