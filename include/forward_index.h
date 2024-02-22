#ifndef _FORWARD_INDEX_H_
#define _FORWARD_INDEX_H_

#include <stdlib.h>
#include <cstdint>
#include <vector>
#include <unordered_map>
#include <cstring>
#include <fstream>
#include <algorithm>
#include <cassert>

#include <parlay/sequence.h>
#include <parlay/primitives.h>
#include <openssl/rand.h>

#include "hashutil.h"

template <typename val_type = float>
class forward_index {
public:
	using coord_t = std::pair<uint32_t, val_type>;
	using point_t = parlay::sequence<coord_t>;
	uint32_t dims;
	parlay::sequence<point_t> points;

	forward_index(uint64_t _dims) : dims(_dims) {}
	forward_index(const char *filename, const char *filetype, const size_t _num_to_read = -1ULL) {
		if (strcmp(filetype, "csr") == 0) {
			std::ifstream indptr_reader(filename);
			if (!indptr_reader.is_open()) {
				return;
			}
			std::ifstream index_reader(filename);
			std::ifstream value_reader(filename);

			uint64_t num_vecs, num_dims, num_vals;
			indptr_reader.read((char*)(&num_vecs), sizeof(uint64_t));
			indptr_reader.read((char*)(&num_dims), sizeof(uint64_t));
			indptr_reader.read((char*)(&num_vals), sizeof(uint64_t));
			index_reader.seekg((num_vecs + 4) * sizeof(uint64_t));
			value_reader.seekg((num_vecs + 4) * sizeof(uint64_t) + num_vals * sizeof(uint32_t));

			dims = num_dims;
			size_t num_to_read = (num_vecs < _num_to_read ? num_vecs : _num_to_read);

			uint64_t indptr_start, indptr_end;
			uint32_t index;
			val_type value;
			indptr_reader.read((char*)(&indptr_end), sizeof(uint64_t));
			for (size_t i_vecs = 0; i_vecs < num_to_read; i_vecs++) {
				parlay::sequence<std::pair<uint32_t, val_type>> point;
				indptr_start = indptr_end;
				indptr_reader.read((char*)(&indptr_end), sizeof(uint64_t));
				for (; indptr_start < indptr_end; indptr_start++) {
					index_reader.read((char*)(&index), sizeof(uint32_t));
					value_reader.read((char*)(&value), sizeof(val_type));
					point.push_back(std::make_pair(index, value));
				}
				points.push_back(point);
			}

			indptr_reader.close();
			index_reader.close();
			value_reader.close();
		}
		else {
		}
	}
	
	static forward_index<val_type> copy(forward_index<val_type>& fwd_index) {
		forward_index<val_type> clone(fwd_index.dims);
		clone.points = parlay::sequence<point_t>::from_function(fwd_index.dims,
			[&] (size_t i) -> point_t {
				auto cloned_vector = point_t::uninitialized(fwd_index.dims);
				parlay::copy(fwd_index.points[i], cloned_vector);
				return cloned_vector;
			}
		);
		return clone;
	}
	static forward_index<val_type> group_and_max(forward_index<val_type>& fwd_index, size_t comp_dims) {
		forward_index<val_type> grouped(comp_dims);
		grouped.points = parlay::sequence<point_t>::from_function(comp_dims,
			[&] (size_t i) -> point_t {
				parlay::sequence<std::atomic<val_type>> maxes(fwd_index.dims, 0);
				size_t start = fwd_index.size() * i / comp_dims;
				size_t end = fwd_index.size() * (i + 1) / comp_dims;
				parlay::parallel_for(start, end,
					[&] (size_t j) {
						for (auto pair : fwd_index[j]) {
							val_type exp_max;
							do {
								exp_max = maxes[pair.first];
								if (exp_max >= pair.second) break;
							} while (maxes[pair.first].compare_exchange_weak(exp_max, pair.second));
						}
					}
				);

				point_t max_of_group;
				for (int j = 0; j < maxes.size(); j++) {
					if (maxes[j] == 0) continue;
					max_of_group.push_back(std::make_pair(j, maxes[j]));
				}
				return max_of_group;
			}
		);
		return grouped;
	}

	template <typename T>
	void reorder_dims(parlay::sequence<T>& map) {
		assert(map.size() >= dims);
		parlay::parallel_for(0, points.size(),
			[&] (size_t i) {
				parlay::sort_inplace(points[i],
					[&] (coord_t a, coord_t b) -> bool {
						return map[a.first] < map[b.first];
					}
				);
			}
		);
	}
	void shuffle_dims() {
		auto weights = parlay::sequence<uint32_t>::uninitialized(dims);
		RAND_bytes((unsigned char*)(&weights[0]), dims * sizeof(weights[0]));
		auto perm = parlay::sequence<uint32_t>::from_function(dims,
			[&] (uint32_t i) { return i; }
		);
		parlay::sort(perm,
			[&] (uint32_t a, uint32_t b) -> bool {
				return weights[a] < weights[b];
			}
		);
		reorder_dims(perm);
	}

	static val_type dist(parlay::sequence<std::pair<uint32_t, val_type>>& p1, parlay::sequence<std::pair<uint32_t, val_type>>& p2) {
		val_type total = (val_type)0;
		for (int i = 0, j = 0; i < p1.size() && j < p2.size(); ) {
			if (p1[i].first < p2[j].first) i++;
			else if (p1[i].first > p2[j].first) j++;
			else {
				total += p1[i].second * p2[j].second;
				i++;
				j++;
			}
		}
		return total;
	}

	bool write_to_file(const char *filename, const char *filetype) {
		std::ofstream writer(filename);
		if (!writer.is_open()) return false;

		uint64_t num_vecs = points.size();
		uint64_t num_dims = dims;
		uint64_t num_vals = parlay::reduce(
			parlay::delayed_tabulate(points.size(),
				[&] (size_t i) -> size_t { return points[i].size(); }
			)
		);

		writer.write((char*)(&num_vecs), sizeof(uint64_t));
		writer.write((char*)(&num_dims), sizeof(uint64_t));
		writer.write((char*)(&num_vals), sizeof(uint64_t));

		
	}
};

#endif
