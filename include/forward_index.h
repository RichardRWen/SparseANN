#ifndef _FORWARD_INDEX_H_
#define _FORWARD_INDEX_H_

#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <cstring>
#include <fstream>
#include <algorithm>
#include <cassert>

#include <parlay/sequence.h>
#include <parlay/delayed_sequence.h>
#include <parlay/primitives.h>
#include <openssl/rand.h>

#include "hashutil.h"

/*#ifndef _RANGE_CONCEPT_
#define _RANGE_CONCEPT_
template <typename T>
concept Range = requires(T a) {
	{ a.begin() } -> std::input_or_output_iterator;
	{ a.end() } -> std::input_or_output_iterator;
}
#endif*/

template <typename val_type = float>
class forward_index {
public:
	using coord_t = std::pair<uint32_t, val_type>;
	using point_t = parlay::sequence<coord_t>;
	uint32_t dims;
	parlay::sequence<point_t> points;

	forward_index() {}
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
        else if (strcmp(filetype, "vec") == 0) {
            std::ifstream reader(filename);
            if (!reader.is_open()) {
                return;
            }
            
            uint32_t num_vecs, num_dims;
            reader.read((char*)(&num_vecs), sizeof(uint32_t));
            reader.read((char*)(&num_dims), sizeof(uint32_t));
            dims = num_dims;
            
            for (uint32_t i = 0; i < num_vecs; i++) {
                parlay::sequence<std::pair<uint32_t, val_type>> point;
                for (uint32_t j = 0; j < num_dims; j++) {
                    val_type val;
                    reader.read((char*)(&val), sizeof(val_type));
                    if (val != 0) {
                        point.push_back(std::make_pair(j, val));
                    }
                }
                points.push_back(point);
            }

            reader.close();
        }
		else {
            std::cout << "File type not supported: " << filetype << std::endl;
		}
	}
	
	static forward_index<val_type> copy(forward_index<val_type>& fwd_index) {
		forward_index<val_type> clone(fwd_index.dims);
		clone.points = parlay::sequence<point_t>::from_function(fwd_index.points.size(),
			[&] (size_t i) -> point_t {
				return point_t::from_function(fwd_index.points[i].size(),
					[&] (size_t j) { return std::make_pair(fwd_index.points[i][j].first, fwd_index.points[i][j].second); });
			}
		);

		return clone;
	}
	static forward_index<val_type> head(forward_index<val_type>& fwd_index, size_t head_size) {
		forward_index<val_type> head_index(fwd_index.dims);
		head_index.points = parlay::sequence<point_t>::from_function(head_size,
			[&] (uint32_t i) -> point_t {
				return fwd_index.points[i];
			}
		);
		return head_index;
	}
	static forward_index<val_type> sample(forward_index<val_type>& fwd_index, size_t sample_size) {
		if (sample_size >= fwd_index.points.size()) return forward_index<val_type>::copy(fwd_index);
		
		auto iota = parlay::delayed_tabulate(fwd_index.points.size(), [] (uint32_t i) { return i; });
		auto priority = parlay::sequence<uint32_t>::uninitialized(fwd_index.points.size());
		RAND_bytes((unsigned char*)(&priority[0]), priority.size() * sizeof(uint32_t));
		auto perm = parlay::integer_sort(iota, [&] (uint32_t i) { return priority[i]; });

		forward_index<val_type> sample_index(fwd_index.dims);
		sample_index.points = parlay::sequence<point_t>::from_function(sample_size,
			[&] (uint32_t i) -> point_t {
				return fwd_index.points[perm[i]];
			}
		);

		return sample_index;
	}
	static forward_index<val_type> group_and_max(forward_index<val_type>& fwd_index, size_t comp_dims) {
		forward_index<val_type> grouped(comp_dims);
		grouped.points = parlay::sequence<point_t>::from_function(fwd_index.points.size(),
			[&] (size_t i) -> point_t {
				point_t comp_vector;
				int curr_comp_dim = -1, prev_comp_dim;
				for (const auto& coord : fwd_index.points[i]) {
					prev_comp_dim = curr_comp_dim;
					curr_comp_dim = coord.first * comp_dims / fwd_index.dims;
					if (curr_comp_dim == prev_comp_dim) {
						if (coord.second > comp_vector[comp_vector.size() - 1].second) {
							comp_vector[comp_vector.size() - 1].second = coord.second;
						}
					}
					else {
						comp_vector.emplace_back(curr_comp_dim, coord.second);
					}
				}
				return comp_vector;
			}
		);
		return grouped;
	}
	static forward_index<val_type> group_and_sum(forward_index<val_type>& fwd_index, size_t comp_dims) {
		forward_index<val_type> grouped(comp_dims);
		grouped.points = parlay::sequence<point_t>(fwd_index.points.size());

		parlay::parallel_for(0, fwd_index.points.size(), [&] (size_t i) {
			int range_start = 0, range_end = 0;
			while (range_start < fwd_index.points[i].size()) {
				uint32_t range_comp_dim = fwd_index.points[i][range_start].first * comp_dims / fwd_index.dims;
				val_type range_sum = 0;
				do { range_sum += fwd_index.points[i][range_end].second; range_end++; } while (fwd_index.points[i][range_end].first * comp_dims / fwd_index.dims == range_comp_dim);

				grouped.points[i].push_back(std::make_pair(range_comp_dim, range_sum));

				range_start = range_end;
			}
		});

		return grouped;
	}
	static forward_index<val_type> group(forward_index<val_type>& fwd_index, size_t comp_dims, val_type (*combine)(parlay::sequence<std::pair<uint32_t, val_type>>& coords, size_t start, size_t end)) {
		forward_index<val_type> grouped(comp_dims);
		grouped.points = parlay::sequence<point_t>(fwd_index.points.size());

		parlay::parallel_for(0, fwd_index.points.size(), [&] (size_t i) {
			int range_start = 0, range_end = 0;
			while (range_start < fwd_index.points[i].size()) {
				uint32_t range_comp_dim = fwd_index.points[i][range_start].first * comp_dims / fwd_index.dims;
				do { range_end++; } while (fwd_index.points[i][range_end].first * comp_dims / fwd_index.dims == range_comp_dim);

				grouped.points[i].push_back(std::make_pair(range_comp_dim, combine(fwd_index.points[i], range_start, range_end)));

				range_start = range_end;
			}
		});

		return grouped;
	}

    point_t& operator [] (size_t i) {
        return points[i];
    }

	template <typename T>
	void reorder_dims(parlay::sequence<T>& map) {
		assert(map.size() >= dims);
		parlay::parallel_for(0, points.size(),
			[&] (size_t i) {
				for (int j = 0; j < points[i].size(); j++) {
					points[i][j].first = map[points[i][j].first];
				}
				std::sort(points[i].begin(), points[i].end(),
					[&] (coord_t a, coord_t b) -> bool {
						return a.first < b.first;
					}
				);
			}
		);
	}
	void shuffle_dims() {
		auto weights = parlay::sequence<uint32_t>::uninitialized(dims);
		RAND_bytes((unsigned char*)(&weights[0]), dims * sizeof(weights[0]));
		auto iota = parlay::delayed_tabulate(weights.size(), [] (uint32_t i) { return i; });
		auto perm = parlay::integer_sort(iota,
			[&] (uint32_t a) -> uint32_t {
				return weights[a];
			}
		);
		reorder_dims(perm);
	}

	inline size_t size() {
		return points.size();
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
        if (strcmp(filetype, "csr") == 0) {
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

            uint64_t buffer = 0;
            writer.write((char*)(&buffer), sizeof(uint64_t));
            size_t bytes_written = 4 * sizeof(uint64_t);
            //std::cout << bytes_written << " bytes written" << std::endl;
            for (auto point : points) {
                buffer += point.size();
                writer.write((char*)(&buffer), sizeof(uint64_t));
            }
            //std::cout << bytes_written << " bytes written" << std::endl;
            for (auto point : points) {
                for (auto coord : point) {
                    writer.write((char*)(&coord.first), sizeof(uint32_t));
                }
            }
            //std::cout << bytes_written << " bytes written" << std::endl;
            for (auto point : points) {
                for (auto coord : point) {
                    writer.write((char*)(&coord.second), sizeof(val_type));
                }
            }
            //std::cout << bytes_written << " bytes written" << std::endl;

            writer.close();
            return true;
        }
        else if (strcmp(filetype, "vec") == 0) {
            std::ofstream writer(filename);
            if (!writer.is_open()) return false;

            uint32_t num_dims = dims, num_vecs = points.size();
            writer.write((char*)(&num_vecs), sizeof(uint32_t));
            writer.write((char*)(&num_dims), sizeof(uint32_t));
            
            for (uint32_t i = 0; i < num_vecs; i++) {
                parlay::sequence<val_type> point(dims, (val_type)0);
                parlay::parallel_for(0, points[i].size(), [&] (size_t j) {
                    point[points[i][j].first] = points[i][j].second;
                });
                writer.write((char*)(&point[0]), dims * sizeof(val_type));
            }

            writer.close();
            return true;
        }
	}
};

#endif
