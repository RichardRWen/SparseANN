#ifndef _FORWARD_INDEX_H_
#define _FORWARD_INDEX_H_

#include <stdlib.h>
#include <cstdint>
#include <vector>
#include <unordered_map>
#include <cstring>
#include <fstream>
#include <algorithm>

#include <parlay/sequence.h>

template <typename val_type = float>
class forward_index {
public:
	uint32_t dims;
	parlay::sequence<parlay::sequence<std::pair<uint32_t, val_type>>> points;

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
};

#endif
