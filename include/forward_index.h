#include <stdlib.h>
#include <cstdint>
#include <vector>
#include <unordered_map>
#include <cstring>
#include <fstream>
#include <algorithm>

template <typename val_type = float, typename index_type = uint32_t>
class forward_index {
	public:
	struct coord {
		index_type index;
		val_type value;

		coord(const index_type _index, const val_type _value) : index(_index), value(_value) {}
	};

	uint64_t dims;
	std::vector<std::vector<coord>> points;

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
			float value;
			indptr_reader.read((char*)(&indptr_end), sizeof(uint64_t));
			for (size_t i_vecs = 0; i_vecs < num_to_read; i_vecs++) {
				std::vector<coord> point;
				indptr_start = indptr_end;
				indptr_reader.read((char*)(&indptr_end), sizeof(uint64_t));
				for (; indptr_start < indptr_end; indptr_start++) {
					index_reader.read((char*)(&index), sizeof(uint32_t));
					value_reader.read((char*)(&value), sizeof(float));
					coord new_coord(index, value);
					point.push_back(new_coord);
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
};
