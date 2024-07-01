/*#include <iostream>
#include <cstdio>
#include <fstream>
#include <strings.h>
#include <string>
#include <vector>
#include <algorithm>
#include <cassert>

int main(int argc, char **argv) {
	if (argc < 4) {
		std::cout << "Usage: " << argv[0] << " [csr infile] [trec outfile] [sample size]" << std::endl;
		exit(0);
	}

	std::ifstream reader(argv[1]);
	if (!reader.is_open()) {
		printf("Could not open file\n");
		exit(0);
	}

	uint64_t num_vecs, num_dims, num_vals;
	reader.read((char*)(&num_vecs), sizeof(uint64_t));
	reader.read((char*)(&num_dims), sizeof(uint64_t));
	reader.read((char*)(&num_vals), sizeof(uint64_t));

	std::vector<unsigned int> sample(num_vecs, 0);
	for (int i = 0; i < num_vecs; i++) sample[i] = i;
	std::random_shuffle(sample.begin(), sample.end());

	unsigned int sample_size = std::stoi(argv[3]);
	if (sample_size > num_vecs) sample_size = num_vecs;

	std::ofstream writer(argv[2]);
	for (int i = 0; i < sample_size; i++) {
		reader.seekg((sample[i] + 3) * sizeof(uint64_t));
		uint64_t indptr_start, indptr_end;
		reader.read((char*)(&indptr_start), sizeof(uint64_t));
		reader.read((char*)(&indptr_end), sizeof(uint64_t));

		reader.seekg((num_vecs + 4) * sizeof(uint64_t) + indptr_start * sizeof(unsigned int));
		writer << "<top>\n" << "<num> Number: " << sample[i] << "\n" << "<desc>\n";
		unsigned int temp_index;
		for (; indptr_start < indptr_end; indptr_start++) {
			reader.read((char*)(&temp_index), sizeof(unsigned int));
			writer << temp_index << " ";
		}
		writer << "\n" << "</top>\n\n";
	}

	reader.close();
	writer.close();

	return 0;
}*/

#include <iostream>
#include <chrono>
#include <immintrin.h> // for AVX
#include <bitset>

// Function to measure AVX operations
void test_avx() {
    volatile __m256 a = _mm256_set1_ps(1.0f);
    volatile __m256 b = _mm256_set1_ps(2.0f);
    volatile __m256 c;

    for (int i = 0; i < 1000000; ++i) {
        c = _mm256_add_ps(a, b);
    }
}

// Function to measure popcount operations
void test_popcount() {
    volatile int count = 0;
    for (int i = 0; i < 1000000; ++i) {
        count += __builtin_popcount(i);
    }
}

int main() {
    // Measure time for AVX operations
    auto start_avx = std::chrono::high_resolution_clock::now();
    test_avx();
    auto end_avx = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_avx = end_avx - start_avx;
    std::cout << "Time taken for 1,000,000 AVX operations: " << duration_avx.count() << " seconds" << std::endl;

    // Measure time for popcount operations
    auto start_popcount = std::chrono::high_resolution_clock::now();
    test_popcount();
    auto end_popcount = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_popcount = end_popcount - start_popcount;
    std::cout << "Time taken for 1,000,000 popcount operations: " << duration_popcount.count() << " seconds" << std::endl;

    return 0;
}
