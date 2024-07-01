#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <iostream>
#include <fstream>
#include <unordered_set>
#include <utility>
#include <cassert>

#include <immintrin.h>

#include <parlay/sequence.h>

#define BITMASK(X) ((1ull << X) - 1)
#define BYTE_ROUND_UP(BITS) ((BITS + 7) / 8)
#define DIV_ROUND_UP(X, Y) (((X - 1) / Y) + 1)

struct bitvector {
    size_t size;
    parlay::sequence<unsigned char> data;

    bitvector() : size(0) {}
    bitvector(size_t size) : size(size) {
        size_t num_bytes = BYTE_ROUND_UP(size);
        data = parlay::sequence<unsigned char>(num_bytes);
        std::memset(&data[0], (unsigned char)0, num_bytes);
    }

    std::string see_bits(size_t k) const {
        std::string s = "";
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                s += (data[k + i] & (1 << j) ? "1" : "0");
            }
        }
        return s;
    }

    inline bool get(size_t i) const {
        return data[i >> 3] & ((unsigned char)1 << (i & 0b111));
    }

    inline void set(const size_t i) {
        data[i >> 3] |= ((unsigned char)1 << (i & 0b111));
    }

    inline void unset(const size_t i) {
        data[i >> 3] &= ~((unsigned char)1 << (i & 0b111));
    }
};

inline uint16_t popcnt256(__m256i v) {
    return (uint16_t)(
        _mm_popcnt_u64(_mm256_extract_epi64(v, 0)) +
        _mm_popcnt_u64(_mm256_extract_epi64(v, 1)) +
        _mm_popcnt_u64(_mm256_extract_epi64(v, 2)) +
        _mm_popcnt_u64(_mm256_extract_epi64(v, 3)));
}

inline uint16_t dot256(__m256i a, __m256i b) {
    return popcnt256(_mm256_and_si256(a, b));
}

struct avx_bitvector {
    size_t size;
    parlay::sequence<unsigned char> data;

    avx_bitvector() : size(0) {}
    avx_bitvector(size_t _size) : size(_size) {
        size_t num_bytes = DIV_ROUND_UP(_size, 256) * 32;
        data = parlay::sequence<unsigned char>(num_bytes);
        std::memset(&data[0], (unsigned char)0, num_bytes);
    }
    
    template <typename id_type, typename val_type>
    static avx_bitvector from_sparse_coords(parlay::sequence<std::pair<id_type, val_type>>& vector, size_t dims) {
        avx_bitvector bv(dims);
        for (int i = 0; i < vector.size(); i++) {
            bv.set(vector[i].first);
        }
        return bv;
    }

    std::string see_bits(size_t k) const {
        std::string s = "";
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                s += (data[k + i] & (1 << j) ? "1" : "0");
            }
        }
        return s;
    }

    inline bool get(size_t i) const {
        return data[i >> 3] & ((unsigned char)1 << (i & 0b111));
    }

    inline void set(const size_t i) {
        data[i >> 3] |= ((unsigned char)1 << (i & 0b111));
    }

    inline void unset(const size_t i) {
        data[i >> 3] &= ~((unsigned char)1 << (i & 0b111));
    }

    uint16_t dot_product(avx_bitvector& other) {
        assert(size == other.size);
        size_t num_chunks = data.size() / 32;
        uint16_t count = 0;
        for (int i = 0; i < num_chunks; i++) {
            count += dot256(*((__m256i*)&data[i * 32]), *((__m256i*)&other.data[i * 32]));
        }
        return count;
    }

    static uint16_t dot_product(avx_bitvector &bv1, avx_bitvector &bv2) {
        assert(bv1.size == bv2.size);
        size_t num_chunks = bv1.data.size() / 32;
        uint16_t count = 0;
        for (int i = 0; i < num_chunks; i++) {
            //__m256i chunk1 = *((__m256i*)&bv1.data[i * 32]);
            //__m256i chunk2 = *((__m256i*)&bv1.data[i * 32]);
            __m256i chunk1 = _mm256_loadu_si256((__m256i*)&bv1.data[i * 32]);
            __m256i chunk2 = _mm256_loadu_si256((__m256i*)&bv2.data[i * 32]);
            count += dot256(chunk1, chunk2);
            //count += dot256(*((__m256i*)&bv1.data[i * 32]), *((__m256i*)&bv2.data[i * 32]));
        }
        return count;
    }
};

/*struct bloom_filter {
    bit_array bits;
    std::vector<uint32_t> hash_seeds;
    
    bloom_filter() {}
    bloom_filter(int n, float p) {
        assert(0 < p && p < 1);
        double ln2 = log(2);
        double m = (double)(-n) * log(p) / (ln2 * ln2);
        double k = m / n * ln2;

        bits = bit_array(ceil(m));
        hash_seeds = std::vector<uint32_t>(ceil(k));

        srand(time(NULL));
        for (uint32_t& seed : hash_seeds) {
            seed = rand();
        }
    }

    void insert(std::unordered_set<std::string>& keys) {
        for (std::string key : keys) {
            for (uint32_t seed : hash_seeds) {
                bits.set(MurmurHash64A(key.c_str(), key.size(), seed) % bits.size);
            }
        }
    }

    void query(std::vector<std::string>& keys) {
        for (std::string key : keys) {
            bool could_exist = true;
            for (uint32_t seed : hash_seeds) {
                if (!bits.get(MurmurHash64A(key.c_str(), key.size(), seed) % bits.size)) {
                    could_exist = false;
                    break;
                }
            }
            if (could_exist) std::cout << key << "\tPROB_YES" << std::endl;
            else std::cout << key << "\tNO" << std::endl;
        }
    }

    void save(char *filename) {
        std::ofstream writer(filename);
        if (!writer.is_open()) {
            std::cout << "Unable to open file " << filename << " for writing" << std::endl;
            exit(0);
        }

        int m = bits.size;
        int k = hash_seeds.size();
        writer.write((char*)&m, sizeof(int));
        writer.write((char*)&k, sizeof(int));
        writer.write((char*)bits.data, bits.num_bytes);
        writer.write((char*)&hash_seeds[0], hash_seeds.size() * sizeof(uint32_t));

        writer.close();
    }

    void load(char *filename) {
        std::ifstream reader(filename);
        if (!reader.is_open()) {
            std::cout << "Unable to open file " << filename << " for reading" << std::endl;
            exit(0);
        }

        int m, k;
        reader.read((char*)&m, sizeof(int));
        reader.read((char*)&k, sizeof(int));
        
        bits = bit_array(m);
        hash_seeds = std::vector<uint32_t>(k);
        reader.read((char*)bits.data, bits.num_bytes);
        reader.read((char*)&hash_seeds[0], hash_seeds.size() * sizeof(uint32_t));

        reader.close();
    }
};*/
