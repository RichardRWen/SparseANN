#ifndef _SINNAMON_SKETCH_H_
#define _SINNAMON_SKETCH_H_

#include <cstdint>
#include <cstdlib>
#include <utility>

#include <parlay/sequence.h>
#include <openssl/rand.h>

#include "forward_index.h"

struct sinnamon_sketch {
    uint32_t dims;
    uint32_t quant_dims;

    std::vector<uint32_t> mapping;

    sinnamon_sketch(uint32_t _dims, uint32_t _quant_dims) : dims(_dims), quant_dims(_quant_dims) {
        mapping = std::vector<uint32_t>(dims);
        RAND_bytes((unsigned char*)&mapping[0], dims * sizeof(uint32_t));
        for (size_t i = 0; i < dims; i++) {
            mapping[i] %= quant_dims;
        }
    }

    template <typename T>
    parlay::sequence<T> transform_csr_to_qvec(parlay::sequence<std::pair<uint32_t, T>>& vec) {
        parlay::sequence<T> qvec(quant_dims, (T)0);
        for (auto& pair : vec) {
            if (pair.second > qvec[mapping[pair.first]]) qvec[mapping[pair.first]] = pair.second;
        }
        return qvec;
    }

    template <typename T>
    parlay::sequence<std::pair<uint32_t, T>> transform_qvec_to_qcsr(parlay::sequence<T>& vec) {
        parlay::sequence<std::pair<uint32_t, T>> qcsr;
        for (uint32_t i = 0; i < vec.size(); i++) {
            if (vec[i]) {
                qcsr.push_back(std::make_pair(i, vec[i]));
            }
        }
        return qcsr;
    }
};

#endif
