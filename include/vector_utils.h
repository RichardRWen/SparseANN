#ifndef _VECTOR_UTILS_H_
#define _VECTOR_UTILS_H_

#include <cstdint>
#include <cstdlib>
#include <utility>

#include <parlay/sequence.h>

template <typename T>
parlay::sequence<T> csr_to_vec(parlay::sequence<std::pair<uint32_t, T>>& csr, uint32_t dims) {
    parlay::sequence<T> vec(dims, (T)0);
    for (auto& pair : csr) {
        vec[pair.first] = pair.second;
    }
    return vec;
}

template <typename T>
parlay::sequence<std::pair<uint32_t, T>> vec_to_csr(parlay::sequence<T>& vec) {
    parlay::sequence<std::pair<uint32_t, T>> csr;
    for (uint32_t i = 0; i < vec.size(); i++) {
        if (vec[i]) {
            csr.push_back(std::make_pair(i, vec[i]));
        }
    }
    return csr;
}

#endif
