#pragma once

#include <cstdint>

#include <parlay/parallel.h>
#include <parlay/primitives.h>

#include "utils/sparse_mips_point.h"
#include "utils/sparse_point_range.h"

#include "forward_index.h"


template<typename T, class Point_>
SparsePointRange<T, Point_> fi_to_spr(forward_index<T> &fwd_index) {
    SparsePointRange<T, Point_> PR;

    PR.n = fwd_index.size();
    PR.dims = fwd_index.dims;

    auto vector_lens = parlay::delayed_tabulate(PR.n, [&] (size_t i) {
        return fwd_index[i].size();
    });
    size_t num_values = parlay::reduce(vector_lens);

    PR.indptrs = (uint64_t*) malloc((PR.n + 1) * sizeof(uint64_t));
    PR.indices = (unsigned int*) malloc(num_values * sizeof(unsigned int));
    PR.values = (T*) malloc(num_values * sizeof(T));
    parlay::parallel_for(0, PR.n, [&] (size_t i) {
        PR.indptrs[i] = i == 0 ? 0 : PR.indptrs[i - 1] + fwd_index[i - 1].size();
        for (size_t j = 0; j < fwd_index[i].size(); j++) {
            PR.indices[PR.indptrs[i] + j] = fwd_index[i][j].first;
            PR.values[PR.indptrs[i] + j] = fwd_index[i][j].second;
        }
    });

    return PR;
}