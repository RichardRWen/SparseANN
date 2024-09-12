#include <cstdint>
#include <cstring>
#include <iostream>
#include <algorithm>
#include <cassert>

#include <parlay/parallel.h>
#include <parlay/primitives.h>

#include "utils/types.h"
#include "utils/mips_point.h"
#include "utils/point_range.h"
#include "utils/sparse_mips_point.h"
#include "utils/sparse_point_range.h"
#include "utils/graph.h"
#include "vamana/neighbors.h"

#include "forward_index.h"
#include "point_range_util.h"
#include "vector_utils.h"

#include "sinnamon_sketch.h"
#include "count_min_sketch.h"
#include "jl_transform.h"

using index_type = uint32_t;
using value_type = float;
using point_type = SparseMipsPoint<value_type>;
using point_range_type = SparsePointRange<value_type, point_type>;

int main(int argc, char* argv[]) {
    std::string quant_type = ""; // default query type
    if (argc > 1) {
        std::string arg = argv[1];
        if (arg == "-q") {
            if (argc > 2) {
                std::string query_arg = argv[2];
                if (query_arg == "countmin" || query_arg == "sinnamon" || query_arg == "jl" || query_arg == "rabitq") {
                    quant_type = query_arg;
                } else {
                    std::cout << "Invalid query type" << std::endl;
                    return 1;
                }
            } else {
                std::cout << "Missing query type argument" << std::endl;
                return 1;
            }
        }
    }

    long k = 10;
    auto BP = BuildParams(
        64, // max degree
        256, // beam size
        1.2, // alpha
        1, // number of passes
        0, // number of clusters
        0, // cluster size
        0, // MST degree
        0, // delta
        false, // verbose
        false, // quantize build
        0.0, // radius
        0.0, // radius 2
        false, // self
        false, // range
        0 // single batch
    );

    char pr_file[] = "data/base_small.csr";
    auto inserts = forward_index<value_type>(pr_file, "csr");

    char q_file[] = "data/queries.dev.csr";
    auto queries = forward_index<value_type>(q_file, "csr");

    if (inserts.dims != queries.dims) {
        std::cout << "Dimensionality of inserts and queries must match: " << inserts.dims << " != " << queries.dims << std::endl;
        return 1;
    }

    auto PR = fi_to_spr<value_type, point_type>(inserts);
    auto Q = fi_to_spr<value_type, point_type>(queries);

    char gt_file[] = "data/base_small.dev.gt";
    auto GT = groundTruth<index_type>(gt_file);

    char g_file[] = "data/graph/base_small_64_64";
    auto G = Graph<index_type>(g_file);

    char r_file[] = "res_output.txt";

    point_range_type QPR;
    point_range_type QQ;
    if (quant_type == "countmin") {
        auto sketch = count_min_sketch(inserts.dims, 400);
        forward_index<value_type> qinserts, qqueries;
        qinserts.dims = qqueries.dims = 400;
        qinserts.points = parlay::tabulate(inserts.size(), [&] (size_t i) {
            auto qvec = sketch.transform_csr_to_qvec<value_type>(inserts[i]);
            return sketch.transform_qvec_to_qcsr<value_type>(qvec);
        });
        qqueries.points = parlay::tabulate(queries.size(), [&] (size_t i) {
            auto qvec = sketch.transform_csr_to_qvec<value_type>(queries[i]);
            return sketch.transform_qvec_to_qcsr<value_type>(qvec);
        });
        QPR = fi_to_spr<value_type, point_type>(qinserts);
        QQ = fi_to_spr<value_type, point_type>(qqueries);
    }
    else if (quant_type == "sinnamon") {
        auto sketch = sinnamon_sketch(inserts.dims, 400);
        forward_index<value_type> qinserts, qqueries;
        qinserts.dims = qqueries.dims = 400;
        qinserts.points = parlay::tabulate(inserts.size(), [&] (size_t i) {
            auto qvec = sketch.transform_csr_to_qvec<value_type>(inserts[i]);
            return sketch.transform_qvec_to_qcsr<value_type>(qvec);
        });
        qqueries.points = parlay::tabulate(queries.size(), [&] (size_t i) {
            auto qvec = sketch.transform_csr_to_qvec<value_type>(queries[i]);
            return sketch.transform_qvec_to_qcsr<value_type>(qvec);
        });
        QPR = fi_to_spr<value_type, point_type>(qinserts);
        QQ = fi_to_spr<value_type, point_type>(qqueries);
    }
    else if (quant_type == "jl") {
        auto projection_matrix = generate_random_projection_matrix(inserts.dims, 400);
        forward_index<value_type> qinserts, qqueries;
        qinserts.dims = qqueries.dims = 400;
        qinserts.points = parlay::tabulate(queries.size(), [&] (size_t i) {
            auto insert_vec = csr_to_vec(queries[i], queries.dims);
            auto jl_transform = apply_jl_transform(insert_vec, projection_matrix);
            return vec_to_csr(jl_transform);
        });
        qqueries.points = parlay::tabulate(inserts.size(), [&] (size_t i) {
            auto insert_vec = csr_to_vec(inserts[i], inserts.dims);
            auto jl_transform = apply_jl_transform(insert_vec, projection_matrix);
            return vec_to_csr(jl_transform);
        });
        QPR = fi_to_spr<value_type, point_type>(qinserts);
        QQ = fi_to_spr<value_type, point_type>(qqueries);
    }
    else if (quant_type == "rabitq") {
        QQ = Q;
        QPR = PR;
    }
    else {
        QQ = Q;
        QPR = PR;
    }
    
    ANN_<point_type, point_range_type, point_range_type, index_type>(G, k, BP, Q, QQ, GT, r_file, true, PR, QPR);
 
    return 0;
}
