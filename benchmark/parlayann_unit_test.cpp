#include <cstdint>
#include <iostream>
#include <algorithm>

#include <parlay/parallel.h>
#include <parlay/primitives.h>

#include "utils/types.h"
#include "utils/mips_point.h"
#include "utils/point_range.h"
#include "utils/graph.h"
#include "vamana/neighbors.h"

using index_type = uint32_t;
using value_type = float;
using point_type = Mips_Point<value_type>;
using point_range_type = PointRange<value_type, point_type>; // TODO: change both of these to Sparse versions

int main(int argc, char* argv[]) {
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
    auto PR = point_range_type(pr_file);

    char q_file[] = "data/queries.dev.csr";
    auto Q = point_range_type(q_file);

    char gt_file[] = "data/base_small.dev.gt";
    auto GT = groundTruth<index_type>(gt_file);

    char g_file[] = "data/graph/base_small_64_64";
    auto G = Graph<index_type>(g_file);

    char r_file[] = "res_output.txt";

    ANN_<point_type, point_range_type, point_range_type, index_type>(G, k, BP, Q, Q, GT, r_file, true, PR, PR);
 
    return 0;
}
