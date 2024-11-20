#include <cstdint>
#include <cstring>
#include <iostream>
#include <algorithm>
#include <cassert>
#include <random>
#include <unordered_set>

#include <parlay/parallel.h>
#include <parlay/primitives.h>

#include "forward_index.h"
#include "point_range_util.h"
#include "vector_utils.h"

using index_type = uint32_t;
using value_type = float;

int main(int argc, char* argv[]) {
    size_t num_vectors = 5000, dims = 30109, non_zero_values = 175;
    forward_index<value_type> fwd_index(dims);

    // Random number generators
    std::mt19937 rng(std::random_device{}()); // Seeded with random device
    std::uniform_real_distribution<value_type> val_dist(0.0, 1.0); // Random float values between 0 and 1
    std::uniform_int_distribution<uint32_t> dim_dist(0, dims - 1); // Random indices between 0 and dims-1

    // For each vector
    for (uint32_t i = 0; i < num_vectors; ++i) {
        // Use a set to ensure unique random indices
        std::unordered_set<uint32_t> selected_indices;
        parlay::sequence<std::pair<uint32_t, value_type>> point;

        while (selected_indices.size() < non_zero_values) {
            // Generate a random dimension and a random value
            uint32_t index = dim_dist(rng);
            if (selected_indices.find(index) == selected_indices.end()) {
                selected_indices.insert(index);
                value_type value = val_dist(rng);
                std::cout << value << std::endl;
                point.push_back(std::make_pair(index, value)); // Add the (index, value) pair
            }
        }

        // Add the generated point to the forward_index
        fwd_index.points.push_back(point);
    }
    
    fwd_index.write_to_file("random_queries_1.csr", "csr");
 
    return 0;
}
